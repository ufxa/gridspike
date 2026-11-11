"""Total monitoring harness for the FedGrid Vast fleet -- "map absolutely everything".

Every cycle (~30-60 s) :class:`FleetMonitor` merges three data sources into one
machine-readable map (``runs/fleet_status.json``) and one compact human line:

* the Vast backend (``show instances`` -> status, GPU model, $/hr, boot time) ,
* each worker's pulled self-report (``runs/status/<id>.json`` -> VRAM, utilisation,
  current run, runs completed, per-run wall-times, heartbeat) , and
* the canonical result store (``runs/<run_id>.json``) diffed against the manifest.

From that it surfaces exactly what the user asked to never miss:

* **per-instance** health, GPU/VRAM, utilisation, current run, heartbeat age, $/hr
  and accrued $;
* **deceiving/slow hosts** -- a host whose per-run wall-time exceeds ``slow_factor``x
  the fleet median is flagged for destroy-and-replace, and a host reporting the
  wrong GPU is flagged ``gpu_mismatch``;
* **stalled** hosts (no fresh result within ``stall_minutes``) and **dead** hosts
  (heartbeat older than ``dead_grace_s`` or an offline status) -- the triggers for
  gap-repair + reprovision;
* **money** -- accrued, live projection, the **hard $30 cap** (crossing it freezes
  provisioning and prints a loud STATUS line -- never exceeded) and the Vast
  account credit read every cycle;
* **progress** -- N/total runs, ETA from measured throughput, per-block completion.

Run standalone: ``python -m experiments.fleet.monitor`` prints one line per cycle.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.fleet.fleet import (
    Fleet,
    Instance,
    VastBackend,
    burn_rate_usd_per_hr,
    eta_hours,
    parse_instance,
    project_total_spend,
    total_accrued_usd,
)

log = logging.getLogger("fedgrid.fleet.monitor")

STATUS_SCHEMA_VERSION = 1
STATUS_FILENAME = "fleet_status.json"


@dataclass
class MonitorConfig:
    """Monitoring thresholds (defaults honour CONTRACT amendment h + user asks)."""

    target_size: int = 20
    spend_cap_usd: float = 30.0
    #: A per-run wall-time above ``slow_factor`` x fleet median => "slow/deceiving".
    slow_factor: float = 2.0
    #: No fresh result within this many minutes while work remains => "stalled".
    stall_minutes: float = 12.0
    #: Heartbeat age beyond this => the worker is not reporting.
    heartbeat_grace_s: float = 120.0
    #: Heartbeat age beyond this (or an offline status) => "dead" (reprovision).
    dead_grace_s: float = 180.0
    #: Need at least this many reporting workers before the median rule fires.
    min_median_samples: int = 3
    gpu_match_token: str = "3060"
    min_gpu_ram_gb: float = 11.0
    runs_total: int | None = None


@dataclass(frozen=True)
class WorkerReport:
    """A worker's self-report, written by ``run_shard.sh`` and pulled by ``collect``."""

    instance_id: int
    gpu_name: str = ""
    vram_total_mb: float = 0.0
    vram_used_mb: float = 0.0
    vram_free_mb: float = 0.0
    gpu_util_pct: float = 0.0
    dataset_sha_ok: bool = False
    current_run_id: str | None = None
    runs_completed: int = 0
    shard_index: int | None = None
    n_shards: int | None = None
    heartbeat_ts: float | None = None
    last_run_walltime_s: float | None = None
    mean_run_walltime_s: float | None = None
    last_run_finished_ts: float | None = None
    phase: str = "unknown"

    @property
    def representative_walltime_s(self) -> float | None:
        """The number the slow-host rule compares to the fleet median."""
        return self.mean_run_walltime_s or self.last_run_walltime_s


def _num(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def parse_report(row: Mapping[str, Any]) -> WorkerReport:
    return WorkerReport(
        instance_id=int(row.get("instance_id", 0) or 0),
        gpu_name=str(row.get("gpu_name", "") or ""),
        vram_total_mb=_num(row.get("vram_total_mb")),
        vram_used_mb=_num(row.get("vram_used_mb")),
        vram_free_mb=_num(row.get("vram_free_mb")),
        gpu_util_pct=_num(row.get("gpu_util_pct")),
        dataset_sha_ok=bool(row.get("dataset_sha_ok", False)),
        current_run_id=row.get("current_run_id"),
        runs_completed=int(row.get("runs_completed", 0) or 0),
        shard_index=row.get("shard_index"),
        n_shards=row.get("n_shards"),
        heartbeat_ts=row.get("heartbeat_ts"),
        last_run_walltime_s=row.get("last_run_walltime_s"),
        mean_run_walltime_s=row.get("mean_run_walltime_s"),
        last_run_finished_ts=row.get("last_run_finished_ts"),
        phase=str(row.get("phase", "unknown")),
    )


def load_reports(status_dir: str | Path) -> dict[int, WorkerReport]:
    """Read every ``runs/status/<id>.json`` self-report (tolerant of junk files)."""
    out: dict[int, WorkerReport] = {}
    d = Path(status_dir)
    if not d.is_dir():
        return out
    for p in d.glob("*.json"):
        try:
            rep = parse_report(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        iid = rep.instance_id or int(p.stem) if p.stem.isdigit() else rep.instance_id
        out[iid] = rep
    return out


def block_completion(
    manifest: Iterable[Mapping[str, Any]], collected: set[str]
) -> dict[str, dict[str, int]]:
    """Per-``block`` ``{done, total}`` from the manifest vs the collected result set."""
    blocks: dict[str, dict[str, int]] = {}
    for line in manifest:
        block = str(line.get("block", "?"))
        rid = str(line["run_id"])
        b = blocks.setdefault(block, {"done": 0, "total": 0})
        b["total"] += 1
        if rid in collected:
            b["done"] += 1
    return blocks


def flag_slow_instances(
    walltimes: Mapping[int, float], *, slow_factor: float, min_samples: int
) -> tuple[list[int], float | None]:
    """Return (slow instance ids, fleet median). Fires only with enough samples."""
    values = [v for v in walltimes.values() if v and v > 0]
    if len(values) < min_samples:
        return [], (statistics.median(values) if values else None)
    median = statistics.median(values)
    if median <= 0:
        return [], median
    slow = [iid for iid, v in walltimes.items() if v and v > slow_factor * median]
    return sorted(slow), median


class FleetMonitor:
    """The observability engine. Pure core (:meth:`build_status`) + IO shell."""

    def __init__(
        self,
        manifest: Sequence[Mapping[str, Any]],
        runs_dir: str | Path,
        *,
        backend: VastBackend | None = None,
        config: MonitorConfig | None = None,
        fleet: Fleet | None = None,
        clock: Any = time.time,
    ) -> None:
        self.manifest = list(manifest)
        self.runs_dir = Path(runs_dir)
        self.backend = backend
        self.config = config or MonitorConfig()
        self.fleet = fleet
        self._clock = clock
        self.status_dir = self.runs_dir / "status"
        self.runs_total = self.config.runs_total or len(self.manifest)
        self._start_ts: float | None = None
        self._start_done: int | None = None

    # -- pure core ------------------------------------------------------------
    def build_status(
        self,
        instances: Sequence[Instance],
        reports: Mapping[int, WorkerReport],
        collected: set[str],
        *,
        now: float,
        credit: float | None,
        throughput_per_hr: float,
    ) -> dict[str, Any]:
        """Compute the full fleet map from already-gathered inputs (no IO).

        This is the schema written to ``runs/fleet_status.json`` and the one the
        tests assert against.
        """
        cfg = self.config
        by_id = {i.id: i for i in instances}
        all_ids = sorted(set(by_id) | set(reports))

        instance_views: list[dict[str, Any]] = []
        walltimes: dict[int, float] = {}
        alive = loading = offline = 0
        gpu_mismatch: list[int] = []

        for iid in all_ids:
            inst = by_id.get(iid)
            rep = reports.get(iid)
            hb_age = (now - rep.heartbeat_ts) if (rep and rep.heartbeat_ts) else None
            status = self._instance_status(inst, hb_age)
            if status == "running":
                alive += 1
            elif status == "loading":
                loading += 1
            else:
                offline += 1

            rep_wall = rep.representative_walltime_s if rep else None
            if rep_wall:
                walltimes[iid] = rep_wall

            gpu_name = (rep.gpu_name if rep and rep.gpu_name else (inst.gpu_name if inst else ""))
            mismatch = self._is_gpu_mismatch(inst, rep)
            if mismatch:
                gpu_mismatch.append(iid)

            last_finish = rep.last_run_finished_ts if rep else None
            stalled = self._is_stalled(status, last_finish, now)
            dead = self._is_dead(inst, status, hb_age)

            flags: list[str] = []
            if mismatch:
                flags.append("gpu_mismatch")
            if stalled:
                flags.append("stalled")
            if dead:
                flags.append("dead")

            instance_views.append(
                {
                    "instance_id": iid,
                    "status": status,
                    "gpu_name": gpu_name,
                    "gpu_ok": not mismatch,
                    "vram_total_mb": rep.vram_total_mb if rep else None,
                    "vram_used_mb": rep.vram_used_mb if rep else None,
                    "vram_free_mb": rep.vram_free_mb if rep else None,
                    "gpu_util_pct": rep.gpu_util_pct if rep else None,
                    "dataset_sha_ok": rep.dataset_sha_ok if rep else None,
                    "current_run_id": rep.current_run_id if rep else None,
                    "runs_completed": rep.runs_completed if rep else 0,
                    "shard_index": rep.shard_index if rep else None,
                    "n_shards": rep.n_shards if rep else None,
                    "last_heartbeat_age_s": round(hb_age, 1) if hb_age is not None else None,
                    "last_run_walltime_s": rep.last_run_walltime_s if rep else None,
                    "mean_run_walltime_s": rep.mean_run_walltime_s if rep else None,
                    "dph": inst.dph_total if inst else None,
                    "accrued_usd": round(inst.accrued_usd(now), 4) if inst else None,
                    "flags": flags,
                }
            )

        slow, median = flag_slow_instances(
            walltimes, slow_factor=cfg.slow_factor, min_samples=cfg.min_median_samples
        )
        for view in instance_views:
            if view["instance_id"] in slow:
                view["flags"].append("slow")

        stalled_hosts = [v["instance_id"] for v in instance_views if "stalled" in v["flags"]]
        dead_hosts = [v["instance_id"] for v in instance_views if "dead" in v["flags"]]

        runs_done = len([1 for line in self.manifest if str(line["run_id"]) in collected])
        eta_h = eta_hours(runs_done, self.runs_total, throughput_per_hr)

        accrued = total_accrued_usd(instances, now)
        rate = burn_rate_usd_per_hr(instances)
        projected = project_total_spend(accrued, rate, eta_h if eta_h is not None else 0.0)
        cap = cfg.spend_cap_usd
        cap_breached = accrued >= cap or projected >= cap
        if cap_breached and self.fleet is not None:
            self.fleet.provisioning_frozen = True

        alerts: list[str] = []
        if cap_breached:
            alerts.append(
                f"SPEND-CAP: accrued ${accrued:.2f} / projected ${projected:.2f} vs "
                f"cap ${cap:.2f} -- provisioning frozen."
            )
        if gpu_mismatch:
            alerts.append(f"DECEIVING-HOST(gpu): instances {gpu_mismatch} report wrong GPU.")
        if slow:
            alerts.append(f"SLOW-HOST: instances {slow} exceed {cfg.slow_factor}x median.")
        if dead_hosts:
            alerts.append(f"DEAD: instances {dead_hosts} -> gap_repair + reprovision.")

        return {
            "schema_version": STATUS_SCHEMA_VERSION,
            "ts": now,
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            "target_fleet_size": cfg.target_size,
            "instances": instance_views,
            "fleet": {
                "alive": alive,
                "loading": loading,
                "offline": offline,
                "target": cfg.target_size,
                "runs_done": runs_done,
                "runs_total": self.runs_total,
                "progress_pct": round(100.0 * runs_done / max(1, self.runs_total), 1),
                "median_run_walltime_s": round(median, 1) if median else None,
                "throughput_runs_per_hr": round(throughput_per_hr, 1),
                "eta_hours": round(eta_h, 2) if eta_h is not None else None,
                "slow_hosts": slow,
                "stalled_hosts": stalled_hosts,
                "dead_hosts": dead_hosts,
                "gpu_mismatch_hosts": gpu_mismatch,
            },
            "money": {
                "accrued_usd": round(accrued, 4),
                "rate_usd_per_hr": round(rate, 4),
                "projected_total_usd": round(projected, 4),
                "cap_usd": cap,
                "cap_breached": cap_breached,
                "provisioning_frozen": cap_breached,
                "account_credit_usd": credit,
            },
            "blocks": block_completion(self.manifest, collected),
            "alerts": alerts,
        }

    # -- status sub-rules -----------------------------------------------------
    def _instance_status(self, inst: Instance | None, hb_age: float | None) -> str:
        if inst is not None:
            if inst.is_offline:
                return "offline"
            if inst.is_running:
                # A "running" box that stopped heart-beating is effectively offline.
                if hb_age is not None and hb_age > self.config.dead_grace_s:
                    return "offline"
                return "running"
            return "loading"
        # No backend row: infer from the heartbeat alone.
        if hb_age is None:
            return "offline"
        return "running" if hb_age <= self.config.dead_grace_s else "offline"

    def _is_gpu_mismatch(self, inst: Instance | None, rep: WorkerReport | None) -> bool:
        token = self.config.gpu_match_token
        # The worker's own nvidia-smi report is the ground truth against deceit.
        if rep is not None and rep.gpu_name:
            if token not in rep.gpu_name.replace("_", " "):
                return True
            if rep.vram_total_mb and rep.vram_total_mb < self.config.min_gpu_ram_gb * 1024:
                return True
        return False

    def _is_stalled(self, status: str, last_finish: float | None, now: float) -> bool:
        if status != "running":
            return False
        if last_finish is None:
            return False  # never finished a run yet -> "loading/first run", not stalled
        return (now - last_finish) > self.config.stall_minutes * 60.0

    def _is_dead(self, inst: Instance | None, status: str, hb_age: float | None) -> bool:
        if inst is not None and inst.is_offline:
            return True
        if hb_age is not None and hb_age > self.config.dead_grace_s:
            return True
        return status == "offline" and inst is not None
    # -- IO shell -------------------------------------------------------------

    def _gather_instances(self) -> list[Instance]:
        if self.backend is None:
            return []
        try:
            rows = self.backend.show_instances(label=None)
        except RuntimeError as exc:
            log.warning("monitor: show_instances failed: %s", exc)
            return []
        return [parse_instance(r) for r in rows]

    def _gather_credit(self) -> float | None:
        if self.backend is None:
            return None
        try:
            return self.backend.account_credit()
        except RuntimeError as exc:
            log.info("monitor: account_credit unavailable: %s", exc)
            return None

    def _collected(self) -> set[str]:
        if not self.runs_dir.is_dir():
            return set()
        return {p.stem for p in self.runs_dir.glob("*.json") if not p.name.startswith("fleet_")}

    def _throughput_per_hr(self, runs_done: int, now: float) -> float:
        """Measured throughput since this monitor process started."""
        if self._start_ts is None:
            self._start_ts, self._start_done = now, runs_done
            return 0.0
        elapsed = now - self._start_ts
        if elapsed <= 0 or self._start_done is None:
            return 0.0
        return max(0, runs_done - self._start_done) / (elapsed / 3600.0)

    def sample(self) -> dict[str, Any]:
        """Gather everything for one cycle and return the full status dict."""
        now = self._clock()
        instances = self._gather_instances()
        reports = load_reports(self.status_dir)
        collected = self._collected()
        runs_done = len([1 for line in self.manifest if str(line["run_id"]) in collected])
        throughput = self._throughput_per_hr(runs_done, now)
        credit = self._gather_credit()
        return self.build_status(
            instances, reports, collected,
            now=now, credit=credit, throughput_per_hr=throughput,
        )

    def write_status(self, status: Mapping[str, Any]) -> Path:
        """Atomically write ``runs/fleet_status.json`` (the full map)."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        path = self.runs_dir / STATUS_FILENAME
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        return path

    @staticmethod
    def compact_line(status: Mapping[str, Any]) -> str:
        """One dense line per cycle for the Monitor tool to tail."""
        f, m = status["fleet"], status["money"]
        cred = f"{m['account_credit_usd']:.2f}" if m["account_credit_usd"] is not None else "?"
        eta = f"{f['eta_hours']:.1f}h" if f["eta_hours"] is not None else "?"
        med = f"{f['median_run_walltime_s']:.0f}s" if f["median_run_walltime_s"] else "?"
        prefix = "!! SPEND-CAP !! " if m["cap_breached"] else ""
        return (
            f"{prefix}[{status['iso']}] alive {f['alive']}/{f['target']} "
            f"(load{f['loading']} off{f['offline']}) | runs {f['runs_done']}/{f['runs_total']} "
            f"{f['progress_pct']}% | ${m['accrued_usd']:.2f}/${m['cap_usd']:.0f} "
            f"proj${m['projected_total_usd']:.2f} cred${cred} | ETA {eta} "
            f"thr {f['throughput_runs_per_hr']:.0f}/h | slow{len(f['slow_hosts'])} "
            f"stall{len(f['stalled_hosts'])} dead{len(f['dead_hosts'])} "
            f"mism{len(f['gpu_mismatch_hosts'])} | med {med}"
        )

    def run_cycle(self) -> str:
        """One full cycle: sample -> write map -> return (and log) the compact line."""
        status = self.sample()
        self.write_status(status)
        line = self.compact_line(status)
        for alert in status["alerts"]:
            log.warning(alert)
        return line

    def run_forever(self, interval_s: float = 45.0, *, max_cycles: int | None = None) -> None:
        cycles = 0
        while max_cycles is None or cycles < max_cycles:
            cycles += 1
            print(self.run_cycle(), flush=True)  # noqa: T201 -- the tailed status line
            time.sleep(interval_s)


# --------------------------------------------------------------------------- #
# CLI entrypoint: python -m experiments.fleet.monitor
# --------------------------------------------------------------------------- #


def _load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FedGrid Vast fleet monitor.")
    parser.add_argument("--runs-dir", default="runs", type=Path)
    parser.add_argument("--manifest", default="experiments/manifest.jsonl", type=Path)
    parser.add_argument("--interval", default=45.0, type=float)
    parser.add_argument("--target-size", default=20, type=int)
    parser.add_argument("--spend-cap", default=30.0, type=float)
    parser.add_argument("--once", action="store_true", help="one cycle then exit")
    parser.add_argument(
        "--no-backend", action="store_true",
        help="observe local dirs only (no Vast calls; no key needed)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    backend: VastBackend | None = None
    if not args.no_backend:
        from experiments.fleet.fleet import CliVastBackend

        backend = CliVastBackend()

    cfg = MonitorConfig(target_size=args.target_size, spend_cap_usd=args.spend_cap)
    monitor = FleetMonitor(
        _load_manifest(args.manifest), args.runs_dir, backend=backend, config=cfg
    )
    if args.once:
        print(monitor.run_cycle(), flush=True)  # noqa: T201
    else:
        monitor.run_forever(args.interval)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
