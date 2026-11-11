"""Unit tests for the Vast fleet controller + monitor.

Everything runs against an in-memory :class:`MockVastBackend` -- there is **no real
API call and no money spent**. The mock models the whole instance lifecycle (offers
-> create -> run/produce results -> pull -> destroy) so the controller's loops can
be driven to convergence deterministically with a frozen clock.

Coverage maps to the user's asks:
* striding covers every manifest line exactly once  (nothing lost)
* gap_repair re-dispatches exactly the missing ids and converges  (auto-resurrect)
* the $30 cap refuses to provision past the limit  (spend cap, never exceeded)
* GPU-mismatch / underdelivering hosts are marked + replaced  (deceiving hosts)
* the monitor's slow-host rule flags a planted 3x outlier  (deceiving hosts)
* fleet_status.json carries the full schema  (map everything)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

# `experiments` lives at the repo root (not in the installed `src` tree); make sure
# it resolves regardless of how pytest was invoked.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.fleet import fleet as F  # noqa: E402
from experiments.fleet import monitor as M  # noqa: E402

NOW = 1_700_000_000.0


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def make_manifest(n: int) -> list[dict]:
    """A synthetic manifest with Runner's fields (run_id, arm, ..., block, ...)."""
    out = []
    for i in range(n):
        out.append(
            {
                "run_id": f"r{i:03d}",
                "arm": f"B{i % 12 + 1}",
                "model": "snn" if i % 2 else "cnn_lstm",
                "seed": i % 5,
                "tier": 1,
                "block": "T1a" if i < n // 2 else "T1b",
                "config": {"lr": 1e-3},
                "config_hash": f"h{i:03d}",
            }
        )
    return out


def offer_row(oid: int, dph: float = 0.05, *, gpu_name: str = "RTX 3060",
              reliability: float = 0.99, verified: bool = True, cpu: float = 8.0,
              inet_down: float = 500.0, up_cost: float = 0.002, down_cost: float = 0.002,
              cuda: float = 12.4, num_gpus: int = 1, gpu_ram: float = 12.0,
              deceive: bool = False) -> dict:
    return {
        "id": oid, "dph_total": dph, "gpu_name": gpu_name, "num_gpus": num_gpus,
        "gpu_ram": gpu_ram, "reliability": reliability, "cpu_cores_effective": cpu,
        "inet_down": inet_down, "inet_up_cost": up_cost, "inet_down_cost": down_cost,
        "cuda_max_good": cuda, "verified": verified, "geolocation": "US",
        "machine_id": oid, "_deceive": deceive,
    }


def instance_row(iid: int, *, label: str = "fedgrid-fleet", gpu_name: str = "RTX 3060",
                 gpu_ram: float = 12.0, actual_status: str = "running",
                 dph_total: float = 0.05, start_date: float = NOW - 60.0,
                 cur_state: str = "running") -> dict:
    return {
        "id": iid, "label": label, "actual_status": actual_status, "gpu_name": gpu_name,
        "num_gpus": 1, "gpu_ram": gpu_ram, "dph_total": dph_total,
        "start_date": start_date, "machine_id": iid * 7, "cur_state": cur_state,
        "status_msg": "",
    }


def minimal_runresult(run_id: str, seed: int = 0) -> dict:
    return {
        "method": "fedavg", "seed": seed, "config": {"run_id": run_id}, "rounds": [],
        "final": {"acc": 0.9}, "energy": None, "latency": None,
        "git_commit": "deadbeef", "env_fingerprint": "test",
    }


class MockVastBackend:
    """In-memory Vast backend: offers, instance lifecycle, produce/pull results."""

    def __init__(self, offers=None, manifest_run_ids=None, credit: float = 97.98) -> None:
        self.offers = list(offers or [])
        self._manifest_ids = list(manifest_run_ids or [])
        self.credit = credit
        self.instances: dict[int, dict] = {}
        self._produced: dict[int, set[str]] = {}
        self._next = 1000
        self.blocked: set[str] = set()  # ids a stride "fails" to produce (fallen worker)
        self.auto_complete = True
        self.create_calls: list[int] = []
        self.destroy_calls: list[int] = []
        self.execute_calls: list[tuple[int, str]] = []
        self.copy_calls: list[tuple[str, str]] = []

    # -- protocol -------------------------------------------------------------
    def search_offers(self, query: str, *, order: str = "dph_total") -> list[dict]:
        return list(self.offers)

    def create_instance(self, offer_id, *, image, disk_gb, label, onstart_cmd, env) -> int:
        iid = self._next
        self._next += 1
        self.create_calls.append(offer_id)
        row = instance_row(iid, label=label, start_date=NOW - 60.0)
        meta = next((o for o in self.offers if o.get("id") == offer_id), {})
        if meta.get("_deceive"):
            row["gpu_name"], row["gpu_ram"] = "GTX 1660", 6.0  # the deceiving host
        self.instances[iid] = row
        self._produced.setdefault(iid, set())
        return iid

    def show_instances(self, *, label=None) -> list[dict]:
        rows = list(self.instances.values())
        if label is not None:
            rows = [r for r in rows if r.get("label") == label]
        return rows

    def destroy_instance(self, instance_id: int) -> None:
        self.destroy_calls.append(instance_id)
        self.instances.pop(instance_id, None)
        self._produced.pop(instance_id, None)

    def execute(self, instance_id: int, command: str) -> str:
        self.execute_calls.append((instance_id, command))
        if self.auto_complete:
            self._simulate(instance_id, command)
        return ""

    def copy(self, src: str, dst: str) -> None:
        self.copy_calls.append((src, dst))
        m = re.match(r"(\d+):.*results", src)
        if m:
            iid = int(m.group(1))
            d = Path(dst)
            d.mkdir(parents=True, exist_ok=True)
            for rid in sorted(self._produced.get(iid, set())):
                (d / f"{rid}.json").write_text(json.dumps(minimal_runresult(rid)), "utf-8")
            return
        m2 = re.match(r"(\d+):.*self\.json", src)
        if m2:
            iid = int(m2.group(1))
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            Path(dst).write_text(
                json.dumps(
                    {
                        "instance_id": iid, "gpu_name": "RTX 3060",
                        "runs_completed": len(self._produced.get(iid, set())),
                        "heartbeat_ts": NOW,
                    }
                ),
                "utf-8",
            )

    def account_credit(self) -> float | None:
        return self.credit

    # -- simulation -----------------------------------------------------------
    def _simulate(self, instance_id: int, command: str) -> None:
        produced = self._produced.setdefault(instance_id, set())
        m = re.search(r"run_shard\.sh\s+(\d+)\s+(\d+)", command)
        if m:  # static stride start
            i, n = int(m.group(1)), int(m.group(2))
            produced.update(r for r in self._manifest_ids[i::n] if r not in self.blocked)
        for rid in re.findall(r'"run_id":\s*"([^"]+)"', command):  # explicit gap repair
            produced.add(rid)

    def produce(self, instance_id: int, run_ids) -> None:
        self._produced.setdefault(instance_id, set()).update(run_ids)


def make_fleet(tmp_path: Path, manifest, *, backend=None, config=None) -> F.Fleet:
    be = backend or MockVastBackend(
        offers=[offer_row(i, dph=0.04 + 0.001 * i) for i in range(1, 9)],
        manifest_run_ids=[ln["run_id"] for ln in manifest],
    )
    cfg = config or F.FleetConfig(target_size=4, over_provision=2)
    return F.Fleet(
        be, manifest, runs_dir=tmp_path / "runs", bundle_dir=tmp_path / "bundle",
        config=cfg, clock=lambda: NOW,
    )


# --------------------------------------------------------------------------- #
# 1. Striding covers every manifest line exactly once
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_runs,n_shards", [(10, 1), (12, 4), (13, 5), (540, 20), (7, 20)])
def test_striding_partitions_manifest_exactly_once(n_runs, n_shards):
    run_ids = [f"r{i:03d}" for i in range(n_runs)]
    shards = F.assign_shards(run_ids, n_shards)
    assert len(shards) == n_shards
    union = [rid for shard in shards for rid in shard]
    assert sorted(union) == sorted(run_ids)  # total coverage
    assert len(union) == len(run_ids)  # no duplication
    for i, shard in enumerate(shards):
        assert shard == F.stride_shard(run_ids, i, n_shards)  # recomputable


def test_stride_shard_rejects_bad_index():
    with pytest.raises(ValueError):
        F.stride_shard(["a", "b"], 3, 2)


# --------------------------------------------------------------------------- #
# 2. Offer search: filter + rank
# --------------------------------------------------------------------------- #


def test_search_offers_filters_and_ranks(tmp_path):
    offers = [
        offer_row(1, dph=0.07),                       # ok, mid price
        offer_row(2, dph=0.05),                       # ok, cheapest -> first
        offer_row(3, dph=0.04, gpu_name="RTX 4090"),  # wrong GPU -> drop
        offer_row(4, dph=0.03, reliability=0.80),     # unreliable -> drop
        offer_row(5, dph=0.02, down_cost=0.05),       # egress outlier -> drop
        offer_row(6, dph=0.20),                        # over price cap -> drop
        offer_row(7, dph=0.06),                       # ok
    ]
    be = MockVastBackend(offers=offers)
    manifest = make_manifest(4)
    fl = F.Fleet(be, manifest, runs_dir=tmp_path / "runs",
                 config=F.FleetConfig(target_size=2, over_provision=1), clock=lambda: NOW)
    picked = fl.search_offers()
    assert [o.id for o in picked] == [2, 7, 1]  # cheapest-first, bad ones gone
    assert all(o.gpu_name == "RTX 3060" and o.dph_total <= 0.08 for o in picked)


def test_build_search_query_encodes_filters():
    q = F.build_search_query(F.FleetConfig())
    for token in ["gpu_name=RTX_3060", "num_gpus=1", "verified=true",
                  "reliability>=0.95", "cpu_cores_effective>=4", "inet_down>=200",
                  "disk_space>=32", "dph_total<=0.08"]:
        assert token in q


# --------------------------------------------------------------------------- #
# 3. Provisioning: idempotent, cap-guarded, deceiving-host replacement
# --------------------------------------------------------------------------- #


def test_provision_is_idempotent_never_double_creates(tmp_path):
    manifest = make_manifest(8)
    fl = make_fleet(tmp_path, manifest)
    fl.provision()
    first = len(fl.backend.create_calls)
    assert first == 4  # target_size
    fl.provision()  # already at target -> creates nothing more
    assert len(fl.backend.create_calls) == first


def test_spend_cap_refuses_provisioning(tmp_path):
    manifest = make_manifest(8)
    be = MockVastBackend(
        offers=[offer_row(i) for i in range(1, 9)],
        manifest_run_ids=[ln["run_id"] for ln in manifest],
    )
    # Ten already-running instances that have each accrued ~$3.50 (70 h @ $0.05) ->
    # ~$35 total, already over the $30 cap.
    for k in range(10):
        be.instances[500 + k] = instance_row(500 + k, dph_total=0.05,
                                              start_date=NOW - 70 * 3600.0)
    fl = F.Fleet(be, manifest, runs_dir=tmp_path / "runs",
                 config=F.FleetConfig(target_size=20), clock=lambda: NOW)
    accrued, _ = fl.spend_state()
    assert accrued > 30.0
    result = fl.provision()
    assert be.create_calls == []            # refused: nothing created
    assert fl.provisioning_frozen is True
    assert len(result) == 10                # untouched


def test_spend_cap_blocks_on_projection(tmp_path):
    manifest = make_manifest(4)
    be = MockVastBackend(offers=[offer_row(i) for i in range(1, 9)],
                         manifest_run_ids=[ln["run_id"] for ln in manifest])
    # Tiny cap so the *projection* (n_new x price_cap x horizon) blows it, even at $0 accrued.
    fl = F.Fleet(be, manifest, runs_dir=tmp_path / "runs",
                 config=F.FleetConfig(target_size=20, spend_cap_usd=2.0), clock=lambda: NOW)
    fl.provision()
    assert be.create_calls == []
    assert fl.provisioning_frozen is True


def test_gpu_mismatch_classified_and_replaced(tmp_path):
    # classify_instance is the pure rule the provisioner and monitor share.
    cfg = F.FleetConfig()
    ok = F.parse_instance(instance_row(1))
    wrong_gpu = F.parse_instance(instance_row(2, gpu_name="GTX 1660"))
    small_vram = F.parse_instance(instance_row(3, gpu_ram=6.0))
    booting = F.parse_instance(instance_row(4, actual_status="loading", cur_state="loading"))
    assert F.classify_instance(ok, cfg) == "ok"
    assert F.classify_instance(wrong_gpu, cfg) == "mismatch"
    assert F.classify_instance(small_vram, cfg) == "mismatch"
    assert F.classify_instance(booting, cfg) == "pending"

    # End-to-end: the cheapest offer is a deceiver; provision must destroy + replace it.
    manifest = make_manifest(6)
    offers = [offer_row(1, dph=0.04, deceive=True), offer_row(2, dph=0.05),
              offer_row(3, dph=0.06), offer_row(4, dph=0.07)]
    be = MockVastBackend(offers=offers, manifest_run_ids=[ln["run_id"] for ln in manifest])
    fl = F.Fleet(be, manifest, runs_dir=tmp_path / "runs",
                 config=F.FleetConfig(target_size=2, over_provision=2), clock=lambda: NOW)
    fl.provision()
    live = fl.live_instances()
    assert all(F.classify_instance(i, fl.config) == "ok" for i in live)
    assert len(live) == 2
    assert be.destroy_calls  # the deceiver was destroyed
    assert be.create_calls == [1, 2, 3]  # deficit(1,2) + one replacement from spare(3)


# --------------------------------------------------------------------------- #
# 4. Collect: dedupe, never lose
# --------------------------------------------------------------------------- #


def test_collect_dedupes_and_never_loses(tmp_path):
    manifest = make_manifest(6)
    fl = make_fleet(tmp_path, manifest)
    fl.provision()
    fl.assign_shards()  # auto_complete produces every run
    added = fl.collect()
    assert added == 6
    have = F.collected_run_ids(fl.runs_dir)
    assert have == {ln["run_id"] for ln in manifest}
    # Second collect adds nothing (idempotent) ...
    assert fl.collect() == 0
    # ... and losing every instance does not lose a single collected result.
    for iid in list(fl.backend.instances):
        fl.backend.destroy_instance(iid)
    assert F.collected_run_ids(fl.runs_dir) == have


def test_collect_never_overwrites_existing_result(tmp_path):
    manifest = make_manifest(4)
    fl = make_fleet(tmp_path, manifest)
    sentinel = fl.runs_dir / "r000.json"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text('{"sentinel": true}', "utf-8")
    fl.provision()
    fl.assign_shards()
    fl.collect()
    assert json.loads(sentinel.read_text("utf-8")) == {"sentinel": True}  # untouched


# --------------------------------------------------------------------------- #
# 5. Gap repair: exactly the missing ids, converges
# --------------------------------------------------------------------------- #


def test_plan_gap_repair_covers_missing_exactly():
    missing = ["r001", "r005", "r009", "r010"]
    plan = F.plan_gap_repair(missing, [10, 20, 30])
    assigned = [rid for ids in plan.values() for rid in ids]
    assert sorted(assigned) == sorted(missing)
    assert set(plan) == {10, 20, 30}


def test_gap_repair_redispatches_exactly_missing_and_converges(tmp_path):
    manifest = make_manifest(12)
    fl = make_fleet(tmp_path, manifest, config=F.FleetConfig(target_size=4, over_provision=0))
    fl.backend.blocked = {"r003", "r007", "r011"}  # these fail on the first pass
    fl.provision()
    fl.assign_shards()
    fl.collect()
    missing = F.find_missing_run_ids(manifest, fl.runs_dir)
    assert sorted(missing) == ["r003", "r007", "r011"]

    fl.backend.execute_calls.clear()
    redispatched = fl.gap_repair()
    assert sorted(redispatched) == ["r003", "r007", "r011"]
    dispatched_ids = re.findall(
        r'"run_id":\s*"([^"]+)"', " ".join(c for _, c in fl.backend.execute_calls)
    )
    assert sorted(set(dispatched_ids)) == ["r003", "r007", "r011"]  # only the missing

    fl.collect()
    assert F.find_missing_run_ids(manifest, fl.runs_dir) == []  # converged
    assert fl.gap_repair() == []  # nothing left to do


def test_gap_repair_prefers_fastest_alive_workers(tmp_path):
    manifest = make_manifest(6)
    fl = make_fleet(tmp_path, manifest, config=F.FleetConfig(target_size=3, over_provision=0))
    fl.provision()
    ids = sorted(fl.backend.instances)
    # Nothing produced yet -> everything is missing.
    speeds = {ids[0]: 900.0, ids[1]: 100.0, ids[2]: 500.0}  # ids[1] fastest
    fl.backend.execute_calls.clear()
    fl.gap_repair(worker_speeds=speeds)
    first_target = fl.backend.execute_calls[0][0]
    assert first_target == ids[1]  # fastest worker gets the first (largest) slice


# --------------------------------------------------------------------------- #
# 6. run_fleet: full state machine, resumable, converges, tears down
# --------------------------------------------------------------------------- #


def test_run_fleet_converges_and_tears_down(tmp_path):
    manifest = make_manifest(12)
    fl = make_fleet(tmp_path, manifest, config=F.FleetConfig(target_size=4, over_provision=0))
    fl.backend.blocked = {"r002", "r008"}  # first pass leaves two gaps
    result = fl.run_fleet()
    assert result.completed is True
    assert result.runs_done == 12
    assert result.missing == []
    assert F.collected_run_ids(fl.runs_dir) == {ln["run_id"] for ln in manifest}
    assert len(fl.backend.create_calls) == 4       # no over-creation
    # Teardown destroyed exactly the fleet it built (the shard-index keys are ids).
    assert sorted(fl.backend.destroy_calls) == sorted(int(k) for k in fl.state["shard_index"])
    assert fl.live_instances() == []


def test_run_fleet_resumes_without_restarting_completed(tmp_path):
    manifest = make_manifest(8)
    # Pre-seed half the results as already collected (a prior, crashed run).
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    preseeded = {"r000", "r001", "r002", "r003"}
    for rid in preseeded:
        (runs_dir / f"{rid}.json").write_text('{"preseeded": true}', "utf-8")
    fl = make_fleet(tmp_path, manifest, config=F.FleetConfig(target_size=4, over_provision=0))
    result = fl.run_fleet()
    assert result.completed is True
    # The pre-seeded results were never regenerated / overwritten.
    for rid in preseeded:
        assert json.loads((runs_dir / f"{rid}.json").read_text("utf-8")) == {"preseeded": True}


# --------------------------------------------------------------------------- #
# 7. Teardown / kill-switch
# --------------------------------------------------------------------------- #


def test_teardown_destroys_all_and_reports_spend(tmp_path):
    manifest = make_manifest(4)
    fl = make_fleet(tmp_path, manifest)
    fl.provision()
    ids = sorted(fl.backend.instances)
    fl.teardown()
    assert sorted(fl.backend.destroy_calls) == ids
    assert fl.live_instances() == []


# --------------------------------------------------------------------------- #
# 8. run_one command == Runner's interface
# --------------------------------------------------------------------------- #


def test_run_one_command_matches_runner_interface():
    line = {"run_id": "r001", "arm": "B1", "seed": 0}
    cmd = F.run_one_command(line, "/workspace/data", "/workspace/results")
    assert cmd[:3] == ["python", "-m", "experiments.run_one"]
    assert "--manifest-line" in cmd and "--data-dir" in cmd and "--out-dir" in cmd
    assert cmd[cmd.index("--device") + 1] == "cuda"  # GPU workers, not run_one's cpu default
    payload = cmd[cmd.index("--manifest-line") + 1]
    assert json.loads(payload)["run_id"] == "r001"


# --------------------------------------------------------------------------- #
# 9. Monitor: slow-host rule, spend cap, full schema
# --------------------------------------------------------------------------- #


def _report(iid, *, mean=None, gpu="RTX 3060", vram_total=12288, runs=1,
            hb=NOW, finished=NOW, current=None):
    return M.WorkerReport(
        instance_id=iid, gpu_name=gpu, vram_total_mb=vram_total, vram_used_mb=4000,
        vram_free_mb=vram_total - 4000, gpu_util_pct=95, dataset_sha_ok=True,
        current_run_id=current, runs_completed=runs, shard_index=0, n_shards=4,
        heartbeat_ts=hb, mean_run_walltime_s=mean, last_run_walltime_s=mean,
        last_run_finished_ts=finished, phase="running",
    )


def test_monitor_flags_3x_slow_outlier(tmp_path):
    manifest = make_manifest(8)
    mon = M.FleetMonitor(manifest, tmp_path / "runs", backend=None, clock=lambda: NOW)
    instances = [F.parse_instance(instance_row(i)) for i in (1, 2, 3, 4)]
    reports = {
        1: _report(1, mean=100.0),
        2: _report(2, mean=105.0),
        3: _report(3, mean=98.0),
        4: _report(4, mean=300.0),  # the planted ~3x outlier
    }
    status = mon.build_status(instances, reports, set(), now=NOW, credit=90.0,
                              throughput_per_hr=10.0)
    assert status["fleet"]["slow_hosts"] == [4]
    assert 90 < status["fleet"]["median_run_walltime_s"] < 110
    view4 = next(v for v in status["instances"] if v["instance_id"] == 4)
    assert "slow" in view4["flags"]
    assert any("SLOW-HOST" in a for a in status["alerts"])


def test_monitor_slow_rule_needs_min_samples(tmp_path):
    manifest = make_manifest(4)
    mon = M.FleetMonitor(manifest, tmp_path / "runs", clock=lambda: NOW)
    instances = [F.parse_instance(instance_row(i)) for i in (1, 2)]
    reports = {1: _report(1, mean=100.0), 2: _report(2, mean=400.0)}
    status = mon.build_status(instances, reports, set(), now=NOW, credit=None,
                              throughput_per_hr=0.0)
    assert status["fleet"]["slow_hosts"] == []  # too few samples to trust a median


def test_monitor_detects_gpu_mismatch_from_report(tmp_path):
    manifest = make_manifest(4)
    mon = M.FleetMonitor(manifest, tmp_path / "runs", clock=lambda: NOW)
    instances = [F.parse_instance(instance_row(1))]
    reports = {1: _report(1, gpu="GTX 1660", vram_total=6144, mean=100.0)}
    status = mon.build_status(instances, reports, set(), now=NOW, credit=None,
                              throughput_per_hr=0.0)
    view = status["instances"][0]
    assert view["gpu_ok"] is False and "gpu_mismatch" in view["flags"]
    assert status["fleet"]["gpu_mismatch_hosts"] == [1]


def test_monitor_stall_and_dead_detection(tmp_path):
    manifest = make_manifest(4)
    mon = M.FleetMonitor(manifest, tmp_path / "runs", clock=lambda: NOW,
                         config=M.MonitorConfig(stall_minutes=10, dead_grace_s=120))
    instances = [F.parse_instance(instance_row(i)) for i in (1, 2)]
    reports = {
        1: _report(1, mean=100.0, finished=NOW - 20 * 60, hb=NOW),          # stalled
        2: _report(2, mean=100.0, finished=NOW - 60, hb=NOW - 10 * 60),     # dead (no heartbeat)
    }
    status = mon.build_status(instances, reports, set(), now=NOW, credit=None,
                              throughput_per_hr=1.0)
    assert status["fleet"]["stalled_hosts"] == [1]
    assert 2 in status["fleet"]["dead_hosts"]


def test_monitor_spend_cap_freezes_and_shouts(tmp_path):
    manifest = make_manifest(4)

    class _FleetRef:
        provisioning_frozen = False

    ref = _FleetRef()
    mon = M.FleetMonitor(manifest, tmp_path / "runs", fleet=ref, clock=lambda: NOW,
                         config=M.MonitorConfig(spend_cap_usd=30.0))
    # Two instances that already accrued ~$18 each (360 h @ $0.05) -> ~$36 > $30.
    instances = [
        F.parse_instance(instance_row(1, dph_total=0.05, start_date=NOW - 360 * 3600)),
        F.parse_instance(instance_row(2, dph_total=0.05, start_date=NOW - 360 * 3600)),
    ]
    status = mon.build_status(instances, {}, set(), now=NOW, credit=50.0,
                              throughput_per_hr=1.0)
    assert status["money"]["cap_breached"] is True
    assert status["money"]["provisioning_frozen"] is True
    assert ref.provisioning_frozen is True  # the monitor froze the controller
    assert mon.compact_line(status).startswith("!! SPEND-CAP !!")
    assert any("SPEND-CAP" in a for a in status["alerts"])


def test_fleet_status_full_schema_and_roundtrip(tmp_path):
    manifest = make_manifest(10)
    mon = M.FleetMonitor(manifest, tmp_path / "runs", clock=lambda: NOW)
    instances = [F.parse_instance(instance_row(i)) for i in (1, 2, 3)]
    reports = {i: _report(i, mean=120.0) for i in (1, 2, 3)}
    collected = {"r000", "r001", "r002"}  # 3/10, split across blocks
    status = mon.build_status(instances, reports, collected, now=NOW, credit=93.5,
                              throughput_per_hr=30.0)

    assert set(status) == {
        "schema_version", "ts", "iso", "target_fleet_size", "instances",
        "fleet", "money", "blocks", "alerts",
    }
    assert set(status["fleet"]) >= {
        "alive", "loading", "offline", "target", "runs_done", "runs_total",
        "progress_pct", "median_run_walltime_s", "throughput_runs_per_hr",
        "eta_hours", "slow_hosts", "stalled_hosts", "dead_hosts", "gpu_mismatch_hosts",
    }
    assert set(status["money"]) == {
        "accrued_usd", "rate_usd_per_hr", "projected_total_usd", "cap_usd",
        "cap_breached", "provisioning_frozen", "account_credit_usd",
    }
    assert status["fleet"]["runs_done"] == 3
    assert status["fleet"]["runs_total"] == 10
    assert status["blocks"]["T1a"]["total"] == 5 and status["blocks"]["T1a"]["done"] == 3
    for view in status["instances"]:
        assert set(view) >= {
            "instance_id", "status", "gpu_name", "gpu_ok", "vram_used_mb", "vram_free_mb",
            "gpu_util_pct", "current_run_id", "runs_completed", "last_heartbeat_age_s",
            "dph", "accrued_usd", "flags",
        }

    path = mon.write_status(status)
    assert path.name == "fleet_status.json"
    assert json.loads(path.read_text("utf-8"))["schema_version"] == M.STATUS_SCHEMA_VERSION
    # The status file must not pollute the collected-results scan.
    assert F.collected_run_ids(mon.runs_dir) == set()


def test_monitor_sample_and_cycle_against_mock(tmp_path):
    manifest = make_manifest(4)
    be = MockVastBackend(offers=[offer_row(i) for i in range(1, 5)],
                         manifest_run_ids=[ln["run_id"] for ln in manifest])
    be.instances[42] = instance_row(42, label="fedgrid-fleet")
    mon = M.FleetMonitor(manifest, tmp_path / "runs", backend=be, clock=lambda: NOW)
    line = mon.run_cycle()
    assert "alive" in line and "runs 0/4" in line
    assert (tmp_path / "runs" / "fleet_status.json").exists()
    status = mon.sample()
    assert status["money"]["account_credit_usd"] == 97.98  # credit read each cycle


# --------------------------------------------------------------------------- #
# 10. Money helpers + safety-of-import
# --------------------------------------------------------------------------- #


def test_money_helpers():
    now = NOW
    insts = [F.parse_instance(instance_row(1, dph_total=0.05, start_date=now - 3600)),
             F.parse_instance(instance_row(2, dph_total=0.05, start_date=now - 1800))]
    assert F.total_accrued_usd(insts, now) == pytest.approx(0.05 + 0.025)
    assert F.burn_rate_usd_per_hr(insts) == pytest.approx(0.10)
    assert F.project_total_spend(5.0, 2.0, 3.0) == pytest.approx(11.0)
    assert F.eta_hours(100, 540, 100.0) == pytest.approx(4.4)
    assert F.eta_hours(540, 540, 0.0) == 0.0
    assert F.eta_hours(0, 10, 0.0) is None


def test_cli_backend_requires_key_only_when_used(monkeypatch):
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    be = F.CliVastBackend(api_key=None)  # construction never touches the network
    with pytest.raises(RuntimeError, match="VAST_API_KEY"):
        be.search_offers(F.build_search_query(F.FleetConfig()))


# --------------------------------------------------------------------------- #
# 11. SSH transport (the real backend's data plane) -- pure parts only
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("1234:/workspace/bundle", (1234, "/workspace/bundle")),
        ("46114485:/workspace/status/self.json", (46114485, "/workspace/status/self.json")),
        (r"E:\paper7\fedgrid\bundle", (None, r"E:\paper7\fedgrid\bundle")),  # Windows drive
        ("/home/u/runs", (None, "/home/u/runs")),
        ("runs/incoming", (None, "runs/incoming")),
    ],
)
def test_split_remote_never_mistakes_a_drive_letter_for_an_instance(spec, expected):
    assert F.split_remote(spec) == expected


def test_ssh_endpoint_prefers_direct_then_proxy():
    direct = {
        "id": 1, "public_ipaddr": "203.0.113.9 ", "ssh_host": "ssh3.vast.ai", "ssh_port": 34484,
        "ports": {"22/tcp": [{"HostPort": "40011"}]}, "image_runtype": "ssh_direc ssh_proxy",
    }
    assert F.ssh_endpoint(direct) == ("203.0.113.9", 40011)

    proxy = {"id": 2, "ssh_host": "ssh3.vast.ai", "ssh_port": 34484,
             "image_runtype": "ssh_direc ssh_proxy"}
    assert F.ssh_endpoint(proxy) == ("ssh3.vast.ai", 34484)

    # jupyter-runtype instances expose sshd one port above ssh_port (vastai ssh-url).
    jup = {"id": 3, "ssh_host": "ssh5.vast.ai", "ssh_port": 12000, "image_runtype": "jupyter"}
    assert F.ssh_endpoint(jup) == ("ssh5.vast.ai", 12001)


def test_ssh_endpoints_keeps_the_proxy_as_a_fallback():
    # public_ipaddr is not always the host's address (Vast echoes the caller's IP
    # while an instance boots), so the proxy must stay in the candidate list.
    row = {
        "id": 1, "public_ipaddr": "203.0.113.9", "ports": {"22/tcp": [{"HostPort": "40011"}]},
        "ssh_host": "ssh9.vast.ai", "ssh_port": 36404, "image_runtype": "ssh_direc ssh_proxy",
    }
    assert F.ssh_endpoints(row) == [("203.0.113.9", 40011), ("ssh9.vast.ai", 36404)]


def test_ssh_endpoint_raises_while_the_instance_is_still_loading():
    assert F.ssh_endpoints({"id": 4, "actual_status": "loading"}) == []
    with pytest.raises(RuntimeError, match="no ssh endpoint"):
        F.ssh_endpoint({"id": 4, "actual_status": "loading"})


def test_copy_rejects_nonsensical_directions(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "unused-never-sent")
    be = F.CliVastBackend()
    with pytest.raises(RuntimeError, match="neither side names an instance"):
        be.copy("/tmp/a", "/tmp/b")
    with pytest.raises(RuntimeError, match="instance-to-instance"):
        be.copy("11:/workspace/x", "22:/workspace/x")


def test_state_survives_a_vanished_runs_dir(tmp_path):
    import shutil as _shutil

    fl = make_fleet(tmp_path, make_manifest(2))
    _shutil.rmtree(fl.runs_dir)  # the store disappears mid-battery
    fl.state["phase"] = "ASSIGN"
    fl._save_state()  # must not crash with instances already rented
    assert json.loads(fl.state_path.read_text("utf-8"))["phase"] == "ASSIGN"


def test_provision_refuses_an_incomplete_bundle(tmp_path):
    manifest = make_manifest(2)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "run_shard.sh").write_text("#!/bin/sh\n", "utf-8")
    assert "manifest.jsonl" in F.validate_bundle(bundle)
    assert "experiments/run_one.py" in F.validate_bundle(bundle)
    assert "fedgrid-*.whl" in F.validate_bundle(bundle)

    be = MockVastBackend(offers=[offer_row(1)],
                         manifest_run_ids=[ln["run_id"] for ln in manifest])
    fl = F.Fleet(be, manifest, runs_dir=tmp_path / "runs", bundle_dir=bundle,
                 config=F.FleetConfig(target_size=1, over_provision=0), clock=lambda: NOW)
    with pytest.raises(RuntimeError, match="missing"):
        fl.provision(1)
    assert be.create_calls == []  # not a single dollar spent on a broken bundle


def test_excluded_machines_are_never_re_rented():
    offers = [F.parse_offer(offer_row(1)), F.parse_offer(offer_row(2))]
    cfg = F.FleetConfig(exclude_machine_ids=(1,))  # machine_id == oid in offer_row
    kept = F.filter_offers(offers, cfg)
    assert [o.machine_id for o in kept] == [2]


def test_default_image_is_a_vast_ready_image():
    # SSH launch mode replaces the image ENTRYPOINT: an image without an sshd
    # (pytorch/pytorch:*) is unreachable, which cost a whole paid rehearsal.
    assert F.DEFAULT_IMAGE.startswith("vastai/")
    assert "cu12" in F.DEFAULT_IMAGE  # cu12.x keeps the driver>=525 host mix usable


def test_poll_until_ready_treats_unreachable_sshd_as_pending(tmp_path):
    manifest = make_manifest(2)

    class SshLateBackend(MockVastBackend):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.probes = 0

        def ssh_ready(self, instance_id: int) -> bool:
            self.probes += 1
            return self.probes > 3  # sshd only answers on the 4th probe

    be = SshLateBackend(offers=[offer_row(i) for i in range(1, 4)],
                        manifest_run_ids=[ln["run_id"] for ln in manifest])
    cfg = F.FleetConfig(target_size=1, over_provision=0, poll_interval_s=0.0)
    fl = make_fleet(tmp_path, manifest, backend=be, config=cfg)
    fl.provision(1)
    assert be.probes >= 4  # provision kept polling instead of failing fast
    assert be.destroy_calls == []  # a slow-to-boot sshd is NOT a deceiving host


def test_worker_scripts_honour_the_remote_contract():
    fleet_dir = REPO_ROOT / "experiments" / "fleet"
    onstart = (fleet_dir / "onstart.sh").read_text("utf-8")
    run_shard = (fleet_dir / "run_shard.sh").read_text("utf-8")
    # onstart: GPU verify + dataset sha256 + idle-wait, no baked secret.
    assert "nvidia-smi" in onstart
    assert "sha256sum" in onstart and "FEDGRID_DATA_SHA256" in onstart
    assert F.REMOTE_STATUS_FILE.split("/")[-1] in onstart  # writes self.json
    assert "VAST_API_KEY" not in onstart and "VAST_API_KEY" not in run_shard
    # run_shard: the exact idempotent worker entrypoint (on GPU) + heartbeat.
    assert "experiments.run_one" in run_shard
    assert "--data-dir" in run_shard and "--out-dir" in run_shard
    assert "--device cuda" in run_shard
    assert "heartbeat" in run_shard.lower()
