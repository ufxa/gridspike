"""Execute exactly one manifest line into a ``runs/<run_id>.json`` RunResult.

The fleet depends on this exact contract (phase 4 workers shell out to it, one
manifest line per invocation):

* **Idempotent** -- if ``runs/<run_id>.json`` already exists and loads as a valid
  ``RunResult`` the run is skipped (prints ``SKIP <run_id>``, exit 0, no training).
  A worker can re-claim a shard without redoing finished work.
* **Built from the config alone** -- the federation is loaded from ``--data-dir``
  when its generator manifest matches (k, alpha, days, seed) and generated
  in-process otherwise (loud log line); the model, strategy, defense, attacks and
  LDP all come from :func:`experiments.build.build_run` (the manifest's own
  interpreter), so a line and its run agree by construction.
* **Atomic** -- the result is written to ``<run_id>.json.tmp`` and ``os.replace``-d
  onto the final path, so a killed worker never leaves a half-written JSON that a
  later idempotency check would mistake for a finished run.
* **Best-effort live streaming** -- with ``--stream-url`` each round event is
  POSTed to the console; a streaming failure is logged once and never touches the
  run's exit status.
* **exit 0 on success, non-zero on failure** -- the fleet detects failures by
  exit code and re-queues the shard (within the sec 3.9 re-run margin).

Usage::

    python -m experiments.run_one --manifest-line '<json>' \
        --data-dir data --out-dir runs [--stream-url URL] \
        [--device cuda|cpu] [--limit-rounds N]

Single-run executor module.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

__all__ = ["main", "run_line"]


def _load_valid(path: Path) -> bool:
    """True iff ``path`` already holds a loadable RunResult with final metrics."""
    from fedgrid.metrics.runresult import load_run

    try:
        result = load_run(path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, TypeError):
        return False
    return isinstance(result.final, dict) and bool(result.final)


class _Streamer:
    """Best-effort POST of each round event to ``stream_url``; never raises out.

    One failure disables further attempts and logs a single warning -- a dead
    console must not slow (let alone kill) the training loop.
    """

    def __init__(self, url: str | None, log: Any) -> None:
        self._url = url
        self._log = log
        self._alive = url is not None

    def __call__(self, event: dict[str, Any]) -> None:
        if not self._alive:
            return
        import urllib.error
        import urllib.request

        data = json.dumps(event).encode("utf-8")
        req = urllib.request.Request(
            self._url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            urllib.request.urlopen(req, timeout=2.0).close()  # noqa: S310 (operator-supplied URL)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            self._alive = False
            self._log(f"[run_one] streaming disabled after error: {exc!r}")


def run_line(
    line: str | dict[str, Any],
    data_dir: str | Path | None,
    out_dir: str | Path,
    stream_url: str | None = None,
    device: str = "cpu",
    limit_rounds: int | None = None,
    log: Any = print,
) -> tuple[int, Path]:
    """Run one manifest entry; returns (exit_code, output_path).

    Importable twin of the CLI so tests drive it without a subprocess.
    """
    from experiments.build import build_run
    from experiments.manifest import config_hash
    from fedgrid.metrics.runresult import save_run

    entry: dict[str, Any] = json.loads(line) if isinstance(line, str) else dict(line)
    run_id = str(entry["run_id"])
    config: dict[str, Any] = entry["config"]

    out_dir = Path(out_dir)
    final_path = out_dir / f"{run_id}.json"
    if final_path.exists() and _load_valid(final_path):
        log(f"SKIP {run_id}")
        return 0, final_path

    # Cheap integrity check: a tampered manifest line is a loud warning, not a
    # crash -- the run still proceeds from the config it actually carries.
    expected = entry.get("config_hash")
    if expected and expected != config_hash(config):
        log(f"[run_one] WARNING: config_hash mismatch for {run_id} (manifest edited?)")

    log(f"[run_one] START {run_id} (device={device}, limit_rounds={limit_rounds})")
    built = build_run(config, data_dir=data_dir, device=device, limit_rounds=limit_rounds, log=log)
    try:
        result = built.run(on_round=_Streamer(stream_url, log))
    except ValueError as exc:
        # A NaN/Inf fidelity score under a strong attack makes the audit ledger reject the
        # record by design (consensus/ledger._canonical uses allow_nan=False: "a NaN score
        # is a bug upstream that should surface loudly here"). For a sweep that deliberately
        # drives attacks to the breaking point, that divergence IS the result -- record a
        # completed run flagged `diverged` rather than crashing and leaving the config
        # missing forever. Only NaN/Inf-in-ledger is treated this way; any other ValueError
        # re-raises so a real bug is never masked.
        if "JSON compliant" not in str(exc) and "nan" not in str(exc).lower():
            raise
        from fedgrid.metrics.runresult import (RunResult, current_env_fingerprint,
                                               current_git_commit)
        log(f"[run_one] DIVERGED {run_id}: NaN fidelity score under strong attack -- recording as diverged")
        result = RunResult(
            method=str(config.get("model", {}).get("name", "?")),
            seed=int(config.get("seed", 0)),
            config={"manifest": entry, "diverged_reason": str(exc)[:200]},
            rounds=[],
            final={"diverged": True, "acc": 0.0, "macro_f1": 0.0,
                   "note": "diverged (NaN fidelity score) under strong attack; "
                           "audit ledger rejects NaN by design"},
            energy=None, latency=None,
            git_commit=current_git_commit(Path(__file__).resolve().parents[1]),
            env_fingerprint=current_env_fingerprint(),
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = out_dir / f"{run_id}.json.tmp"
        save_run(result, tmp_path)
        os.replace(tmp_path, final_path)
        log(f"[run_one] DONE {run_id} (diverged) -> {final_path}")
        return 0, final_path

    # Stamp full provenance onto the record: the whole manifest entry plus the
    # engine's own derived config, so every number traces back to its plan.
    result.config = {"manifest": entry, "engine": result.config}

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / f"{run_id}.json.tmp"
    save_run(result, tmp_path)
    os.replace(tmp_path, final_path)  # atomic: no half-written final file ever exists
    log(f"[run_one] DONE {run_id} -> {final_path} "
        f"(acc={result.final.get('acc')}, macro_f1={result.final.get('macro_f1')})")
    return 0, final_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m experiments.run_one",
        description="Execute one battery manifest line into a RunResult JSON.",
    )
    p.add_argument("--manifest-line", required=True, help="one JSON manifest entry")
    p.add_argument("--data-dir", default="data", help="pre-generated federation dir (default data)")
    p.add_argument("--out-dir", default="runs", help="output dir for <run_id>.json (default runs)")
    p.add_argument("--stream-url", default=None, help="POST each round event here (best-effort)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="torch device")
    p.add_argument("--limit-rounds", type=int, default=None, help="dev cap on FL rounds (smoke)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        code, _path = run_line(
            args.manifest_line,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            stream_url=args.stream_url,
            device=args.device,
            limit_rounds=args.limit_rounds,
        )
        return code
    except Exception:  # noqa: BLE001 -- the fleet reads the exit code; surface the trace
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
