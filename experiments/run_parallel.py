"""Run several shards concurrently on one big GPU box.

A single sequential shard leaves a 24 GB RTX 3090 at ~40% utilisation and ~1.8 GB VRAM for
these small models, so the box is mostly idle. This launches N worker processes over disjoint
strides of the same manifest, each an ordinary idempotent `run_one` loop, and reports a
combined progress line. VRAM is the binding limit: at ~1.9 GB per run, N=5 fits comfortably
alongside a desktop session on a 24 GB card.

    python -m experiments.run_parallel --manifest experiments/manifest_tau.jsonl \
        --workers 5 --data-dir /root/fedgrid/data_battery --out-dir /root/fedgrid/runs
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

LOCK = threading.Lock()
DONE: dict[int, int] = {}


def shard_loop(idx: int, rows: list[dict], args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    for row in rows:
        if (out / f"{row['run_id']}.json").exists():
            continue
        t = time.time()
        # Hard per-run timeout: a single federated run is ~1-3 min, so anything past
        # RUN_TIMEOUT_S is a hung run (the 10-hour zombies we hit). subprocess.run
        # kills the child on timeout, so no run can wedge the GPU overnight.
        try:
            rc = subprocess.run(
                [sys.executable, "-m", "experiments.run_one", "--manifest-line", json.dumps(row),
                 "--data-dir", args.data_dir, "--out-dir", str(out), "--device", args.device],
                capture_output=True, text=True, timeout=args.run_timeout,
            ).returncode
        except subprocess.TimeoutExpired:
            rc = -9  # killed as hung; the run stays "missing" and is retried next pass
        with LOCK:
            DONE[idx] = DONE.get(idx, 0) + 1
            total = sum(DONE.values())
            print(f"[w{idx}] {row['run_id']} rc={rc} {time.time() - t:.0f}s "
                  f"| combined {total}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--data-dir", default="data_battery")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--run-timeout", type=float, default=480.0,
                    help="kill any single run exceeding this many seconds (hung-run guard)")
    args = ap.parse_args(argv)

    rows = [json.loads(line) for line in
            args.manifest.read_text(encoding="utf-8").splitlines() if line]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if not (out / f"{r['run_id']}.json").exists()]
    print(f"{len(todo)} runs pending across {args.workers} concurrent workers", flush=True)

    t0 = time.time()
    threads = [threading.Thread(target=shard_loop, args=(i, todo[i::args.workers], args))
               for i in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"ALL DONE combined={sum(DONE.values())} wall={(time.time() - t0) / 60:.1f}min",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
