"""Run a stride-shard of the battery manifest on a local/own GPU.

The same contract the Vast workers honour, minus the rental machinery: idempotent
(``run_one`` SKIPs a run whose result exists), atomic per-run writes, and a stable
stride so two owned GPUs (server 3090 = shard 0, local 3060 = shard 1) never collide.
Tier-1 blocks run first so the reference frame lands even if the battery is cut short.

    python -m experiments.run_shard_local --shard 1 --n-shards 2 \
        --data-dir data_battery --out-dir runs --device cuda
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

#: Tier-1 first: the 12-arm reference frame and the attack x defense grid carry the
#: paper's core claims; the sensitivity sweeps can tolerate being finished last.
BLOCK_PRIORITY = {"t1a": 0, "t1b": 1, "t2a": 2, "t2c": 3, "t2b": 4, "t2e": 5, "t2d": 6}


def load_shard(manifest: Path, shard: int, n_shards: int) -> list[dict]:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    rows.sort(key=lambda r: (BLOCK_PRIORITY.get(r.get("block", ""), 9), r["run_id"]))
    return [r for i, r in enumerate(rows) if i % n_shards == shard]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("experiments/manifest.jsonl"))
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--n-shards", type=int, required=True)
    p.add_argument("--data-dir", default="data_battery")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    shard = load_shard(args.manifest, args.shard, args.n_shards)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"shard {args.shard}/{args.n_shards}: {len(shard)} runs", flush=True)

    t0 = time.time()
    done = failed = skipped = 0
    for i, row in enumerate(shard, 1):
        rid = row["run_id"]
        if (out / f"{rid}.json").exists():
            skipped += 1
            continue
        t = time.time()
        rc = subprocess.run(
            [sys.executable, "-m", "experiments.run_one", "--manifest-line", json.dumps(row),
             "--data-dir", args.data_dir, "--out-dir", str(out), "--device", args.device],
            capture_output=True, text=True,
        ).returncode
        if rc == 0:
            done += 1
        else:
            failed += 1
        rate = (time.time() - t0) / max(done, 1)
        left = (len(shard) - i) * rate / 60
        print(f"[{i}/{len(shard)}] {rid} rc={rc} {time.time() - t:.0f}s "
              f"| done={done} fail={failed} skip={skipped} | ETA {left:.0f}min", flush=True)
    print(f"SHARD COMPLETE done={done} failed={failed} skipped={skipped} "
          f"wall={(time.time() - t0) / 60:.1f}min", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
