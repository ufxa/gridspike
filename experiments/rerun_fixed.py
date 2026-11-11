"""Re-run specific manifest lines on the local (post-fix) code, overwriting stale results.

Fleet workers execute a wheel frozen at push time, so a semantics fix landed mid-battery
never reaches them: their results for the affected arm keep the old behaviour and are
quarantined by ``experiments.quarantine_stale``. This re-runs those lines here, where the
code is current, and force-overwrites any lingering stale output.

    python -m experiments.rerun_fixed --prefix t1a-local- --data-dir data_battery
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("experiments/manifest.jsonl"))
    ap.add_argument("--prefix", required=True, help="run_id prefix to re-run (e.g. t1a-local-)")
    ap.add_argument("--data-dir", default="data_battery")
    ap.add_argument("--out-dir", type=Path, default=Path("runs"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    rows = [json.loads(line) for line in
            args.manifest.read_text(encoding="utf-8").splitlines() if line]
    todo = [r for r in rows if r["run_id"].startswith(args.prefix)]
    print(f"re-running {len(todo)} runs matching {args.prefix!r} with current code", flush=True)

    ok = fail = 0
    t0 = time.time()
    for i, row in enumerate(todo, 1):
        out = args.out_dir / f"{row['run_id']}.json"
        out.unlink(missing_ok=True)  # force: run_one SKIPs when the result exists
        t = time.time()
        rc = subprocess.run(
            [sys.executable, "-m", "experiments.run_one", "--manifest-line", json.dumps(row),
             "--data-dir", args.data_dir, "--out-dir", str(args.out_dir),
             "--device", args.device],
            capture_output=True, text=True,
        ).returncode
        ok, fail = (ok + 1, fail) if rc == 0 else (ok, fail + 1)
        print(f"[{i}/{len(todo)}] {row['run_id']} rc={rc} {time.time() - t:.0f}s", flush=True)
    print(f"done ok={ok} fail={fail} wall={(time.time() - t0) / 60:.1f}min", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
