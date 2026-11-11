"""Quarantine collected results produced by pre-fix code.

Fleet workers run a wheel frozen at push time, so a fix landed mid-battery does not reach
them. `RunResult.git_commit` is the ground truth: any run whose commit predates a fix that
changes its arm's semantics must not enter a paper table. This moves such results to
`runs_invalid/` so the arm simply reads as missing (and is re-run) instead of silently wrong.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

#: run_id prefix -> commit that fixed its semantics. A result carrying an older commit
#: for that prefix is stale. (2026-07-28: local-floor evaluation fix, commit 8f41e22.)
FIX_COMMITS: dict[str, str] = {"t1a-local-": "8f41e22"}
STALE_BEFORE: frozenset[str] = frozenset({"b2c5abc", "0005410", "4aae475", "e5c8826",
                                          "c17f747", "8112a0f", "cac643c"})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--quarantine", type=Path, default=Path("runs_invalid"))
    args = ap.parse_args()
    args.quarantine.mkdir(exist_ok=True)

    moved = 0
    for p in sorted(args.runs_dir.glob("*.json")):
        prefix = next((k for k in FIX_COMMITS if p.stem.startswith(k)), None)
        if prefix is None:
            continue
        try:
            commit = str(json.loads(p.read_text(encoding="utf-8")).get("git_commit", ""))
        except (OSError, ValueError):
            continue
        if commit[:7] in STALE_BEFORE:
            shutil.move(str(p), args.quarantine / p.name)
            moved += 1
            print(f"quarantined {p.name} (commit {commit[:7]} predates {FIX_COMMITS[prefix]})")
    print(f"quarantined {moved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
