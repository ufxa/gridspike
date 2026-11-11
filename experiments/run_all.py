"""Enumerate (``--dry-run``) or drive the whole battery.

``--dry-run`` is the deliverable the artifact bar requires committed before
Tier-1 (VALIDATION_STANDARD sec 3.8): it writes ``experiments/manifest.jsonl`` (one
JSON line per planned run) and prints the per-block count table, the total, and
every documented cap -- **without training anything**. Without ``--dry-run`` it
executes the battery sequentially through :func:`experiments.run_one.run_line`
(each run idempotent, so re-invocation resumes), which is the single-host path;
the phase 4 fleet shards the same ``manifest.jsonl`` across workers instead.

Usage::

    python -m experiments.run_all --dry-run
    python -m experiments.run_all --data-dir data --out-dir runs   # sequential execute

Batch driver module.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.manifest import (
    BATTERY_NOTES,
    COMPACT_TIER1_SEEDS,
    COMPACT_TIER2_SEEDS,
    battery_runs,
    block_counts,
)

__all__ = ["main", "write_manifest"]

_DEFAULT_MANIFEST = Path("experiments") / "manifest.jsonl"


def write_manifest(
    path: Path, tier1_seeds: int, tier2_seeds: int, include_ssm: bool = False
) -> int:
    """Write one JSON line per planned run; returns the number of runs written."""
    runs = battery_runs(tier1_seeds, tier2_seeds, include_ssm=include_ssm)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for spec in runs:
            fh.write(spec.to_json_line())
            fh.write("\n")
    return len(runs)


def _print_table(tier1_seeds: int, tier2_seeds: int, include_ssm: bool = False) -> None:
    runs = battery_runs(tier1_seeds, tier2_seeds, include_ssm=include_ssm)
    counts = block_counts(runs)
    tier_of = {"t1a": 1, "t1b": 1, "t2a": 2, "t2b": 2, "t2c": 2, "t2d": 2, "t2e": 2,
               "t1a_ssm": 1}
    labels = {
        "t1a": "clean 12-arm reference frame",
        "t1b": "attack x defense grid (f=3/10)",
        "t2a": "alpha + IID sweep",
        "t2b": "attack-fraction sweep",
        "t2c": "LDP sigma x tau curve",
        "t2d": "validators M sweep",
        "t2e": "model/strategy ablations",
        "t1a_ssm": "SSM activation frame (opt-in)",
    }
    print(f"FedGrid Compact battery (Tier-1 = {tier1_seeds} seeds, Tier-2 = {tier2_seeds} seeds)")
    print(f"{'block':<8} {'tier':>4} {'runs':>5}  description")
    print(f"{'-'*8} {'-'*4:>4} {'-'*5:>5}  {'-'*30}")
    # Iterate the blocks actually present (block_counts preserves builder order,
    # then appends any opt-in block) so --include-ssm renders without a hardcode.
    for block in (b for b in counts if b != "TOTAL"):
        print(f"{block:<8} {tier_of.get(block, 0):>4} {counts[block]:>5}  "
              f"{labels.get(block, block)}")
    total = counts["TOTAL"]
    margin = round(total * 0.08)
    print(f"{'-'*8} {'-'*4:>4} {'-'*5:>5}")
    print(f"{'TOTAL':<8} {'':>4} {total:>5}  distinct runs")
    print(f"{'':<8} {'':>4} {'':>5}  + ~8% re-run margin ({margin}) -> ~{total + margin} all-in")
    print("\nDocumented bounds (never silent, per sec 3.9):")
    for note in BATTERY_NOTES:
        print(f"  - {note}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m experiments.run_all",
        description="Enumerate (--dry-run) or execute the FedGrid battery.",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="write manifest.jsonl + print the count table; no training")
    p.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST,
                   help=f"manifest output path (default {_DEFAULT_MANIFEST})")
    p.add_argument("--tier1-seeds", type=int, default=COMPACT_TIER1_SEEDS)
    p.add_argument("--tier2-seeds", type=int, default=COMPACT_TIER2_SEEDS)
    p.add_argument("--data-dir", default="data", help="pre-generated federation dir")
    p.add_argument("--out-dir", default="runs", help="RunResult output dir (execute mode)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--limit-rounds", type=int, default=None, help="dev cap on rounds")
    p.add_argument("--include-ssm", action="store_true",
                   help="append the opt-in SSM activation frame (t1a_ssm block); "
                        "the default battery is unchanged without this flag")
    args = p.parse_args(argv)

    if args.dry_run:
        n = write_manifest(args.manifest, args.tier1_seeds, args.tier2_seeds,
                           include_ssm=args.include_ssm)
        _print_table(args.tier1_seeds, args.tier2_seeds, include_ssm=args.include_ssm)
        print(f"\nwrote {n} manifest lines -> {args.manifest}")
        return 0

    # Sequential execute: idempotent per run, so this resumes a partial battery.
    from experiments.run_one import run_line

    runs = battery_runs(args.tier1_seeds, args.tier2_seeds, include_ssm=args.include_ssm)
    failures = 0
    for i, spec in enumerate(runs, 1):
        print(f"[run_all] {i}/{len(runs)} {spec.run_id}")
        try:
            code, _ = run_line(
                spec.to_json_line(), data_dir=args.data_dir, out_dir=args.out_dir,
                device=args.device, limit_rounds=args.limit_rounds,
            )
            failures += int(code != 0)
        except Exception as exc:  # noqa: BLE001 -- one bad run must not abort the battery
            failures += 1
            print(f"[run_all] FAILED {spec.run_id}: {exc!r}")
    print(f"[run_all] done: {len(runs) - failures}/{len(runs)} succeeded, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
