"""Re-emit the headline battery (T1a + T1b) under the contiguous temporal split.

Why this exists: windows are cut with ``partition.STRIDE = 16`` at ``WINDOW_T = 64``,
so neighbouring windows share 75 % of their samples, and the battery's seeded
*permutation* split therefore puts a near-copy of almost every held-out window in
the training set. Measured on the shipped ``data_battery`` (client 00 of
``fed_4329260f68``): **99.7 %** of test windows share samples with a train window.
Every absolute macro-F1 in the paper is inflated by that leak, and no reviewer of a
time-series paper will accept a random split over overlapping windows.

The fix needs no new data: the windows are already stored in simulated-time order
per client, so the split is recomputed *at load* as contiguous blocks -- oldest 70 %
train, next 15 % val, newest 15 % test, with a four-window purge at each boundary
(64 samples = one full window) so no held-out window overlaps a train one. That is
``federation.split_mode = "temporal"`` (:func:`fedgrid.data.apply_temporal_split`),
which this generator stamps on a copy of every T1a/T1b line.

Block naming (deliberate, and the reason the report needs no change): the temporal
runs go to **``t1a_temporal`` / ``t1b_temporal``**, not to ``t1a``/``t1b`` with a
flag. ``experiments.report.Row.block`` is read verbatim from the manifest entry and
every table selects with ``Bank.select(block="t1a", ...)``, so re-using the old
block names would silently pool two different protocols into every existing number.
Distinct blocks keep the historical tables bit-identical and let the temporal table
select its own rows -- the same convention already used by ``t1a_ssm`` and ``t3tau``.

    python -m experiments.gen_temporal_battery --out experiments/manifest_temporal.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from experiments.manifest import config_hash

__all__ = ["RUN_ID_PREFIX", "SOURCE_BLOCKS", "temporal_rows"]

#: Source manifests holding the headline battery: the CNN-LSTM/SNN lines and the
#: Mamba (SSM) lines that were added in a second pass. Together T1a = 90 runs
#: (6 arms x 3 models x 5 seeds) and T1b = 300 (4 attacks x 5 arms x 3 models x 5 seeds).
DEFAULT_MANIFESTS: tuple[Path, ...] = (
    Path("experiments") / "manifest.jsonl",
    Path("experiments") / "manifest_ssm.jsonl",
)

#: Only the two Tier-1 blocks are replicated: the reviewer's objection is about the
#: headline numbers, and re-running the Tier-2 sweeps would triple the GPU bill.
SOURCE_BLOCKS: tuple[str, ...] = ("t1a", "t1b")

#: Prefix on the original run_id. Distinct from any block name, so a run file can
#: never be confused with its random-split twin (``temporal-t1a-fedavg-snn-s3``).
RUN_ID_PREFIX = "temporal-"


def temporal_rows(manifests: tuple[Path, ...] = DEFAULT_MANIFESTS) -> list[dict]:
    """One temporal-split copy of every T1a/T1b line, with fresh ids and hashes."""
    out: list[dict] = []
    for path in manifests:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("block") not in SOURCE_BLOCKS:
                continue
            new = copy.deepcopy(row)
            # The only config change: how the loaded windows are cut into splits.
            new["config"]["federation"]["split_mode"] = "temporal"
            new["block"] = f"{row['block']}_temporal"
            new["run_id"] = f"{RUN_ID_PREFIX}{row['run_id']}"
            new["config_hash"] = config_hash(new["config"])
            # arm / model / seed / tier are copied verbatim from the original line,
            # so a temporal row pairs with its random-split twin on every axis.
            out.append(new)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m experiments.gen_temporal_battery",
                                 description=__doc__)
    ap.add_argument("--manifests", type=Path, nargs="+", default=list(DEFAULT_MANIFESTS),
                    help="source manifests to replicate (default: main + ssm)")
    ap.add_argument("--out", type=Path, default=Path("experiments") / "manifest_temporal.jsonl")
    ap.add_argument("--expect", type=int, default=390,
                    help="expected run count (T1a 90 + T1b 300); 0 disables the check")
    args = ap.parse_args(argv)

    rows = temporal_rows(tuple(args.manifests))
    ids = [r["run_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("run_id collision in the temporal battery")
    if args.expect and len(rows) != args.expect:
        raise SystemExit(f"expected {args.expect} temporal runs, built {len(rows)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows),
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for r in rows:
        key = f"{r['block']}/{r['model']}"
        counts[key] = counts.get(key, 0) + 1
    print(f"wrote {len(rows)} temporal-split runs -> {args.out}")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
