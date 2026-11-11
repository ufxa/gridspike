"""Emit the gap-fill manifest for the tripled (x3) battery -> manifest_gap_x3.jsonl.

Why this exists: ``manifest_x3.jsonl`` (1500 runs, CNN-LSTM + SNN) and
``manifest_ssm_x3.jsonl`` (1065 runs, SSM) tripled the Compact battery's seed
coverage -- Tier-1 blocks to seeds 0-14, Tier-2/3 blocks to seeds 0-8 -- but the
expansion skipped every template that lived outside ``manifest.jsonl`` /
``manifest_ssm.jsonl``'s enumerated blocks:

* ``temporal-*``  -- the chronological-split headline battery (t1a_temporal /
  t1b_temporal, ``federation.split_mode="temporal"``), the paper's leak-free
  headline claim.
* ``t4mag-*``     -- the attacker-magnitude sweep (AdaptiveCosine scale 1/2/5/10).
* ``t3tau-*``     -- the fidelity-threshold sweep for CNN-LSTM and SNN (the x3
  run only extended the SSM), PLUS the SSM t2c-derived variants the SSM x3 pass
  itself skipped (sources with ldp tau != 0.3).
* eight ``t2e-*`` ablation variants -- SNN ``codingrate-ts4/ts16`` (rate-coding x
  t_steps) and SSM ``ts4/ts16``.

Selection rule (verified against ``docs/X3_ANALYSIS.md`` sec 0, which reports the
same 278 templates): a template (= run_id minus the ``-sN`` suffix) goes in iff
it has >=1 result file in ``runs/`` AND no line in ``manifest_x3.jsonl`` or
``manifest_ssm_x3.jsonl``. All 278 qualify; every one already holds its original
seed budget in ``runs/`` (temporal: s0-4, the rest: s0-2).

Seeds emitted -- the *missing* tail of the x3 range, never re-running an existing
result (x3 itself treats the old rows as carried over, cf. X3_ANALYSIS sec 4.4):
``tier==1`` templates get seeds 5-14, ``tier>1`` get 3-8.

Line construction replicates the x3 transform bit-for-bit: clone the template's
seed-0 line from its canonical source manifest, then change exactly the four
fields x3 changes per seed -- ``run_id`` suffix, top-level ``seed``,
``config.seed`` and ``config.federation.seed``. Everything else is verbatim:
``config_hash`` stays at the s0 value (stale across seeds, exactly like x3 --
see X3_ANALYSIS sec 4.5) and ``config.ldp.seed`` is NOT re-seeded either (x3's
own t2c lines keep ``ldp.seed=0`` at every seed).

Canonical sources (each verified: for every uncovered result file, the executed
config embedded at ``config.manifest.config`` equals the canonical line's
config, modulo the one documented exception):

    temporal-*  -> experiments/manifest_temporal.jsonl
    t4mag-*     -> experiments/manifest_magnitude.jsonl
    t3tau-*     -> experiments/manifest_tau.jsonl
    t2e-*-ssm-* -> experiments/manifest_ssm.jsonl
    t2e-fedavg-snn-codingrate-ts* -> experiments/manifest_rate.jsonl

Documented edge cases:

* ``t3tau-*-snn-*`` (60 templates): the s0-2 results were executed from
  ``manifest_snn_battery.jsonl`` lines, which carry an explicit
  ``model.threshold_init=0.25`` absent from ``manifest_tau.jsonl``'s snn rows.
  Functionally identical -- ``experiments/build.py`` defaults
  ``threshold_init`` to 0.25 -- so the canonical ``manifest_tau`` line is
  cloned (full RunSpec schema, honest s0 hash); only the stored hash/key
  presence differs from the executed line.
* ``manifest_rate.jsonl`` rows have the minimal ``{run_id, config,
  config_hash}`` schema; the missing top-level fields are synthesised from the
  run_id convention ``<block>-<arm>-<model>-<tags>-s<seed>`` (block=t2e,
  arm=fedavg, model=snn, tier=2) so every gap line is a full RunSpec dict.
* Serialisation follows the x3 manifests: one JSON object per line,
  ``sort_keys=True``, compact separators.

    python -m experiments.gen_gap_x3 --out experiments/manifest_gap_x3.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path

__all__ = ["CANONICAL_SOURCES", "TIER1_SEEDS_X3", "TIER2_SEEDS_X3", "gap_rows"]

#: x3 coverage: Tier-1 blocks ran seeds 0-14, Tier-2/3 blocks ran seeds 0-8.
TIER1_SEEDS_X3 = 15
TIER2_SEEDS_X3 = 9

_SEED_RE = re.compile(r"-s(\d+)$")

_EXP = Path("experiments")

#: Canonical source manifest per run_id prefix, in the order rows are emitted.
CANONICAL_SOURCES: tuple[tuple[str, str], ...] = (
    ("temporal-", "manifest_temporal.jsonl"),
    ("t4mag-", "manifest_magnitude.jsonl"),
    ("t3tau-", "manifest_tau.jsonl"),
    ("t2e-", "manifest_ssm.jsonl"),   # ssm variants only; snn handled below
)
_RATE_SRC = "manifest_rate.jsonl"    # t2e-fedavg-snn-codingrate-ts* templates

#: Directories holding already-executed results. An emitted run_id must never
#: collide with a file stem here.
RESULT_DIRS = ("runs", "runs_x3", "runs_ssm_x3")

#: The x3 manifests whose templates are already covered (never matched by hash --
#: their per-seed config_hash is stale, see X3_ANALYSIS sec 4.5).
X3_MANIFESTS = ("manifest_x3.jsonl", "manifest_ssm_x3.jsonl")


def _template(run_id: str) -> str:
    return _SEED_RE.sub("", run_id)


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def _canonical_name(run_id: str) -> str:
    for prefix, name in CANONICAL_SOURCES:
        if run_id.startswith(prefix):
            if prefix == "t2e-" and "-ssm-" not in run_id:
                return _RATE_SRC   # snn codingrate-ts* ablations live there
            return name
    raise ValueError(f"no canonical source for run_id {run_id!r}")


def _x3_transform(row: dict, seed: int) -> dict:
    """The exact x3 per-seed edit: four fields move, everything else verbatim.

    ``config_hash`` deliberately stays at the template's s0 hash (x3 convention,
    stale by design) and ``config.ldp.seed`` is not touched (x3 t2c keeps
    ``ldp.seed=0`` on every expanded seed).
    """
    new = copy.deepcopy(row)
    new["run_id"] = f"{_template(row['run_id'])}-s{seed}"
    new["seed"] = seed
    new["config"]["seed"] = seed
    new["config"]["federation"]["seed"] = seed
    return new


def _synthesise_schema(row: dict) -> dict:
    """Fill the RunSpec top-level fields a minimal-schema source line lacks.

    ``manifest_rate.jsonl`` rows carry only ``run_id``/``config``/``config_hash``;
    the gap manifest speaks the full schema, so arm/block/model/tier are parsed
    from the run_id layout ``<block>-<arm>-<model>-<tags>-s<seed>`` and ``seed``
    is set by the caller via :func:`_x3_transform`.
    """
    parts = _template(row["run_id"]).split("-")
    out = {
        "arm": parts[1],
        "block": parts[0],
        "config": row["config"],
        "config_hash": row["config_hash"],
        "model": parts[2],
        "run_id": row["run_id"],
        "seed": row["config"]["seed"],
        "tier": 2,   # every t2e ablation is Tier-2 (manifest._BLOCK_BUILDERS)
    }
    return out


def gap_rows(root: Path = Path(".")) -> list[dict]:
    """All gap-fill rows: missing x3 seeds for templates the expansion skipped."""
    exp = root / "experiments"

    x3_ids: set[str] = set()
    for name in X3_MANIFESTS:
        x3_ids.update(r["run_id"] for r in _load(exp / name))
    x3_templates = {_template(r) for r in x3_ids}

    # run_ids already executed, per results dir -> seeds present per template
    have: dict[str, set[int]] = {}
    for d in RESULT_DIRS:
        for f in (root / d).glob("*.json"):
            m = _SEED_RE.search(f.stem)
            if m:
                have.setdefault(_template(f.stem), set()).add(int(m.group(1)))

    # index canonical source lines by run_id (first occurrence wins)
    src_lines: dict[str, dict] = {}
    for _prefix, name in CANONICAL_SOURCES:
        for r in _load(exp / name):
            src_lines.setdefault(r["run_id"], r)
    rate_lines = {r["run_id"]: r for r in _load(exp / _RATE_SRC)}

    out: list[dict] = []
    skipped_no_source = []
    for t in sorted(have):
        if t in x3_templates:
            continue                                   # already in the x3 battery
        seeds_have = have[t]
        canon = _canonical_name(t + "-s0")             # prefix only; -s0 is a probe
        base = rate_lines.get(t + "-s0") if canon == _RATE_SRC else src_lines.get(t + "-s0")
        if base is None:
            skipped_no_source.append(t)
            continue
        row0 = base if base.get("block") is not None else _synthesise_schema(base)
        tier = int(row0.get("tier", 2))
        target = set(range(TIER1_SEEDS_X3 if tier == 1 else TIER2_SEEDS_X3))
        missing = sorted(target - seeds_have)
        for s in missing:
            out.append(_x3_transform(row0, s))
    if skipped_no_source:
        raise SystemExit(f"uncovered templates with no canonical source: {skipped_no_source}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m experiments.gen_gap_x3",
                               description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path("experiments") / "manifest_gap_x3.jsonl")
    ap.add_argument("--root", type=Path, default=Path("."))
    args = ap.parse_args(argv)

    rows = gap_rows(args.root)
    ids = [r["run_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("run_id collision inside the gap manifest")

    # hard validation: nothing emitted may already exist as a result or x3 line
    x3_ids: set[str] = set()
    for name in X3_MANIFESTS:
        x3_ids.update(r["run_id"] for r in _load(args.root / "experiments" / name))
    dup_x3 = [r for r in ids if r in x3_ids]
    if dup_x3:
        raise SystemExit(f"gap run_ids already in x3 manifests: {dup_x3[:5]}")
    dup_res = []
    for d in RESULT_DIRS:
        have = {f.stem for f in (args.root / d).glob("*.json")}
        dup_res += [r for r in ids if r in have]
    if dup_res:
        raise SystemExit(f"gap run_ids already executed: {dup_res[:5]} ({len(dup_res)} total)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows),
        encoding="utf-8",
    )

    counts = Counter(r["block"] for r in rows)
    print(f"wrote {len(rows)} gap-fill runs -> {args.out}")
    for b in sorted(counts):
        print(f"  {b}: {counts[b]} runs")
    by_model = Counter(r["model"] for r in rows)
    print("  by model:", dict(sorted(by_model.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
