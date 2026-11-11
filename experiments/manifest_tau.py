"""Generate the fidelity-threshold (tau) sweep as an extension manifest.

Why this exists: the battery's fixed `tau = 0.5` sits at the MEDIAN of the honest clients'
observed fidelity scores under realistic non-IID heterogeneity (measured 2026-07-28 over
1410 honest client-rounds: p25 0.436, median 0.502, p75 0.576), so the admission gate
rejects roughly half of the honest population. The consequence in the analysis was a
saturated gate -- honest false-reject rate 1.000 and attacker admission 0.000 at every
(sigma, tau) -- which turns the pillar-2 operating curve into a constant.

The attacker distribution is bimodal by design: the adaptive attacker deliberately scores
LOW (~0.15-0.22, it constrains its own score) while sign-flip scores HIGH (~0.78-1.17).
A threshold placed above the honest p75 therefore admits honest clients, rejects the naive
attacker, and lets the adaptive one through -- which is precisely the paper's thesis, and
cannot be observed at tau = 0.5.

This module re-emits the defended arms of the attack x defense grid at several thresholds,
writing a SEPARATE manifest so a battery already in flight is never disturbed.

Model coverage -- why the SSM has to be *projected*: this sweep derives its rows from the
battery manifest (``experiments/manifest.jsonl``), and that manifest only enumerates
``experiments.manifest.MODELS = ("cnn_lstm", "snn")``. The SSM lives in ``MODELS_EXT`` and is
emitted solely by the opt-in ``t1a_ssm`` activation block, which is the *clean* reference
frame -- it contains no defended t1b/t2c arm to re-emit at a new threshold. The ssm variants
are therefore projected from the cnn_lstm source rows: :func:`experiments.manifest.make_config`
makes no model-dependent choice other than the ``model`` block itself, and attaches the
snn-only ``coding``/``t_steps`` keys for ``model == "snn"`` only, so re-targeting is exactly
"set ``model.name``, drop the snn-only keys". The projected ids follow
:func:`experiments.manifest._run_id`'s ``<block>-<arm>-<model>-<tags>-s<seed>`` layout and
match, as a set, the already-executed ssm tau runs in ``experiments/manifest_ssm.jsonl``.

    python -m experiments.manifest_tau --out experiments/manifest_tau.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from experiments.manifest import MODELS, config_hash

#: Around and above the honest p75 (0.576); 0.5 is already covered by the main battery.
TAU_SWEEP: tuple[float, ...] = (0.6, 0.8, 1.0)

#: Models the tau sweep covers: the battery's two plus the projected SSM (see module docstring).
SWEEP_MODELS: tuple[str, ...] = ("cnn_lstm", "snn", "ssm")

#: Source rows every projected (non-battery) model is re-targeted from.
TEMPLATE_MODEL = "cnn_lstm"


def tau_tag(tau: float) -> str:
    return f"tau{str(tau).replace('.', 'p')}"


def retarget(row: dict, model: str) -> dict:
    """Re-target one battery row at ``model``, exactly as ``make_config`` would build it.

    Only used for models the battery does not enumerate (currently the SSM). ``snn`` is
    refused because its config carries the extra ``coding``/``t_steps`` keys that this
    projection has no source for -- the battery emits real snn rows, so use those.
    """
    if model == "snn":
        raise ValueError("snn is enumerated by the battery; use its own rows, not a projection")
    new = copy.deepcopy(row)
    model_block = new["config"]["model"]
    model_block["name"] = model
    # make_config attaches coding/t_steps for model == "snn" only.
    model_block.pop("coding", None)
    model_block.pop("t_steps", None)
    new["model"] = model
    parts = new["run_id"].split("-")  # manifest._run_id: <block>-<arm>-<model>-<tags>-s<seed>
    if parts[2] != row["model"]:
        raise AssertionError(f"unexpected run_id layout: {row['run_id']!r}")
    parts[2] = model
    new["run_id"] = "-".join(parts)
    new["config_hash"] = config_hash(new["config"])
    return new


def sweep_rows(
    manifest: Path,
    taus: tuple[float, ...] = TAU_SWEEP,
    models: tuple[str, ...] = SWEEP_MODELS,
) -> list[dict]:
    """One row per (defended battery run x model x tau), with a fresh run_id and config hash."""
    models = tuple(models)
    projected = tuple(m for m in models if m not in MODELS)  # not in the battery -> re-targeted
    out: list[dict] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        defense = row["config"].get("defense") or {}
        # Only arms that actually consult the fidelity gate can inform the curve.
        if defense.get("kind") != "pipeline" or row["block"] not in {"t1b", "t2c"}:
            continue
        sources = [row] if row["model"] in models else []
        if row["model"] == TEMPLATE_MODEL:
            sources.extend(retarget(row, m) for m in projected)
        for src in sources:
            for tau in taus:
                if abs(float(src["config"]["defense"].get("threshold", -1)) - tau) < 1e-9:
                    continue  # already in the main battery
                new = copy.deepcopy(src)
                new["config"]["defense"]["threshold"] = tau
                new["block"] = "t3tau"
                new["tier"] = 3
                new["run_id"] = f"t3tau-{tau_tag(tau)}-{src['run_id']}"
                new["config_hash"] = config_hash(new["config"])
                out.append(new)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("experiments/manifest.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("experiments/manifest_tau.jsonl"))
    ap.add_argument("--seeds", type=int, default=3, help="cap seeds per config (cost control)")
    args = ap.parse_args(argv)

    rows = sweep_rows(args.manifest)
    if args.seeds:
        rows = [r for r in rows if int(r.get("seed", 0)) < args.seeds]
    ids = [r["run_id"] for r in rows]
    assert len(ids) == len(set(ids)), "run_id collision in the tau sweep"

    args.out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
                        encoding="utf-8")
    by_tau: dict[float, int] = {}
    by_model: dict[str, int] = {}
    for r in rows:
        t = float(r["config"]["defense"]["threshold"])
        by_tau[t] = by_tau.get(t, 0) + 1
        by_model[r["model"]] = by_model.get(r["model"], 0) + 1
    print(f"wrote {len(rows)} tau-sweep runs -> {args.out}")
    for t in sorted(by_tau):
        print(f"  tau={t}: {by_tau[t]} runs")
    for m in sorted(by_model):
        print(f"  model={m}: {by_model[m]} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
