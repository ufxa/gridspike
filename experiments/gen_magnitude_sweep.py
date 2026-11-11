"""Attacker-magnitude sweep: what a cosine gate cannot see (block T4mag).

Why this exists: the fidelity score is ``1 - cos(delta_i, median(deltas))``, which is
**scale-free**. An adaptive attacker that stays inside the cone can multiply its
update by any factor and the gate does not move by a single digit, while an
averaging aggregator (FedAvg, geometric median) is dragged proportionally further.
The whole battery ran ``AdaptiveCosine`` at its default ``scale = 1.0`` -- honest
magnitude -- so the worst case *inside the cone* has never been measured, and the
paper currently claims a robustness result over the one operating point that is
easiest to defend.

The sweep is the grid ``scale x defense arm x model x seed``:

* ``scale in {1, 2, 5, 10}`` -- 1.0 is the historical operating point, so those runs
  reproduce the T1b adaptive-cosine cells exactly (same arm, model, seed and every
  other config field) and are the sweep's own sanity anchor.
* ``no_defense`` (the undefended reference), ``geomed_alone`` (magnitude-sensitive
  aggregation with no gate), ``pipeline_krum`` and ``pipeline_geomed`` (the full
  fidelity -> PBFT -> aggregate pipelines at the reference ``tau = 0.5``).
* the three battery models x 3 seeds (Tier-2/3 seed budget).

4 x 4 x 3 x 3 = 144 runs. Attack: ``adaptive_cosine`` at ``f_att = 3/10`` with the
battery's ``target_score`` default -- only the magnitude lever moves.

    python -m experiments.gen_magnitude_sweep --out experiments/manifest_magnitude.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.manifest import (
    FULL_ATTACK_FRACTION,
    MODELS_EXT,
    RunSpec,
    _spec,
)

__all__ = ["ARMS", "BLOCK", "SCALES", "SEEDS", "TIER", "magnitude_runs"]

BLOCK = "t4mag"
TIER = 3
#: Magnitude multipliers on the in-cone adaptive update (1.0 = the historical runs).
SCALES: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)
#: Defense arms that can react to magnitude at all, plus the undefended reference.
#: ``krum_alone`` is excluded on purpose: Krum *selects* one update rather than
#: averaging, so its response to a scaled coalition is already covered by
#: ``pipeline_krum`` and the T1b Krum cells.
ARMS: tuple[str, ...] = ("no_defense", "geomed_alone", "pipeline_krum", "pipeline_geomed")
SEEDS = 3


def magnitude_runs(
    scales: tuple[float, ...] = SCALES,
    arms: tuple[str, ...] = ARMS,
    models: tuple[str, ...] = MODELS_EXT,
    seeds: int = SEEDS,
) -> list[RunSpec]:
    """The T4mag grid as :class:`~experiments.manifest.RunSpec` objects.

    Built through the battery's own ``_spec``/``make_config``, so the config schema,
    the run_id convention (``t4mag-<arm>-<model>-sc<scale>-s<seed>``) and the config
    hash are the same functions the main manifest uses -- there is no second place
    where a run is described.
    """
    out: list[RunSpec] = []
    for seed in range(seeds):
        for scale in scales:
            for arm in arms:
                for model in models:
                    out.append(_spec(
                        BLOCK, arm, model, seed, TIER,
                        tags=[("sc", float(scale))],
                        attack="adaptive_cosine",
                        attack_fraction=FULL_ATTACK_FRACTION,
                        attack_params={"scale": float(scale)},
                    ))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m experiments.gen_magnitude_sweep",
                                 description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("experiments") / "manifest_magnitude.jsonl")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--expect", type=int, default=144,
                    help="expected run count (4 scales x 4 arms x 3 models x 3 seeds)")
    args = ap.parse_args(argv)

    runs = magnitude_runs(seeds=args.seeds)
    ids = [r.run_id for r in runs]
    if len(ids) != len(set(ids)):
        raise SystemExit("run_id collision in the magnitude sweep")
    if args.expect and len(runs) != args.expect:
        raise SystemExit(f"expected {args.expect} magnitude runs, built {len(runs)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(r.to_json_line() + "\n" for r in runs), encoding="utf-8")
    print(f"wrote {len(runs)} magnitude-sweep runs -> {args.out}")
    for scale in SCALES:
        n = sum(1 for r in runs
                if float(json.loads(r.to_json_line())["config"]["attack"]["params"]["scale"])
                == scale)
        print(f"  scale={scale}: {n} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
