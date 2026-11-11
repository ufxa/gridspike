"""Real-data (MSU/ORNL) external-validity battery -- a compact, honest run set.

The synthetic Compact battery (``experiments.manifest``) is ~500 runs; this is
its small external-validity companion: the clean reference-frame arms x the three
classifiers x a few seeds, run on **real relay/PMU measurements** instead of
OpenDSS output, so the paper can show the whole pipeline transfers unmodified.
It is deliberately not a full attack x defense grid -- it is a validation section.

What varies:

* **arms** -- the six clean reference-frame arms (:data:`REAL_ARMS`): the
  centralized ceiling, the local floor, FedAvg, FedProx, and the two robust
  fidelity->PBFT->aggregation pipelines (Krum / geometric median). Same protocol,
  same committee/Byzantine settings as the synthetic frame.
* **models** -- all three classifiers (``cnn_lstm``, ``snn``, ``ssm``); the point
  is that none of them is modified for real data.
* **seeds** -- a few (default 3), enough for a mean +/- spread on a validation set.

Everything else is pinned to the real-data federation: ``federation.source =
"msu"``, ``partition`` (``shard`` -> K=10 for protocol parity, or ``relay`` -> the
honest K=4 vantage-point split), a documented ``stride``, and the extracted-data
directory. Results are collected under ``runs_real/`` (kept apart from the
synthetic ``runs/``). Owner: Real-Data Integration agent.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments.manifest import RunSpec, config_hash, make_config

__all__ = [
    "DEFAULT_MSU_DIR",
    "REAL_ARMS",
    "REAL_MODELS",
    "make_real_config",
    "real_battery_runs",
    "write_real_manifest",
]

# The six clean reference-frame arms (no attack): ceiling, floor, FedAvg, FedProx,
# and the two robust pipelines. Matches experiments.manifest.REFERENCE_ARMS.
REAL_ARMS: tuple[str, ...] = (
    "centralized", "local", "fedavg", "fedprox", "pipeline_krum", "pipeline_geomed",
)
REAL_MODELS: tuple[str, ...] = ("cnn_lstm", "snn", "ssm")

# Server location of the extracted MSU/ORNL ARFF files (data_real.msu_shard_windows
# reads <dir>/extracted/multiclass/*.arff). Overridable per run via --real-data-dir.
DEFAULT_MSU_DIR = "/root/fedgrid/data_real_msu"

# Real-data federation defaults. K=10 with the shard partition keeps the PBFT
# committee (m=10) and Byzantine budget (f=3) identical to the synthetic frame; the
# relay partition forces K=4 (one client per relay) and pairs with m=4/f=1.
_DEFAULT_ROUNDS = 30
_RELAY_K = 4


def make_real_config(
    *,
    model: str,
    arm: str,
    seed: int,
    real_data_dir: str = DEFAULT_MSU_DIR,
    partition: str = "shard",
    k: int = 10,
    stride: int = 8,
    rounds: int = _DEFAULT_ROUNDS,
    m: int = 10,
    f: int = 3,
) -> dict[str, Any]:
    """A synthetic-battery config re-pointed at the real MSU federation.

    Reuses :func:`experiments.manifest.make_config` (so the model/strategy/defense
    wiring is byte-identical to the synthetic path) and then overwrites the
    ``federation`` block to select the real-data source. ``partition="relay"``
    forces K=4 and a proportionate committee (m=4, f=1).
    """
    if partition == "relay":
        k, m, f = _RELAY_K, _RELAY_K, 1
    cfg = make_config(
        model=model, arm=arm, seed=seed, k=k, days=0, rounds=rounds, m=m, f=f,
    )
    cfg["federation"] = {
        "k": int(k),
        "alpha": 0.5,            # bookkeeping only; no Dirichlet skew imposed (data_real docstring)
        "days": 0,
        "seed": int(seed),
        "iid": False,
        "source": "msu",
        "real_data_dir": str(real_data_dir),
        "partition": str(partition),
        "stride": int(stride),
    }
    return cfg


def _run_id(partition: str, arm: str, model: str, seed: int) -> str:
    """Filesystem-safe, collision-free id (block prefix keeps it distinct)."""
    return f"real-{partition}-{arm}-{model}-s{seed}"


def real_battery_runs(
    *,
    seeds: int = 3,
    partition: str = "shard",
    real_data_dir: str = DEFAULT_MSU_DIR,
    k: int = 10,
    stride: int = 8,
    rounds: int = _DEFAULT_ROUNDS,
    models: tuple[str, ...] = REAL_MODELS,
    arms: tuple[str, ...] = REAL_ARMS,
) -> list[RunSpec]:
    """Enumerate the real-data battery as ``RunSpec``s (asserts unique run ids)."""
    if seeds < 1:
        raise ValueError("seeds must be >= 1")
    runs: list[RunSpec] = []
    for s in range(seeds):
        for arm in arms:
            for model in models:
                cfg = make_real_config(
                    model=model, arm=arm, seed=s, real_data_dir=real_data_dir,
                    partition=partition, k=k, stride=stride, rounds=rounds,
                )
                runs.append(RunSpec(
                    run_id=_run_id(partition, arm, model, s),
                    arm=arm, model=model, seed=s, tier=1, block="real",
                    config=cfg, config_hash=config_hash(cfg),
                ))
    ids = [r.run_id for r in runs]
    if len(set(ids)) != len(ids):
        raise AssertionError("real battery run_id collision")
    return runs


def write_real_manifest(path: Path, **kwargs: Any) -> int:
    """Write one JSON manifest line per real-data run; returns the count."""
    runs = real_battery_runs(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for spec in runs:
            fh.write(spec.to_json_line())
            fh.write("\n")
    return len(runs)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m experiments.manifest_real",
        description="Enumerate the MSU/ORNL real-data external-validity battery.",
    )
    p.add_argument("--out", type=Path, default=Path("experiments") / "manifest_real.jsonl")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--partition", choices=["shard", "relay"], default="shard")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--rounds", type=int, default=_DEFAULT_ROUNDS)
    p.add_argument("--real-data-dir", default=DEFAULT_MSU_DIR)
    args = p.parse_args(argv)

    n = write_real_manifest(
        args.out, seeds=args.seeds, partition=args.partition, k=args.k,
        stride=args.stride, rounds=args.rounds, real_data_dir=args.real_data_dir,
    )
    print(f"real-data battery: {n} runs "
          f"({len(REAL_ARMS)} arms x {len(REAL_MODELS)} models x {args.seeds} seeds), "
          f"partition={args.partition}, K={_RELAY_K if args.partition == 'relay' else args.k}, "
          f"stride={args.stride}, rounds={args.rounds}")
    print(f"wrote {n} manifest lines -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
