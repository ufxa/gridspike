"""Battery manifest: the single source of truth for every planned run.

One function -- :func:`battery_runs` -- enumerates the whole Compact battery as a
list of :class:`RunSpec`. The dry-run manifest (:mod:`experiments.run_all`) and
the per-run executor (:mod:`experiments.run_one`) therefore agree *by
construction*: the manifest emits the full ``config`` dict for every run and
``run_one`` consumes that exact dict through :func:`experiments.build.build_run`
-- there is no second place where a run is described.

Sizing (``docs/VALIDATION_STANDARD.md`` sec 3.9, re-seeded for the **Compact
battery** of CONTRACT amendments 2026-07-28e / 2026-07-28h):

* Tier-1 (headline: the 12-arm reference frame + the attack x defense grid) at
  **5 seeds** -- the Wilcoxon floor, stated openly in the manuscript.
* Tier-2 (sensitivity sweeps / ablations) at **3 seeds**.

Every intentional bound relative to the literal sec 3.9 table is recorded in
:data:`BATTERY_NOTES` and printed by the dry run -- the contract forbids silent
trimming (sec 3.9 "never silently"). The full ~1,000-run standard (Tier-1 = 10,
Tier-2 = 5) is still reproducible from the same function via ``tier1_seeds`` /
``tier2_seeds`` for the record.

Manifest generator module.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

__all__ = [
    "BATTERY_NOTES",
    "COMPACT_TIER1_SEEDS",
    "COMPACT_TIER2_SEEDS",
    "MODELS",
    "MODELS_EXT",
    "RunSpec",
    "battery_runs",
    "block_counts",
    "config_hash",
    "make_config",
]

# --------------------------------------------------------------------------- #
# Battery-wide constants (data contract + sec 3 defaults).
# --------------------------------------------------------------------------- #

K = 10  # clients, fixed by the data contract (sec 3.5 declares a K sweep out of scope)
DAYS = 30  # simulated days per client for the paper (generator default is 1 = smoke)
ROUNDS = 30  # FL rounds per run (sec 3.9 cost model)
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LR = 1e-3
# Fix for the majority-class collapse under the realistic 94%-NORMAL prior:
# Adam (was plain SGD, which barely moved off init at lr=1e-3) plus class-
# balanced cross-entropy by effective number of samples (Cui et al., CVPR 2019),
# computed from each client's OWN training labels. Both are config-driven and
# ablatable ("class_weighting": "effective_num"|"inverse"|"none").
OPTIMIZER = "adam"
CLASS_WEIGHTING = "effective_num"

COMPACT_TIER1_SEEDS = 5  # amendment 2026-07-28e: Tier-1 = 5 seeds (Wilcoxon floor)
COMPACT_TIER2_SEEDS = 3  # amendment 2026-07-28e: Tier-2 sweeps = 3 seeds

DEFAULT_ALPHA = 0.5  # Dirichlet label skew (data contract default)
IID_ALPHA = 100.0  # near-uniform Dirichlet stands in for the IID control (Hsu et al. 2019)

DEFAULT_M = 10  # PBFT validators: one per substation, so f=3 covers the 3/10 Byzantine budget
DEFAULT_F = 3  # Byzantine tolerance / Krum budget -> matches f_att = 3/10 at K=10
DEFAULT_TAU = 0.5  # fidelity admission threshold (1 - cos to the coord-wise median, in [0,2])

FULL_ATTACK_FRACTION = 0.3  # T1b Byzantine fraction: f_att = 3/10 = f_max at K=10
DEFAULT_MU = 0.01  # FedProx proximal weight (contract default; B7/B8)

MODELS = ("cnn_lstm", "snn")
# Models the default battery enumerates (every block builder iterates this). The
# SSM is a ready-to-activate contingency (a pure-PyTorch Mamba/S4D-style classifier)
# and is deliberately NOT in MODELS, so the default battery stays the fixed
# Compact set. MODELS_EXT is the superset make_config accepts, so ssm configs can
# be built on activation without ever touching the default battery -- see
# battery_runs(include_ssm=True) and experiments.run_all --include-ssm.
MODELS_EXT = ("cnn_lstm", "snn", "ssm")
ATTACKS = ("sign_flip", "gaussian_noise", "label_flip", "adaptive_cosine")
# The five defense arms swept over the attack grid (sec 3.3): no-defense, the two
# robust aggregators standing alone (no fidelity, no PBFT) and the two full
# fidelity->PBFT->agg pipelines.
DEFENSE_ARMS = ("no_defense", "krum_alone", "geomed_alone", "pipeline_krum", "pipeline_geomed")
# The twelve-arm clean reference frame is these six arms x the two models (sec 3.3, B1-B12).
REFERENCE_ARMS = ("centralized", "local", "fedavg", "fedprox", "pipeline_krum", "pipeline_geomed")

# LDP operating-curve axes (sec 3.5-S3).
LDP_SIGMAS = (0.25, 0.5, 1.0, 2.0)
LDP_TAUS = (0.3, 0.5, 0.7)
LDP_CLIP = 1.0

# Sensitivity-sweep axes.
ALPHA_SWEEP = (0.1, 1.0, "iid")  # 0.5 is the reference frame's value (sec 3.5-S1)
ATTACK_FRACTION_SWEEP = (1, 2)  # 0 = clean (T1a), 3 = f_max (T1b) (sec 3.5-S2)
VALIDATOR_SWEEP = (4, 7)  # M sweep points; 10 is the default already run by every pipeline arm
T_STEPS_SWEEP = (4, 16)  # 8 is the SNN default (sec 3.4-A4)
MU_SWEEP = (0.001, 0.1)  # 0.01 is the reference frame's FedProx value (sec 3.4-A7)

# Documented bounds / interpretations relative to the literal sec 3.9 table. Printed
# by the dry run so no cap is silent (sec 3.9: "each step recorded ... never silently").
BATTERY_NOTES: tuple[str, ...] = (
    "Seeds: Tier-1 = 5, Tier-2 = 3 (CONTRACT amendment 2026-07-28e; supersedes sec 3.9's 10/5).",
    "IID control (sec 3.5-S1) is Dirichlet alpha=100 (near-uniform), not a separate uniform "
    "sampler -- flagged so the manuscript reports it as an approximation, not exact IID.",
    "T2d validators M swept over {4,7}: M=10 is the default committee already exercised by "
    "every pipeline arm (T1a/T1b), so the two distinct new points are the smaller committees "
    "of sec 3.5-S4's {4,7,10}. Count (2 x 2 models x 3 seeds = 12) matches sec 3.9's T2d budget.",
    "S6 fp32 vs INT8 (sec 3.5-S6) is evaluation-time PTQ with no retraining and has no training "
    "block in sec 3.9; all battery runs are fp32 and INT8 is measured post-hoc on the reference "
    "3090 over the fp32 finals (phase 5). 'precision':'fp32' is stamped on every config.",
    "Re-run margin (sec 3.9, ~8% for failures/re-runs) is NOT enumerated: a re-run reuses its "
    "run_id, so it is a budget line, not a distinct config. Distinct runs x 1.08 gives the "
    "~540 the amendment quotes.",
    "net_ms for every PBFT arm is the sourced utility-WAN lognormal (simulation.network), "
    "never pbft.default_net_ms -- resolves the OPEN_QUESTIONS 'wire net_ms' item.",
)


@dataclass(frozen=True)
class RunSpec:
    """One planned run. Serialises to exactly the manifest-line schema."""

    run_id: str
    arm: str
    model: str
    seed: int
    tier: int
    block: str
    config: dict[str, Any]
    config_hash: str

    def to_json_line(self) -> str:
        """One canonical JSON line (stable key order) for ``manifest.jsonl``."""
        return json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------- #
# Canonicalisation + identifiers (pure functions of the config).
# --------------------------------------------------------------------------- #


def _canonical(config: dict[str, Any]) -> str:
    """Canonical serialisation the config hash is taken over."""
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def config_hash(config: dict[str, Any]) -> str:
    """SHA-256 of the canonical config -- stable across processes and calls."""
    return hashlib.sha256(_canonical(config).encode("utf-8")).hexdigest()


def _tag(value: Any) -> str:
    """Filesystem-safe token: floats keep precision with '.'->'p', '-'->'m'."""
    if isinstance(value, float):
        text = repr(value)
    else:
        text = str(value)
    out = text.replace(".", "p").replace("-", "m")
    return "".join(ch for ch in out if ch.isalnum() or ch in ("p", "m", "_")) or "x"


def _run_id(block: str, arm: str, model: str, seed: int, tags: Iterable[tuple[str, Any]]) -> str:
    """Deterministic, filesystem-safe id. ``tags`` are the within-block varying axes.

    ``block`` prefixes every id, so ids from different blocks can never collide;
    within a block the tags carry exactly the axes that vary, and
    :func:`battery_runs` asserts global uniqueness as a backstop.
    """
    parts = [block, arm, model, *(f"{name}{_tag(val)}" for name, val in tags), f"s{seed}"]
    return "-".join(parts)


# --------------------------------------------------------------------------- #
# Config construction -- the one place a run's config dict is defined.
# --------------------------------------------------------------------------- #


def make_config(
    *,
    model: str,
    arm: str,
    seed: int,
    alpha: float = DEFAULT_ALPHA,
    iid: bool = False,
    attack: str | None = None,
    attack_fraction: float = 0.0,
    attack_params: dict[str, Any] | None = None,
    ldp: dict[str, Any] | None = None,
    m: int = DEFAULT_M,
    f: int = DEFAULT_F,
    threshold: float = DEFAULT_TAU,
    mu: float = DEFAULT_MU,
    coding: str = "delta",
    t_steps: int = 8,
    rounds: int = ROUNDS,
    days: int = DAYS,
    k: int = K,
    local_epochs: int = LOCAL_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    optimizer: str = OPTIMIZER,
    class_weighting: str = CLASS_WEIGHTING,
    precision: str = "fp32",
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one fully-specified, JSON-native run config.

    ``arm`` selects the training mode / strategy / defense triple; everything
    else is an axis the battery sweeps. The returned dict is the complete input
    to :func:`experiments.build.build_run` -- no field is implicit.
    """
    if model not in MODELS_EXT:
        raise ValueError(f"unknown model {model!r}; expected one of {MODELS_EXT}")

    mode, strategy, defense = _arm_to_components(arm, m=m, f=f, threshold=threshold, mu=mu)

    attack_cfg: dict[str, Any] | None = None
    if attack is not None:
        if attack not in ATTACKS:
            raise ValueError(f"unknown attack {attack!r}; expected one of {ATTACKS}")
        attack_cfg = {
            "type": attack,
            "fraction": float(attack_fraction),
            "params": dict(attack_params or {}),
        }

    model_block: dict[str, Any] = {"name": model, "kwargs": dict(model_kwargs or {})}
    if model == "snn":
        model_block["coding"] = coding
        model_block["t_steps"] = int(t_steps)

    config: dict[str, Any] = {
        "model": model_block,
        "federation": {
            "k": int(k),
            "alpha": float(alpha),
            "days": int(days),
            "seed": int(seed),
            "iid": bool(iid),
        },
        "training": {
            "rounds": int(rounds),
            "local_epochs": int(local_epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "optimizer": str(optimizer),
            "class_weighting": str(class_weighting),
        },
        "mode": mode,
        "strategy": strategy,
        "defense": defense,
        "attack": attack_cfg,
        "ldp": (dict(ldp) if ldp is not None else None),
        "precision": precision,
        "seed": int(seed),
    }
    return config


def _arm_to_components(
    arm: str, *, m: int, f: int, threshold: float, mu: float
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    """Map an arm name to (training mode, strategy dict, defense dict|None)."""
    pipeline = {"kind": "pipeline", "m": int(m), "f": int(f), "threshold": float(threshold),
                "net_ms": "utility_wan"}
    if arm == "centralized":
        return "centralized", {"name": "fedavg"}, None
    if arm == "local":
        return "local", {"name": "fedavg"}, None
    if arm in ("fedavg", "no_defense"):
        return "federated", {"name": "fedavg"}, None
    if arm == "fedprox":
        return "federated", {"name": "fedprox", "mu": float(mu)}, None
    if arm == "krum_alone":
        return "federated", {"name": "krum", "f": int(f)}, None
    if arm == "geomed_alone":
        return "federated", {"name": "geomedian"}, None
    if arm == "pipeline_krum":
        return "federated", {"name": "fedavg"}, {**pipeline, "agg": "krum"}
    if arm == "pipeline_geomed":
        return "federated", {"name": "fedavg"}, {**pipeline, "agg": "geomedian"}
    if arm == "fidelity_only":
        return ("federated", {"name": "fedavg"},
                {"kind": "fidelity_only", "threshold": float(threshold)})
    raise ValueError(f"unknown arm {arm!r}")


# --------------------------------------------------------------------------- #
# The battery -- one block per sec 3.9 row.
# --------------------------------------------------------------------------- #


def _spec(block: str, arm: str, model: str, seed: int, tier: int,
          tags: list[tuple[str, Any]], **cfg_kwargs: Any) -> RunSpec:
    config = make_config(model=model, arm=arm, seed=seed, **cfg_kwargs)
    return RunSpec(
        run_id=_run_id(block, arm, model, seed, tags),
        arm=arm,
        model=model,
        seed=seed,
        tier=tier,
        block=block,
        config=config,
        config_hash=config_hash(config),
    )


def _t1a_reference(seeds: range) -> list[RunSpec]:
    """T1a -- 12-arm clean reference frame (6 arms x 2 models)."""
    out: list[RunSpec] = []
    for s in seeds:
        for arm in REFERENCE_ARMS:
            for model in MODELS:
                out.append(_spec("t1a", arm, model, s, 1, tags=[]))
    return out


def _t1a_ssm_reference(seeds: range) -> list[RunSpec]:
    """SSM activation set -- the clean reference frame re-run with model="ssm".

    Not part of the default battery: only appended when
    ``battery_runs(include_ssm=True)`` (or ``run_all --include-ssm``). Its run_ids
    live in a distinct ``t1a_ssm`` block, so they never collide with, or perturb
    the count of, the fixed Compact set. This is the one-command activation the
    contingency needs; extend it here (attack grid, ablations) if the SSM becomes
    the paper's anchor.
    """
    out: list[RunSpec] = []
    for s in seeds:
        for arm in REFERENCE_ARMS:
            out.append(_spec("t1a_ssm", arm, "ssm", s, 1, tags=[]))
    return out


def _t1b_attack_defense(seeds: range) -> list[RunSpec]:
    """T1b -- attack x defense grid at f_att = 3/10, alpha = 0.5."""
    out: list[RunSpec] = []
    for s in seeds:
        for attack in ATTACKS:
            for arm in DEFENSE_ARMS:
                for model in MODELS:
                    out.append(_spec(
                        "t1b", arm, model, s, 1,
                        tags=[("atk", attack)],
                        attack=attack, attack_fraction=FULL_ATTACK_FRACTION,
                    ))
    return out


def _t2a_alpha(seeds: range) -> list[RunSpec]:
    """T2a -- non-IID severity sweep x {FedAvg, pipeline-Krum} x {clean, adaptive}."""
    out: list[RunSpec] = []
    for s in seeds:
        for a in ALPHA_SWEEP:
            iid = a == "iid"
            alpha = IID_ALPHA if iid else float(a)
            atag = "iid" if iid else a
            for arm in ("fedavg", "pipeline_krum"):
                for model in MODELS:
                    for attack in (None, "adaptive_cosine"):
                        frac = FULL_ATTACK_FRACTION if attack else 0.0
                        out.append(_spec(
                            "t2a", arm, model, s, 2,
                            tags=[("a", atag), ("atk", attack or "clean")],
                            alpha=alpha, iid=iid,
                            attack=attack, attack_fraction=frac,
                        ))
    return out


def _t2b_attack_fraction(seeds: range) -> list[RunSpec]:
    """T2b -- attack-fraction sweep {1,2}/10 x {SignFlip, Adaptive} x {no-defense, pipeline}."""
    out: list[RunSpec] = []
    for s in seeds:
        for n_att in ATTACK_FRACTION_SWEEP:
            frac = n_att / K
            for attack in ("sign_flip", "adaptive_cosine"):
                for arm in ("no_defense", "pipeline_krum"):
                    for model in MODELS:
                        out.append(_spec(
                            "t2b", arm, model, s, 2,
                            tags=[("atk", attack), ("frac", n_att)],
                            attack=attack, attack_fraction=frac,
                        ))
    return out


def _t2c_ldp(seeds: range) -> list[RunSpec]:
    """T2c -- LDP sigma x tau operating curve on pipeline-Krum (honest LDP + adaptive attacker)."""
    out: list[RunSpec] = []
    for s in seeds:
        for sigma in LDP_SIGMAS:
            for tau in LDP_TAUS:
                for model in MODELS:
                    ldp = {"clip": LDP_CLIP, "sigma": float(sigma), "seed": int(s)}
                    out.append(_spec(
                        "t2c", "pipeline_krum", model, s, 2,
                        tags=[("sg", sigma), ("t", tau)],
                        threshold=tau, ldp=ldp,
                        attack="adaptive_cosine", attack_fraction=FULL_ATTACK_FRACTION,
                    ))
    return out


def _t2d_validators(seeds: range) -> list[RunSpec]:
    """T2d -- validator-committee sweep M in {4,7} (f = floor((M-1)/3)) under adaptive attack."""
    out: list[RunSpec] = []
    for s in seeds:
        for m in VALIDATOR_SWEEP:
            f = (m - 1) // 3
            for model in MODELS:
                out.append(_spec(
                    "t2d", "pipeline_krum", model, s, 2,
                    tags=[("m", m)],
                    m=m, f=f,
                    attack="adaptive_cosine", attack_fraction=FULL_ATTACK_FRACTION,
                ))
    return out


def _t2e_ablations(seeds: range) -> list[RunSpec]:
    """T2e -- model/strategy ablations: SNN coding, SNN T_steps, FedProx mu, fidelity-only."""
    out: list[RunSpec] = []
    for s in seeds:
        # A3 SNN coding = rate, clean + adaptive (delta is the reference-frame default;
        # this ablation shows rate coding degrades the spiking classifier toward the
        # majority-class baseline on fault waveforms -- a real, reported finding).
        for attack in (None, "adaptive_cosine"):
            frac = FULL_ATTACK_FRACTION if attack else 0.0
            out.append(_spec(
                "t2e", "fedavg", "snn", s, 2,
                tags=[("coding", "rate"), ("atk", attack or "clean")],
                coding="rate", attack=attack, attack_fraction=frac,
            ))
        # A4 SNN T_steps in {4,16}, clean + adaptive.
        for t in T_STEPS_SWEEP:
            for attack in (None, "adaptive_cosine"):
                frac = FULL_ATTACK_FRACTION if attack else 0.0
                out.append(_spec(
                    "t2e", "fedavg", "snn", s, 2,
                    tags=[("ts", t), ("atk", attack or "clean")],
                    t_steps=t, attack=attack, attack_fraction=frac,
                ))
        # A7 FedProx mu in {0.001, 0.1} x both models (0.01 is the reference frame).
        for mu in MU_SWEEP:
            for model in MODELS:
                out.append(_spec(
                    "t2e", "fedprox", model, s, 2,
                    tags=[("mu", mu)],
                    mu=mu,
                ))
        # A1 fidelity-only rung x both models, under the adaptive attacker.
        for model in MODELS:
            out.append(_spec(
                "t2e", "fidelity_only", model, s, 2,
                tags=[("atk", "adaptive_cosine")],
                attack="adaptive_cosine", attack_fraction=FULL_ATTACK_FRACTION,
            ))
    return out


_BLOCK_BUILDERS = (
    ("t1a", 1, _t1a_reference),
    ("t1b", 1, _t1b_attack_defense),
    ("t2a", 2, _t2a_alpha),
    ("t2b", 2, _t2b_attack_fraction),
    ("t2c", 2, _t2c_ldp),
    ("t2d", 2, _t2d_validators),
    ("t2e", 2, _t2e_ablations),
)


def battery_runs(
    tier1_seeds: int = COMPACT_TIER1_SEEDS,
    tier2_seeds: int = COMPACT_TIER2_SEEDS,
    *,
    include_ssm: bool = False,
) -> list[RunSpec]:
    """The whole battery as an ordered list of :class:`RunSpec` (the source of truth).

    Defaults size the Compact battery (amendments e/h); pass ``tier1_seeds=10,
    tier2_seeds=5`` to reproduce the full sec 3.9 standard. Asserts every ``run_id``
    is unique -- a collision means a within-block varying axis is missing from the
    id and is a build-time error, never a silent overwrite on the fleet.

    ``include_ssm`` (default False -> the fixed Compact set, unchanged) appends the
    ``t1a_ssm`` activation block (the clean reference frame with the SSM
    classifier) -- the ready-to-activate contingency arm. It is keyword-only so the
    positional full-standard call ``battery_runs(10, 5)`` is unaffected.
    """
    if tier1_seeds < 1 or tier2_seeds < 1:
        raise ValueError("seed counts must be >= 1")
    t1 = range(tier1_seeds)
    t2 = range(tier2_seeds)
    runs: list[RunSpec] = []
    for _name, tier, builder in _BLOCK_BUILDERS:
        runs.extend(builder(t1 if tier == 1 else t2))
    if include_ssm:
        runs.extend(_t1a_ssm_reference(t1))

    ids = [r.run_id for r in runs]
    dupes = [rid for rid, n in Counter(ids).items() if n > 1]
    if dupes:
        raise AssertionError(f"run_id collision(s): {dupes[:5]} ({len(dupes)} total)")
    return runs


def block_counts(runs: list[RunSpec]) -> dict[str, int]:
    """Per-block run counts in block order, plus a 'TOTAL' key."""
    counts: dict[str, int] = {name: 0 for name, _tier, _b in _BLOCK_BUILDERS}
    for r in runs:
        counts[r.block] = counts.get(r.block, 0) + 1
    counts["TOTAL"] = len(runs)
    return counts


# --------------------------------------------------------------------------- #
# Tiny CLI: `python -m experiments.manifest` prints the block table.
# --------------------------------------------------------------------------- #


def _format_table(runs: list[RunSpec]) -> str:
    counts = block_counts(runs)
    lines = ["block   tier  runs", "-----   ----  ----"]
    tier_of = {name: tier for name, tier, _b in _BLOCK_BUILDERS}
    for name, _tier, _b in _BLOCK_BUILDERS:
        lines.append(f"{name:<6}  {tier_of[name]:>4}  {counts[name]:>4}")
    lines.append(f"{'TOTAL':<6}  {'':>4}  {counts['TOTAL']:>4}")
    margin = round(counts["TOTAL"] * 0.08)
    lines.append(f"(+ ~8% re-run margin ~= {margin}  ->  ~= {counts['TOTAL'] + margin} all-in)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m experiments.manifest",
                                description="Print the Compact battery block table.")
    p.add_argument("--tier1-seeds", type=int, default=COMPACT_TIER1_SEEDS)
    p.add_argument("--tier2-seeds", type=int, default=COMPACT_TIER2_SEEDS)
    args = p.parse_args(argv)
    runs = battery_runs(args.tier1_seeds, args.tier2_seeds)
    print(_format_table(runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
