"""Turn one manifest config dict into the concrete objects a run needs.

This is the second half of the manifest's single source of truth: everything the
grid varies is a key in the config (:func:`experiments.manifest.make_config`),
and every key is interpreted here -- nothing about a run is implicit. ``run_one``
calls :func:`build_run` on the exact config the manifest emitted, so the plan and
the execution cannot drift.

Three arms of the reference frame are not plain federated training and are handled
as training *modes*:

* ``centralized`` -- pool every client's split into one client (the accuracy
  ceiling); the engine then runs ordinary SGD on the union.
* ``local`` -- train each client alone with no aggregation and report the
  mean/min/max over the K clients (the accuracy floor).
* ``federated`` -- the normal engine path (all the FedAvg/FedProx/robust arms).

The Krum-alone and geometric-median-alone arms are ``Strategy`` objects (robust
aggregation with *no* fidelity gate and *no* PBFT), and the fidelity-only arm is a
``ValidatorFn`` that gates on the fidelity score alone -- both live here because
they are experiment glue over the phase 2 public API, not new library behaviour.

Experiment-build module. Heavy imports (torch, fedgrid.*) are function-local
so the module and the manifest stay torch-free at import.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fedgrid.fl.engine import ValidationReport

if TYPE_CHECKING:
    from torch import nn
    from torch.utils.data import Dataset

    from fedgrid.attacks.base import Attack
    from fedgrid.data import Federation
    from fedgrid.fl.engine import ClientUpdate, ValidatorFn
    from fedgrid.fl.ldp import LDPConfig
    from fedgrid.fl.strategies import Strategy
    from fedgrid.metrics.runresult import RunResult

__all__ = ["BuiltRun", "build_run", "load_or_make_federation"]


# --------------------------------------------------------------------------- #
# Strategy / validator adapters over the phase 2 public API.
# --------------------------------------------------------------------------- #


class _KrumStrategy:
    """Krum *selection* as a plain ``Strategy`` -- the Krum-alone arm.

    No fidelity score, no PBFT: the engine hands every client update (attackers
    included) straight to Krum, which returns the single most central one. This
    is exactly the "what does the consensus layer buy over the robust aggregator
    alone" arm of VALIDATION_STANDARD sec 3.3.
    """

    def __init__(self, f: int) -> None:
        self.f = int(f)

    def aggregate(
        self, updates: list[ClientUpdate], current: list[np.ndarray]
    ) -> list[np.ndarray]:
        from fedgrid.consensus.aggregation import krum

        if not updates:
            return [np.asarray(a).copy() for a in current]
        return [np.asarray(p).copy() for p in updates[krum(updates, self.f)].params]


class _GeoMedianStrategy:
    """Geometric median as a plain ``Strategy`` -- the geomedian-alone arm."""

    def aggregate(
        self, updates: list[ClientUpdate], current: list[np.ndarray]
    ) -> list[np.ndarray]:
        from fedgrid.consensus.aggregation import geometric_median

        if not updates:
            return [np.asarray(a).copy() for a in current]
        return geometric_median(updates)


class _FidelityOnlyValidator:
    """Admit updates whose fidelity score is <= tau; no PBFT, no robust aggregate.

    Returns a bare ``ValidationReport`` (``consensus=None`` and no ``aggregate``),
    so the engine falls back to its ``Strategy`` (FedAvg) over the admitted set --
    the fidelity-only rung of the sec 3.4-A1 defense decomposition.
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = float(threshold)

    def __call__(
        self, updates: list[ClientUpdate], current: list[np.ndarray]
    ) -> ValidationReport:
        from fedgrid.consensus.fidelity import fidelity_scores

        scores = np.asarray(fidelity_scores(updates, ref=current), dtype=np.float64)
        accepted = [u.cid for i, u in enumerate(updates) if float(scores[i]) <= self.threshold]
        return ValidationReport(accepted=accepted, scores=scores, consensus=None)


# --------------------------------------------------------------------------- #
# Federation adapters for the non-federated training modes.
# --------------------------------------------------------------------------- #


@dataclass
class _PooledClient:
    train: Dataset
    val: Dataset
    test: Dataset
    meta: Any = None

    def __len__(self) -> int:  # engine reads len(client.train), not len(client)
        return len(self.train)  # type: ignore[arg-type]


class _PooledFederation:
    """Single client holding every real client's data -- the centralized ceiling."""

    def __init__(self, fed: Federation) -> None:
        from torch.utils.data import ConcatDataset

        self.clients = [
            _PooledClient(
                train=ConcatDataset([c.train for c in fed.clients]),
                val=ConcatDataset([c.val for c in fed.clients]),
                test=ConcatDataset([c.test for c in fed.clients]),
            )
        ]
        self._fed = fed

    def global_test(self) -> Dataset:
        return self._fed.global_test()


class _SingleClientFederation:
    """One real client as its own federation -- one run of the local floor.

    Training sees only this client's data, but evaluation MUST use the same global test
    set as every other arm: a local model scored on its own skewed test split is measured
    on the distribution it trained on, which inflates the floor above the federated and
    centralized arms and makes the frame incomparable (observed 2026-07-28 on the t1a
    block before this fix). ``global_test`` therefore returns the federation-wide set.
    """

    def __init__(self, client: Any, global_test: Dataset) -> None:
        self.clients = [client]
        self._test = global_test

    def global_test(self) -> Dataset:
        return self._test


# --------------------------------------------------------------------------- #
# Config -> objects.
# --------------------------------------------------------------------------- #


def make_model_fn(config: dict[str, Any]) -> Callable[[], nn.Module]:
    """Model factory from the config's model block (fresh module per client step)."""
    m = config["model"]
    name = m["name"]
    kwargs = dict(m.get("kwargs", {}))
    if name == "cnn_lstm":
        from fedgrid.models.cnn_lstm import CNNLSTM

        return lambda: CNNLSTM(**kwargs)
    if name == "snn":
        from fedgrid.models.snn import SpikingClassifier

        coding = m.get("coding", "rate")
        t_steps = int(m.get("t_steps", 8))
        # threshold_init=0.25 keeps both LIF layers firing at 8-10%; the old
        # default 1.0 starves the second layer (zero spikes, logits std 0.03),
        # which the federated SNN could not recover from -> collapse to the
        # majority-class floor. See models/snn.py init-regime note.
        threshold_init = float(m.get("threshold_init", 0.25))
        seed = int(config["seed"])
        return lambda: SpikingClassifier(coding=coding, T_steps=t_steps,
                                         threshold_init=threshold_init, seed=seed, **kwargs)
    if name == "ssm":
        # Selective state-space classifier (Mamba/S4D-style, pure PyTorch). A
        # ready-to-activate third arm; no ssm arms are in the default battery yet
        # (see experiments.manifest MODELS_EXT / battery_runs(include_ssm=...)).
        from fedgrid.models.ssm import SSMClassifier

        seed = int(config["seed"])
        return lambda: SSMClassifier(seed=seed, **kwargs)
    raise ValueError(f"unknown model {name!r}")


def make_strategy(config: dict[str, Any]) -> Strategy:
    """FedAvg / FedProx / Krum-alone / geomedian-alone from the strategy block."""
    from fedgrid.fl.strategies import FedAvg, FedProx

    s = config["strategy"]
    name = s["name"]
    if name == "fedavg":
        return FedAvg()
    if name == "fedprox":
        return FedProx(mu=float(s.get("mu", 0.01)))
    if name == "krum":
        return _KrumStrategy(f=int(s.get("f", 3)))
    if name == "geomedian":
        return _GeoMedianStrategy()
    raise ValueError(f"unknown strategy {name!r}")


def make_validator(config: dict[str, Any]) -> ValidatorFn | None:
    """Pipeline (fidelity->PBFT->agg) or fidelity-only validator, or None.

    The pipeline arm always injects the sourced utility-WAN lognormal as
    ``net_ms`` (never ``pbft.default_net_ms``), closing the OPEN_QUESTIONS
    "wire net_ms" item -- no published latency number may use the placeholder.
    """
    d = config.get("defense")
    if d is None:
        return None
    kind = d["kind"]
    if kind == "pipeline":
        from fedgrid.consensus.pipeline import make_validator as make_pipeline_validator

        net = None
        if d.get("net_ms") == "utility_wan":
            from fedgrid.simulation.network import utility_wan_net_ms

            net = utility_wan_net_ms
        return make_pipeline_validator(
            m=int(d["m"]), f=int(d["f"]), threshold=float(d["threshold"]),
            agg=str(d["agg"]), seed=int(config["seed"]), net_ms=net,
        )
    if kind == "fidelity_only":
        return _FidelityOnlyValidator(threshold=float(d["threshold"]))
    raise ValueError(f"unknown defense kind {kind!r}")


def make_attacks(config: dict[str, Any]) -> dict[int, Attack] | None:
    """cid -> attack for the last ``round(fraction*K)`` clients (the PBFT faulty tail).

    One shared attack instance drives every attacker: it is stateless for the
    naive attacks and, for AdaptiveCosine, the documented colluding form (same
    ``collude_seed``, round-keyed public-state cache) that lands the coalition on
    one cone point instead of scattering -- the threat model f=3/10 means.
    """
    a = config.get("attack")
    if a is None:
        return None
    k = int(config["federation"]["k"])
    seed = int(config["seed"])
    n_att = int(round(float(a["fraction"]) * k))
    if n_att <= 0:
        return None
    cids = list(range(k - n_att, k))
    params = dict(a.get("params", {}))
    kind = a["type"]
    if kind == "sign_flip":
        from fedgrid.attacks.suite import SignFlip

        atk: Attack = SignFlip(scale=float(params.get("scale", 1.0)))
    elif kind == "gaussian_noise":
        from fedgrid.attacks.suite import GaussianNoise

        atk = GaussianNoise(sigma=float(params.get("sigma", 1.0)))
    elif kind == "label_flip":
        from fedgrid.attacks.suite import LabelFlip

        # frac = fraction of an attacker's own labels flipped (see OPEN_QUESTIONS:
        # the engine re-flips per batch via `.frac`, not LabelFlip.flip's static set).
        atk = LabelFlip(frac=float(params.get("frac", 1.0)), seed=seed)  # type: ignore[assignment]
    elif kind == "adaptive_cosine":
        from fedgrid.attacks.suite import AdaptiveCosine

        atk = AdaptiveCosine(
            target_score=float(params.get("target_score", 0.1)),
            scale=float(params.get("scale", 1.0)),
            collude_seed=seed,
            n_peers=k,
        )
    else:
        raise ValueError(f"unknown attack type {kind!r}")
    return dict.fromkeys(cids, atk)


def make_ldp(config: dict[str, Any]) -> LDPConfig | None:
    """LDPConfig from the ldp block, or None."""
    ldp = config.get("ldp")
    if ldp is None:
        return None
    from fedgrid.fl.ldp import LDPConfig

    return LDPConfig(
        clip=float(ldp["clip"]), sigma=float(ldp["sigma"]), seed=int(ldp.get("seed", 0))
    )


def _federation_manifest_matches(manifest_path: Path, k: int, alpha: float,
                                 days: int, seed: int) -> bool:
    """True iff a generator ``manifest.json`` was built with exactly (k, alpha, days, seed)."""
    if not manifest_path.exists():
        return False
    import json

    try:
        cfg = json.loads(manifest_path.read_text(encoding="utf-8")).get("config", {})
    except (OSError, ValueError):
        return False
    return (
        int(cfg.get("k", -1)) == k
        and abs(float(cfg.get("alpha", -1.0)) - alpha) < 1e-9
        and int(cfg.get("days", -1)) == days
        and int(cfg.get("seed", -1)) == seed
    )


def find_federation_dir(config: dict[str, Any], data_dir: Path | str | None) -> Path | None:
    """Resolve the on-disk federation directory for ``config``, or None to generate.

    Two supported layouts (pure path logic; loads nothing):
      1. ``data_dir`` is itself one generated federation (``data_dir/manifest.json`` matches).
      2. ``data_dir`` is a ROOT holding several ``<fed_id>/`` subdirs (the phase 4 staging
         layout) -- the subdir whose ``manifest.json`` matches (k, alpha, days, seed) wins.
    So one worker can carry all 14 battery federations under a single ``--data-dir`` and every
    run loads its own without regenerating OpenDSS data in-process.
    """
    if data_dir is None:
        return None
    fcfg = config["federation"]
    k, alpha, days, seed = (
        int(fcfg["k"]), float(fcfg["alpha"]), int(fcfg["days"]), int(fcfg["seed"]),
    )
    dd = Path(data_dir)
    if _federation_manifest_matches(dd / "manifest.json", k, alpha, days, seed):
        return dd
    if dd.is_dir():
        for sub in sorted(dd.iterdir()):
            if sub.is_dir() and _federation_manifest_matches(
                sub / "manifest.json", k, alpha, days, seed
            ):
                return sub
    return None


def _load_real_federation(
    fcfg: dict[str, Any], data_dir: Path | str | None, log: Callable[[str], None]
) -> Federation:
    """Resolve a real-measurement federation (``federation.source != "synthetic"``).

    Currently only ``"msu"`` (the MSU/ORNL Power System Attack Datasets, the
    external-validity section). The raw data directory is taken from
    ``federation.real_data_dir``, else ``--data-dir``, else ``$FEDGRID_MSU_DIR`` --
    the loader parses/caches the ARFF windows and partitions them into K clients
    (see :mod:`fedgrid.data_real`). Kept entirely separate from the synthetic
    generate/load path so no OpenDSS code is touched.
    """
    import os

    from fedgrid.data_real import load_msu_federation

    source = str(fcfg["source"])
    if source != "msu":
        raise ValueError(f"unknown federation source {source!r}; expected 'synthetic' or 'msu'")
    raw = fcfg.get("real_data_dir") or data_dir or os.environ.get("FEDGRID_MSU_DIR")
    if raw is None:
        raise ValueError("real federation needs federation.real_data_dir, --data-dir, "
                         "or $FEDGRID_MSU_DIR")
    k, seed = int(fcfg["k"]), int(fcfg["seed"])
    stride = int(fcfg.get("stride", 8))
    partition = str(fcfg.get("partition", "shard"))
    alpha = float(fcfg.get("alpha", 0.5))
    log(f"[build] loading MSU real-data federation from {raw} (k={k}, seed={seed}, "
        f"stride={stride}, partition={partition})")
    return load_msu_federation(raw, k=k, seed=seed, stride=stride, partition=partition,
                               alpha=alpha)


def load_or_make_federation(
    config: dict[str, Any], data_dir: Path | str | None, log: Callable[[str], None] = print
) -> Federation:
    """Resolve the federation for ``config``, honouring ``federation.split_mode``.

    The data itself is resolved by :func:`_resolve_federation` (pre-generated dir,
    in-process generation, or a real dataset). ``federation.split_mode`` then selects
    the train/val/test protocol *at load*, with no regeneration:

    * ``"random"`` (default, and what every run before 2026-07-30 did) -- the seeded
      permutation split persisted in the ``.npz`` files, used bit-identically.
    * ``"temporal"`` -- contiguous chronological blocks per client with a four-window
      purge at each boundary (:func:`fedgrid.data.apply_temporal_split`), the split
      that does not leak 75 %-overlapping windows across train/test.
    """
    from fedgrid.data import apply_split_mode

    fcfg = config["federation"]
    split_mode = str(fcfg.get("split_mode", "random"))
    fed = _resolve_federation(config, data_dir, log)
    if split_mode != "random":
        log(f"[build] re-splitting federation at load (split_mode={split_mode})")
    return apply_split_mode(fed, split_mode)


def _resolve_federation(
    config: dict[str, Any], data_dir: Path | str | None, log: Callable[[str], None]
) -> Federation:
    """Load the pre-generated federation matching (k, alpha, days, seed), else build it.

    Accepts either a single-federation ``data_dir`` or a root of ``<fed_id>/`` subdirs
    (see :func:`find_federation_dir`); on a match we reload identical splits/normalisation
    with no recompute, otherwise we generate the federation in-process with a loud log line
    -- never silently. When ``federation.source`` is a real dataset (e.g. ``"msu"``) the
    real-data loader is used instead (:func:`_load_real_federation`).
    """
    fcfg = config["federation"]
    if str(fcfg.get("source", "synthetic")) != "synthetic":
        return _load_real_federation(fcfg, data_dir, log)
    k, alpha, days, seed = (
        int(fcfg["k"]), float(fcfg["alpha"]), int(fcfg["days"]), int(fcfg["seed"]),
    )
    found = find_federation_dir(config, data_dir)
    if found is not None:
        from fedgrid.data import load_federation

        log(f"[build] loading federation from {found} (k={k}, alpha={alpha}, "
            f"days={days}, seed={seed})")
        return load_federation(found)
    log(f"[build] generating federation in-process (k={k}, alpha={alpha}, days={days}, "
        f"seed={seed}) -- no matching pre-generated data in {data_dir}")
    from fedgrid.simulation.partition import make_federation

    return make_federation(k=k, alpha=alpha, days=days, seed=seed)


# --------------------------------------------------------------------------- #
# BuiltRun: the executable form of one config.
# --------------------------------------------------------------------------- #


@dataclass
class BuiltRun:
    """One config, resolved against data, ready to execute into a ``RunResult``."""

    config: dict[str, Any]
    federation: Federation
    device: str
    limit_rounds: int | None
    log: Callable[[str], None]

    def _rounds(self) -> int:
        r = int(self.config["training"]["rounds"])
        return r if self.limit_rounds is None else min(r, int(self.limit_rounds))

    def run(self, on_round: Callable[[dict], None] | None = None) -> RunResult:
        mode = self.config["mode"]
        if mode == "local":
            return self._run_local()
        return self._run_engine(on_round)

    # ---------------------------------------------------------------- engine
    def _run_engine(self, on_round: Callable[[dict], None] | None) -> RunResult:
        from fedgrid.fl.engine import FederatedEngine

        cfg = self.config
        tr = cfg["training"]
        fed = (_PooledFederation(self.federation)
               if cfg["mode"] == "centralized" else self.federation)
        engine = FederatedEngine(
            fed,
            make_model_fn(cfg),
            make_strategy(cfg),
            rounds=self._rounds(),
            local_epochs=int(tr["local_epochs"]),
            batch_size=int(tr["batch_size"]),
            lr=float(tr["lr"]),
            # Battery defaults (adam + effective-number weighting) fix the
            # majority-class collapse under the 94%-NORMAL prior; .get keeps
            # pre-field configs working while defaulting them to the fix.
            optimizer=str(tr.get("optimizer", "adam")),
            class_weighting=str(tr.get("class_weighting", "effective_num")),
            ldp=make_ldp(cfg),
            attacks=make_attacks(cfg),
            validator=make_validator(cfg),
            device=self.device,
            seed=int(cfg["seed"]),
            on_round=on_round,
        )
        result = engine.run()
        correct, energy = self._final_eval(
            make_model_fn(cfg), engine.global_params,
            want_energy=(cfg["model"]["name"] == "snn"),
        )
        if correct is not None:
            # Per-sample correctness on the (deterministically ordered) global test
            # -- the matched-accuracy McNemar (H1a) pairs two models by test index at
            # a shared seed; the confusion matrix alone cannot supply the discordance.
            result.final["correct"] = correct
            result.final["n_test"] = len(correct)
        if energy is not None:
            result.energy = energy
        return result

    # ----------------------------------------------------------------- local
    def _run_local(self) -> RunResult:
        """Independent per-client training; final = mean/min/max over K clients."""
        from fedgrid.fl.strategies import FedAvg

        cfg = self.config
        tr = cfg["training"]
        from fedgrid.fl.engine import FederatedEngine

        finals: list[dict[str, Any]] = []
        for cid, client in enumerate(self.federation.clients):
            engine = FederatedEngine(
                _SingleClientFederation(client, self.federation.global_test()),
                make_model_fn(cfg),
                FedAvg(),
                rounds=self._rounds(),
                local_epochs=int(tr["local_epochs"]),
                batch_size=int(tr["batch_size"]),
                lr=float(tr["lr"]),
                optimizer=str(tr.get("optimizer", "adam")),
                class_weighting=str(tr.get("class_weighting", "effective_num")),
                device=self.device,
                seed=int(cfg["seed"]),
            )
            res = engine.run()
            finals.append({"cid": cid, "acc": res.final["acc"], "macro_f1": res.final["macro_f1"]})

        f1s = [float(p["macro_f1"]) for p in finals]
        accs = [float(p["acc"]) for p in finals]
        final = {
            "acc": float(np.mean(accs)),
            "macro_f1": float(np.mean(f1s)),
            "local_mean_macro_f1": float(np.mean(f1s)),
            "local_min_macro_f1": float(np.min(f1s)),
            "local_max_macro_f1": float(np.max(f1s)),
            "local_std_macro_f1": float(np.std(f1s)),
            "per_client": finals,
            "split": "local_test",
        }
        return self._local_result(final)

    def _local_result(self, final: dict[str, Any]) -> RunResult:
        import dataclasses

        import fedgrid
        from fedgrid.metrics.runresult import (
            RunResult,
            current_env_fingerprint,
            current_git_commit,
        )

        root = Path(fedgrid.__file__).resolve().parents[2]
        kwargs: dict[str, Any] = {
            "method": "local",
            "seed": int(self.config["seed"]),
            "config": {"strategy": "local", "rounds": self._rounds()},
            "rounds": [],
            "final": final,
            "energy": None,
            "latency": None,
        }
        present = {f.name for f in dataclasses.fields(RunResult)}
        if "git_commit" in present:
            kwargs["git_commit"] = current_git_commit(root)
        if "env_fingerprint" in present:
            kwargs["env_fingerprint"] = current_env_fingerprint()
        return RunResult(**kwargs)

    # ------------------------------------------------------ final eval + energy
    def _final_eval(
        self, model_fn: Callable[[], nn.Module], params: list[np.ndarray], want_energy: bool
    ) -> tuple[list[int] | None, dict[str, float] | None]:
        """One pass over the global test: per-sample correctness (+ SNN energy).

        Correctness is a 0/1 list aligned with the deterministically ordered global
        test, so two models at the same seed pair sample-for-sample for the H1a
        McNemar. When ``want_energy`` (SNN) the same pass reads the SynOps counters
        and estimates per-inference energy against the CNN-LSTM MAC reference
        (Horowitz constants). Fully guarded: any failure degrades to ``(None, None)``
        -- a SynOps-less or torch-less path must never fabricate a number.
        """
        try:
            import torch
            from torch.utils.data import DataLoader

            model = model_fn().to(self.device)
            with torch.no_grad():
                for p, arr in zip(model.parameters(), params, strict=True):
                    p.copy_(torch.as_tensor(arr, dtype=p.dtype, device=self.device))
            reset = getattr(model, "reset_counters", None)
            if callable(reset):
                try:
                    reset()  # do not let a dry/earlier pass pollute the SynOps count
                except NotImplementedError:
                    pass
            model.eval()

            correct: list[int] = []
            n = 0
            loader = DataLoader(self.federation.global_test(), batch_size=128, shuffle=False)
            with torch.no_grad():
                for xb, yb in loader:
                    pred = model(xb.to(self.device)).argmax(dim=-1).cpu().numpy()
                    truth = yb.cpu().numpy()
                    correct.extend(int(a == b) for a, b in zip(pred, truth, strict=True))
                    n += int(truth.shape[0])
            if n == 0:
                return None, None
            energy = self._energy_from(model, n) if want_energy else None
            return correct, energy
        except (NotImplementedError, ValueError, RuntimeError):
            return None, None

    def _energy_from(self, model: nn.Module, n: int) -> dict[str, float] | None:
        """Estimated per-inference energy from a model's SynOps counters, guarded."""
        synops = getattr(model, "synops", None)
        if not callable(synops):
            return None
        try:
            from fedgrid.metrics.energy import energy_uj
            from fedgrid.models.cnn_lstm import CNNLSTM

            d = synops()
            per_inf_synops = float(d["synops"]) / n
            ann_macs = float(CNNLSTM().macs_per_inference())
            e = energy_uj(per_inf_synops, ann_macs)
            return {
                "synops_per_inference": per_inf_synops,
                "snn_dense_macs_per_inference": float(d.get("dense_macs", 0.0)) / n,
                "snn_internal_ratio": float(d.get("ratio", 0.0)),
                "spike_rate": float(d.get("spike_rate", 0.0)),
                "ann_cnn_lstm_macs": ann_macs,
                "snn_uJ": e["snn_uJ"],
                "ann_uJ": e["ann_uJ"],
                "ratio": e["ratio"],
                "n_samples": n,
            }
        except (NotImplementedError, ValueError, RuntimeError):
            return None


def build_run(
    config: dict[str, Any],
    data_dir: Path | str | None = None,
    device: str = "cpu",
    limit_rounds: int | None = None,
    log: Callable[[str], None] = print,
) -> BuiltRun:
    """Resolve a config against ``data_dir`` and return an executable :class:`BuiltRun`."""
    federation = load_or_make_federation(config, data_dir, log=log)
    return BuiltRun(
        config=config, federation=federation, device=device,
        limit_rounds=limit_rounds, log=log,
    )
