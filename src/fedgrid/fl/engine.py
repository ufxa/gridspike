"""Federated engine: local training, update collection, validation, aggregation.

Native in-process engine — a deliberate design decision ratified by contract
amendment 2026-07-28 §5. The whole federation (K local trainers, per-client
attacks and LDP, validation, aggregation, evaluation) runs sequentially in one
process under one seeded RNG tree, which is what makes runs bit-reproducible
on CPU. Flower (Beutel et al., "Flower: A Friendly Federated Learning
Research Framework", arXiv:2007.14390) is the ecosystem standard and is cited
here as the *declined* alternative: a distributed runtime cannot guarantee
bit-level determinism of the attack/consensus pipeline under seeded RNG, and
the Byzantine-FL canon this paper builds on (Fang et al., USENIX Security
2020; FLTrust, NDSS 2021) runs custom in-process loops for the same reason
(docs/SOTA.md, "FL engine").

Per-client pipeline order — a documented threat-model choice::

    local train -> LDP clip+noise (if configured) -> Attack.poison (if this
    cid is attacked) -> submit ClientUpdate

* Honest clients sanitise their own update with LDP: clip + Gaussian noise is
  a client-side *privacy* mechanism applied to what they send.
* A Byzantine client runs the stock pipeline up to the poisoning hook, so the
  attack transforms exactly what the client would otherwise have submitted
  (post-LDP when LDP is on) — but the poisoned result is submitted verbatim:
  nothing clips or noises an update after ``Attack.poison``. The attacker
  does not sanitise itself, so LDP must never be read as a defense-side bound
  on malicious updates; bounding those is the validator/aggregator's job.
* Data-level attacks (contract: ``LabelFlip`` is "consumed by the engine
  pre-train") use duck typing: an entry in ``attacks`` that lacks ``poison``
  but exposes ``frac`` is applied as label flipping on every training batch
  (a ``frac`` fraction of each batch's labels moves to a uniformly random
  *different* class), before the forward pass.

Determinism (CPU): ``np.random.SeedSequence(seed)`` roots a spawn tree — one
child per round, one grandchild per client, three streams per grandchild
(torch / attack / LDP). torch, numpy and ``random`` global states are seeded
once per run, and torch is re-seeded per (round, client) from the tree, so
each client step is deterministic no matter how much RNG anything else
consumed. All numeric content of the emitted round dicts is bit-reproducible
under a fixed seed on CPU; the ``ts`` field is wall-clock observability
metadata and is excluded from determinism claims. Residual CUDA
nondeterminism (cuDNN atomics) is documented, not worked around:
bit-reproducibility is claimed on CPU only.

Round events mirror :class:`fedgrid.api.schemas.RoundEvent` field-for-field
(wire key ``"global"``), so ``on_round`` payloads can be streamed over
``/ws/live`` verbatim; ``ConsensusResult.sim_latency_ms`` maps to the event's
``latency_ms``.

Validator handshake (agreed with the consensus/attacks module, ``consensus/pipeline.py``): a
robust-aggregating validator may extend :class:`ValidationReport` with an
``aggregate`` field. The engine probes it via ``getattr``: when not ``None``
it becomes the round's new global parameters and the ``Strategy`` is bypassed
— Krum / geometric median already aggregated, and re-averaging their output
would undo the robustness. When the report carries no aggregate and an empty
accepted set (a fail-closed PBFT round), the global model stays unchanged.
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fedgrid.fl.ldp import LDPConfig, clip_and_noise

if TYPE_CHECKING:
    import torch
    from torch import nn
    from torch.utils.data import Dataset

    from fedgrid.attacks.base import Attack
    from fedgrid.consensus.pbft import ConsensusResult
    from fedgrid.data import Federation
    from fedgrid.fl.strategies import Strategy
    from fedgrid.metrics.runresult import RunResult

__all__ = ["ClientUpdate", "FederatedEngine", "ValidationReport", "ValidatorFn"]


@dataclass
class ClientUpdate:
    """One client's contribution to a round: full parameters, not deltas."""

    cid: int
    n_samples: int
    params: list[np.ndarray]
    train_loss: float
    attack: str | None = None  # ground-truth tag for eval only — the defense never reads it


@dataclass
class ValidationReport:
    """What a validator hands back to the engine for one round.

    The contract references this type from ``ValidatorFn`` without pinning its
    fields; this phase 1 declaration carries exactly what the RoundEvent stream
    needs. Shared helper candidate — flagged for hoisting
    if consensus-side fields grow.
    """

    accepted: list[int]  # client ids admitted to aggregation
    scores: np.ndarray  # per-update fidelity scores, aligned with the updates list
    consensus: ConsensusResult | None  # None when validation runs without PBFT


ValidatorFn = Callable[[list[ClientUpdate], list[np.ndarray]], ValidationReport]


# --------------------------------------------------------------------------- #
# Seed tree — pure functions so tests can re-derive the exact per-step seeds.
# --------------------------------------------------------------------------- #


def _client_seed_seq(run_seed: int, rnd: int, cid: int) -> np.random.SeedSequence:
    """SeedSequence for (run, round, client), spawn key ``(rnd, cid)``.

    Rebuilt from a fresh root on every call: SeedSequence children depend only
    on entropy + spawn key, so identical calls yield identical streams — the
    engine and the tests share this single derivation.
    """
    root = np.random.SeedSequence(run_seed)
    round_ss = root.spawn(rnd + 1)[rnd]
    return round_ss.spawn(cid + 1)[cid]


def _client_streams(
    run_seed: int, rnd: int, cid: int
) -> tuple[np.random.SeedSequence, np.random.SeedSequence, np.random.SeedSequence]:
    """(torch, attack, ldp) child streams for one client step."""
    t_ss, a_ss, l_ss = _client_seed_seq(run_seed, rnd, cid).spawn(3)
    return t_ss, a_ss, l_ss


def _client_torch_seed(run_seed: int, rnd: int, cid: int) -> int:
    """64-bit torch seed for one client step (init, dropout and shuffle order)."""
    t_ss, _, _ = _client_streams(run_seed, rnd, cid)
    return int(t_ss.generate_state(1, dtype=np.uint64)[0])


def _fold_rng(stream: np.random.SeedSequence, extra_seed: int) -> np.random.Generator:
    """Generator from a tree stream with a config-owned seed folded in.

    ``LDPConfig.seed`` / ``LabelFlip.seed`` must influence their noise streams
    without breaking the per-(round, client) derivation, so both become
    entropy alongside the tree stream's state.
    """
    entropy = [int(extra_seed), *(int(v) for v in stream.generate_state(4, dtype=np.uint32))]
    return np.random.default_rng(np.random.SeedSequence(entropy))


# --------------------------------------------------------------------------- #
# Metrics — numpy only (no sklearn dependency, per pyproject).
# --------------------------------------------------------------------------- #


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Confusion matrix, rows = true class, cols = predicted class."""
    t = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_pred, dtype=np.int64)
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: {t.shape} vs {p.shape}")
    for name, arr in (("y_true", t), ("y_pred", p)):
        if arr.size and (arr.min() < 0 or arr.max() >= n_classes):
            raise ValueError(f"{name} outside [0, {n_classes})")
    return np.bincount(t * n_classes + p, minlength=n_classes * n_classes).reshape(
        n_classes, n_classes
    )


def _f1_from_confusion(conf: np.ndarray) -> tuple[np.ndarray, float]:
    """(per-class F1, macro F1) from a confusion matrix.

    Per-class F1 = 2*TP / (2*TP + FP + FN), 0 where the denominator is 0.
    Macro-F1 averages over classes *present in y_true* (rows with support) —
    the realistic class prior makes some eval slices miss rare classes, and
    averaging absent classes as zero would understate every arm equally but
    noisily. The full per-class vector is always reported alongside.
    """
    conf = np.asarray(conf, dtype=np.float64)
    tp = np.diag(conf)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    denom = 2.0 * tp + fp + fn
    f1 = np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1e-300), 0.0)
    support = conf.sum(axis=1) > 0
    macro = float(f1[support].mean()) if support.any() else 0.0
    return f1, macro


def _flip_labels(
    y: torch.Tensor, frac: float, n_classes: int, rng: np.random.Generator
) -> torch.Tensor:
    """Flip ``round(frac * len(y))`` labels to a uniformly random *different* class."""
    import torch

    n = int(y.shape[0])
    k = int(round(min(max(frac, 0.0), 1.0) * n))
    if k == 0:
        return y
    idx = rng.choice(n, size=k, replace=False)
    offsets = rng.integers(1, n_classes, size=k)  # never 0 -> always a different class
    flipped = y.clone()
    y_np = flipped.cpu().numpy()
    y_np[idx] = (y_np[idx] + offsets) % n_classes
    return torch.as_tensor(y_np, dtype=y.dtype, device=y.device)


# --------------------------------------------------------------------------- #
# Class-imbalance handling — weighted cross-entropy from each client's own data.
#
# Under the realistic substation prior (NORMAL >= 90%, CONTRACT.md §Realism) the
# unweighted-CE minimum is "always predict NORMAL", which collapses every fault
# class to zero recall. Re-weighting the per-class loss restores a gradient
# signal for the rare fault classes. The two schemes:
#
#   * ``effective_num`` — class-balanced weighting by the effective number of
#     samples, w_c ∝ (1 - beta) / (1 - beta^{n_c}) with beta = (N-1)/N (Cui,
#     Jia, Lin, Song, Belongie, "Class-Balanced Loss Based on Effective Number
#     of Samples", CVPR 2019, DOI 10.1109/CVPR.2019.00949). beta tied to the
#     client's own N makes the reweighting saturate for common classes instead
#     of exploding like raw inverse frequency — the property the paper motivates.
#   * ``inverse`` — plain inverse-frequency weighting, w_c ∝ 1 / n_c.
#
# FL constraint (documented, binding): counts come from each client's OWN
# training split only — never the global or test distribution — so no federated
# client ever sees another's label histogram through the loss.
# --------------------------------------------------------------------------- #

_OPTIMIZERS: frozenset[str] = frozenset({"sgd", "adam", "adamw"})
_CLASS_WEIGHTINGS: frozenset[str] = frozenset({"none", "inverse", "effective_num"})


def _dataset_labels(dataset: Any) -> np.ndarray | None:
    """Labels of a dataset as an int array, without materialising the inputs.

    Walks the torch dataset wrappers by duck-typing (no torch import): ``Subset``
    (``dataset``/``indices``), ``ConcatDataset`` (``datasets``), ``TensorDataset``
    (``tensors[1]``) and the contract ``ClientDataset`` (``labels``). Returns
    ``None`` for a shape it cannot read cheaply, so the caller can fall back to a
    full scan.
    """
    indices = getattr(dataset, "indices", None)
    base = getattr(dataset, "dataset", None)
    if indices is not None and base is not None:  # torch Subset
        inner = _dataset_labels(base)
        return None if inner is None else np.asarray(inner)[np.asarray(indices, dtype=np.int64)]
    parts = getattr(dataset, "datasets", None)
    if parts is not None:  # torch ConcatDataset
        collected = [_dataset_labels(d) for d in parts]
        if any(c is None for c in collected):
            return None
        return (
            np.concatenate([np.asarray(c) for c in collected])
            if collected
            else np.zeros(0, dtype=np.int64)
        )
    tensors = getattr(dataset, "tensors", None)
    if tensors is not None and len(tensors) >= 2:  # torch TensorDataset
        return np.asarray(tensors[1])
    labels = getattr(dataset, "labels", None)  # contract ClientDataset
    if labels is not None:
        return np.asarray(labels)
    return None


def _class_weight_vector(counts: np.ndarray, scheme: str) -> np.ndarray:
    """Per-class loss weights from label ``counts`` under ``scheme``.

    Absent classes (count 0) get weight 0; present-class weights are normalised
    to average 1 so the weighted loss keeps roughly the unweighted magnitude
    (the learning rate transfers across schemes). ``effective_num`` and
    ``inverse`` follow the module docstring.
    """
    counts = np.asarray(counts, dtype=np.float64)
    weights = np.zeros(counts.shape[0], dtype=np.float64)
    present = counts > 0
    if not present.any():
        return weights
    if scheme == "inverse":
        weights[present] = 1.0 / counts[present]
    elif scheme == "effective_num":
        n_total = float(counts.sum())
        beta = (n_total - 1.0) / n_total  # in [0, 1): N >= 1 since a class is present
        effective = (1.0 - np.power(beta, counts[present])) / (1.0 - beta)
        weights[present] = 1.0 / effective
    else:
        raise ValueError(f"unknown class_weighting scheme {scheme!r}")
    weights[present] /= weights[present].mean()
    return weights


def _provenance(rr_mod: Any, name: str) -> str:
    """git_commit/env_fingerprint via ``metrics.runresult`` helpers, else "unknown".

    Helpers are the models/metrics module's ``current_git_commit(repo_root)`` /
    ``current_env_fingerprint()`` (bare names also probed for robustness).
    The git helper gets the repo root derived from the installed package so
    provenance does not depend on the caller's cwd; every failure path
    degrades to "unknown" — provenance must never be fabricated or fatal.
    """
    helper = getattr(rr_mod, f"current_{name}", None) or getattr(rr_mod, name, None)
    if not callable(helper):
        return "unknown"
    try:
        if name == "git_commit":
            import fedgrid

            root = Path(fedgrid.__file__).resolve().parents[2]  # src layout: src/fedgrid/
            try:
                return str(helper(root))
            except TypeError:  # helper variant without a repo_root parameter
                return str(helper())
        return str(helper())
    except NotImplementedError:
        return "unknown"


def _get_params(model: nn.Module) -> list[np.ndarray]:
    """Model parameters as detached numpy copies, in ``model.parameters()`` order.

    Parameters only: the contract models (CNN-LSTM, LIF SNN) hold all learned
    state in parameters — learnable beta/threshold included. Buffers (e.g. SNN
    spike counters) are per-step bookkeeping and are re-initialised by
    ``model_fn`` each client step, never federated. Known limit, stated
    openly: a model with learning-relevant buffers (BatchNorm running stats)
    would need a state_dict-level extension of this exchange format.
    """
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]


def _set_params(model: nn.Module, params: list[np.ndarray]) -> None:
    """Copy numpy arrays into ``model.parameters()`` (order-aligned, strict)."""
    import torch

    with torch.no_grad():
        for p, arr in zip(model.parameters(), params, strict=True):
            p.copy_(torch.as_tensor(arr, dtype=p.dtype, device=p.device))


class FederatedEngine:
    """Runs the whole federation in-process: K local trainers around one global model.

    Public state: ``global_params`` (list[np.ndarray], current global model
    parameters) is (re)built at the start of :meth:`run` and updated after
    each round's aggregation — ``on_round`` fires *after* that update and the
    round's global-model evaluation, so a callback observes the post-round
    state. :meth:`run` re-derives all state from ``seed``, so calling it again
    reproduces the same run.
    """

    def __init__(
        self,
        federation: Federation,
        model_fn: Callable[[], nn.Module],
        strategy: Strategy,
        rounds: int,
        local_epochs: int = 1,
        batch_size: int = 64,
        lr: float = 1e-3,
        optimizer: str = "sgd",
        class_weighting: str = "none",
        ldp: LDPConfig | None = None,
        attacks: dict[int, Attack] | None = None,  # cid -> attack
        validator: ValidatorFn | None = None,
        device: str = "cuda",
        seed: int = 0,
        on_round: Callable[[dict], None] | None = None,
    ) -> None:
        if rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {rounds}")
        if local_epochs < 1:
            raise ValueError(f"local_epochs must be >= 1, got {local_epochs}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        # Defaults (``sgd`` + unweighted) keep the centralised-equivalence contract
        # bar exact; the battery selects ``adam`` + ``effective_num`` via config.
        if optimizer not in _OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {sorted(_OPTIMIZERS)}, got {optimizer!r}")
        if class_weighting not in _CLASS_WEIGHTINGS:
            raise ValueError(
                f"class_weighting must be one of {sorted(_CLASS_WEIGHTINGS)}, "
                f"got {class_weighting!r}"
            )
        if not federation.clients:
            raise ValueError("federation has no clients")

        self._federation = federation
        self._model_fn = model_fn
        self._strategy = strategy
        self._rounds = rounds
        self._local_epochs = local_epochs
        self._batch_size = batch_size
        self._lr = lr
        self._optimizer = optimizer
        self._class_weighting = class_weighting
        # Per-client class-weight tensors, computed once from each client's OWN
        # training labels (no global/test leakage) and reused across rounds.
        self._weight_cache: dict[int, Any] = {}
        self._ldp = ldp
        self._attacks = dict(attacks) if attacks else {}
        self._validator = validator
        self._device = device
        self._seed = seed
        self._on_round = on_round

        method = type(strategy).__name__.lower()
        self._method = method
        self._run_id = f"{method}-seed{seed}"  # derived, not random: same-seed runs share it

        self.global_params: list[np.ndarray] = []

    # ------------------------------------------------------------------ runs

    def run(self) -> RunResult:
        """Train for ``rounds`` rounds; emits one ``rounds[]`` dict per round via ``on_round``."""
        import torch
        from torch.utils.data import ConcatDataset

        # Per-run seeding: torch (model init), plus numpy/random global state for
        # any third-party code that still reads it. Per-step determinism does not
        # depend on these — each client step re-seeds from the SeedSequence tree.
        torch.manual_seed(self._seed)
        np.random.seed(self._seed % 2**32)
        random.seed(self._seed)

        self._weight_cache = {}  # rebuilt on device below; a re-run recomputes cleanly
        # Perf (2026-07-29, this file's owner authorised a speed-only edit): materialise
        # each split as device tensors once and reuse across rounds. ClientDataset
        # .__getitem__ re-runs a per-sample numpy normalise + numpy->torch + host->device
        # on every batch of every round; done once per client/eval-set instead it is the
        # same values in the same order (see _client_step / _eval_global), asserted
        # bit-identical by tests/test_fl.py (centralised-equivalence, bit-repro, mu0==fedavg).
        # Caches are keyed within one run() and cleared here so a re-run recomputes cleanly.
        self._train_cache: dict[int, tuple[Any, Any]] = {}
        self._eval_cache: dict[int, tuple[Any, Any]] = {}
        self._dev = torch.device(self._device)
        self._global_model = self._model_fn().to(self._dev)
        self.global_params = _get_params(self._global_model)

        val_union = ConcatDataset([c.val for c in self._federation.clients])
        if len(val_union) == 0:
            raise ValueError("federation has no validation samples")
        self._n_classes = self._infer_n_classes(val_union)

        k = len(self._federation.clients)
        rounds_out: list[dict] = []
        for rnd in range(self._rounds):
            # Deterministic client order: cid 0..K-1, every round, no sampling.
            updates = [self._client_step(rnd, cid) for cid in range(k)]
            report = self._validate(updates)
            accepted = {int(c) for c in report.accepted}
            scores = np.asarray(report.scores, dtype=np.float64)

            robust = getattr(report, "aggregate", None)
            if robust is not None:
                # Robust-aggregating validator handshake (module docstring):
                # the validator already aggregated; adopting anything else
                # would re-average away the robustness.
                self.global_params = [np.asarray(a).copy() for a in robust]
            else:
                admitted = [u for u in updates if u.cid in accepted]
                if admitted:
                    self.global_params = self._strategy.aggregate(admitted, self.global_params)
                # else: an all-rejected (or uncommitted PBFT) round legitimately
                # leaves the global model unchanged — recorded, not raised.

            gmetrics = self._eval_global(val_union)
            event = self._round_event(rnd, updates, report, accepted, scores, gmetrics)
            rounds_out.append(event)
            if self._on_round is not None:
                self._on_round(event)

        final = self._eval_global(self._federation.global_test())
        final["split"] = "global_test"
        return self._build_result(rounds_out, final)

    # ----------------------------------------------------------- client step

    def _client_step(self, rnd: int, cid: int) -> ClientUpdate:
        """Local train -> LDP -> poison, in the documented threat-model order."""
        import torch
        from torch.nn import functional as tf
        from torch.utils.data import DataLoader

        client = self._federation.clients[cid]
        t_ss, a_ss, l_ss = _client_streams(self._seed, rnd, cid)
        torch_seed = int(t_ss.generate_state(1, dtype=np.uint64)[0])
        # Re-seeding torch here makes the step independent of RNG consumption
        # by earlier clients/rounds: init, dropout and shuffle all derive from
        # the (run, round, cid) tree alone.
        torch.manual_seed(torch_seed)

        atk = self._attacks.get(cid)
        atk_tag = type(atk).__name__ if atk is not None else None
        data_level = atk is not None and not hasattr(atk, "poison")
        flip_rng = _fold_rng(a_ss, int(getattr(atk, "seed", 0))) if data_level else None
        flip_frac = float(getattr(atk, "frac", 0.0)) if data_level else 0.0

        model = self._model_fn().to(self._dev)
        _set_params(model, self.global_params)
        model.train()

        hook = getattr(self._strategy, "client_loss_hook", None)
        ref_t = (
            [torch.as_tensor(a, device=self._dev) for a in self.global_params]
            if callable(hook)
            else None
        )

        opt = self._make_optimizer(model.parameters())
        weight = self._client_class_weights(cid, client)
        gen = torch.Generator().manual_seed(torch_seed)
        # Preload once, batch by indexing device tensors. A DataLoader over the *index
        # range* (not the data) reproduces DataLoader(client.train, shuffle=True,
        # generator=gen) bit-for-bit: same generator consumption (the iterator draws a
        # base_seed via .random_(generator=gen) before RandomSampler's randperm, then
        # BatchSampler groups, drop_last=False) for any local_epochs, so the shuffled
        # batch order and contents are identical -- while the per-sample normalise +
        # numpy->torch + host->device in ClientDataset.__getitem__ now happens once per
        # client instead of on every batch of every round.
        x_all, y_all = self._preloaded_train(cid, client.train)
        n = int(x_all.shape[0])
        index_batches = DataLoader(
            range(n), batch_size=self._batch_size, shuffle=True, generator=gen
        )

        loss_sum, n_seen = 0.0, 0
        for _ in range(self._local_epochs):
            for batch_idx in index_batches:
                idx = batch_idx.to(self._dev)
                xb = x_all[idx]
                yb = y_all[idx]
                if flip_rng is not None:
                    yb = _flip_labels(yb, flip_frac, self._n_classes, flip_rng)
                opt.zero_grad()
                base = tf.cross_entropy(model(xb), yb, weight=weight)
                loss = base
                if ref_t is not None:
                    extra = hook(model, ref_t)
                    if extra is not None:
                        loss = base + extra
                loss.backward()
                opt.step()
                # Report the base CE so train_loss is comparable across strategies.
                loss_sum += float(base.detach()) * int(yb.shape[0])
                n_seen += int(yb.shape[0])

        update = ClientUpdate(
            cid=cid,
            n_samples=len(client.train),
            params=_get_params(model),
            train_loss=loss_sum / max(n_seen, 1),
            attack=atk_tag,
        )

        # LDP first: the client's own privacy pipeline sanitises what it would
        # send. Poison last: the attacker has the final word and nothing
        # re-sanitises after it (module docstring, threat-model ordering).
        if self._ldp is not None:
            update.params = clip_and_noise(
                update.params,
                self.global_params,
                self._ldp,
                _fold_rng(l_ss, int(self._ldp.seed)),
            )
        if atk is not None and not data_level:
            from fedgrid.attacks.base import AttackContext

            ctx = AttackContext(
                round=rnd,
                ref=[a.copy() for a in self.global_params],
                rng=_fold_rng(a_ss, int(getattr(atk, "seed", 0))),
            )
            update = atk.poison(update, ctx)
            if update.attack is None:  # keep the ground-truth tag if the attack dropped it
                update.attack = atk_tag
        return update

    def _make_optimizer(self, params: Any) -> Any:
        """Per-step optimizer for ``self._optimizer`` (fresh each client step).

        ``sgd`` (the default, plain SGD) keeps the centralised-equivalence bar
        exact; ``adam``/``adamw`` are the robust choices the battery selects.
        A fresh optimizer per step matches the stateless FedAvg client contract:
        no momentum/moment state carries across rounds through the engine.
        """
        import torch

        if self._optimizer == "sgd":
            return torch.optim.SGD(params, lr=self._lr)
        if self._optimizer == "adam":
            return torch.optim.Adam(params, lr=self._lr)
        return torch.optim.AdamW(params, lr=self._lr)  # validated to {sgd,adam,adamw} in __init__

    def _client_class_weights(self, cid: int, client: Any) -> Any:
        """Cross-entropy class weights for one client, or ``None`` for no weighting.

        Computed once per client from its OWN training labels (FL constraint) and
        cached. Falls back to a full label scan only for dataset shapes
        :func:`_dataset_labels` cannot read cheaply. A client with no usable
        labels weights nothing (``None``) rather than fail.
        """
        if self._class_weighting == "none":
            return None
        if cid in self._weight_cache:
            return self._weight_cache[cid]

        import torch
        from torch.utils.data import DataLoader

        labels = _dataset_labels(client.train)
        if labels is None:
            collected = [yb.cpu().numpy() for _xb, yb in DataLoader(client.train, batch_size=512)]
            labels = np.concatenate(collected) if collected else np.zeros(0, dtype=np.int64)
        counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=self._n_classes)
        weight: Any = None
        if counts.sum() > 0:
            vec = _class_weight_vector(counts, self._class_weighting)
            weight = torch.as_tensor(vec, dtype=torch.float32, device=self._dev)
        self._weight_cache[cid] = weight
        return weight

    # ----------------------------------------------------------- preloading

    def _materialize(self, dataset: Any) -> tuple[Any, Any]:
        """Stack a dataset into ``(X, y)`` device tensors once, via its own ``__getitem__``.

        Bit-identical to per-batch loading: ``DataLoader(shuffle=False)`` collates exactly
        the per-sample tensors ``__getitem__`` yields, in index order; we only move the
        whole thing onto the device a single time instead of re-normalising and
        re-transferring every sample on every round. An empty dataset yields empty tensors.
        """
        import torch
        from torch.utils.data import DataLoader

        xs: list[Any] = []
        ys: list[Any] = []
        for xb, yb in DataLoader(dataset, batch_size=1024, shuffle=False):
            xs.append(xb)
            ys.append(yb)
        if not xs:
            return (
                torch.empty(0, device=self._dev),
                torch.empty(0, dtype=torch.int64, device=self._dev),
            )
        return torch.cat(xs).to(self._dev), torch.cat(ys).to(self._dev)

    def _preloaded_train(self, cid: int, dataset: Any) -> tuple[Any, Any]:
        """Device tensors for client ``cid``'s train split, materialised once and cached."""
        cached = self._train_cache.get(cid)
        if cached is None:
            cached = self._materialize(dataset)
            self._train_cache[cid] = cached
        return cached

    # ------------------------------------------------------------ validation

    def _validate(self, updates: list[ClientUpdate]) -> ValidationReport:
        """Run the plugged validator against the round-start global params."""
        if self._validator is None:
            return ValidationReport(
                accepted=[u.cid for u in updates],
                scores=np.zeros(len(updates), dtype=np.float64),
                consensus=None,
            )
        return self._validator(updates, self.global_params)

    # ------------------------------------------------------------ evaluation

    def _infer_n_classes(self, dataset: Dataset) -> int:
        """Dry forward of one sample fixes the class count (and validates shapes early)."""
        import torch

        x0, _ = dataset[0]
        self._global_model.eval()
        with torch.no_grad():
            logits = self._global_model(x0.unsqueeze(0).to(self._dev))
        self._reset_energy(self._global_model)  # dry forward must not pollute round-1 SynOps
        return int(logits.shape[-1])

    def _eval_global(self, dataset: Dataset) -> dict[str, Any]:
        """Loss/acc/macro-F1/per-class F1/confusion of the global model on ``dataset``."""
        import torch
        from torch.nn import functional as tf

        model = self._global_model
        _set_params(model, self.global_params)
        self._reset_energy(model)
        model.eval()

        # Preloaded, cached by dataset identity: val_union is one object reused every
        # round; the final global_test is used once. Fixed-order slicing reproduces
        # DataLoader(shuffle=False) exactly (indices 0..n-1), so metrics are unchanged.
        cached = self._eval_cache.get(id(dataset))
        if cached is None:
            cached = self._materialize(dataset)
            self._eval_cache[id(dataset)] = cached
        x_all, y_all = cached
        n_total = int(x_all.shape[0])
        eval_bs = max(256, self._batch_size)
        loss_sum, n = 0.0, 0
        preds: list[np.ndarray] = []
        trues: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, n_total, eval_bs):
                xb = x_all[start : start + eval_bs]
                yb = y_all[start : start + eval_bs]
                logits = model(xb)
                loss_sum += float(tf.cross_entropy(logits, yb, reduction="sum"))
                n += int(yb.shape[0])
                preds.append(logits.argmax(dim=-1).cpu().numpy())
                trues.append(yb.cpu().numpy())

        y_pred = np.concatenate(preds)
        y_true = np.concatenate(trues)
        conf = _confusion(y_true, y_pred, self._n_classes)
        per_class, macro = _f1_from_confusion(conf)
        return {
            "loss": loss_sum / n,
            "acc": float((y_true == y_pred).mean()),
            "macro_f1": macro,
            "per_class_f1": [float(v) for v in per_class],
            "confusion": conf.tolist(),
        }

    # --------------------------------------------------------------- events

    def _round_event(
        self,
        rnd: int,
        updates: list[ClientUpdate],
        report: ValidationReport,
        accepted: set[int],
        scores: np.ndarray,
        gmetrics: dict[str, Any],
    ) -> dict[str, Any]:
        """One RoundEvent-shaped dict (JSON-native values only) — streamable verbatim."""
        rejected = sorted(u.cid for u in updates if u.cid not in accepted)
        if report.consensus is not None:
            consensus = {
                "n_messages": int(report.consensus.n_messages),
                "latency_ms": float(report.consensus.sim_latency_ms),
                "rejected": [int(c) for c in report.consensus.rejected],
            }
        else:
            consensus = {"n_messages": 0, "latency_ms": 0.0, "rejected": rejected}
        return {
            "v": 1,
            "run_id": self._run_id,
            "round": rnd + 1,  # 1-based, human-facing; the seed tree is 0-based
            "clients": [
                {
                    "cid": int(u.cid),
                    "n_samples": int(u.n_samples),
                    "loss": float(u.train_loss),
                    "fidelity": float(scores[i]) if i < len(scores) else 0.0,
                    "accepted": bool(u.cid in accepted),
                    "attack": u.attack,
                }
                for i, u in enumerate(updates)
            ],
            "global": {
                "loss": float(gmetrics["loss"]),
                "acc": float(gmetrics["acc"]),
                "macro_f1": float(gmetrics["macro_f1"]),
            },
            "consensus": consensus,
            "energy": self._round_energy(),
            # Wall-clock metadata — the one field excluded from determinism claims.
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    def _round_energy(self) -> dict[str, float]:
        """SynOps stats from the global model when it exposes them, zeros otherwise."""
        syn = getattr(self._global_model, "synops", None)
        if callable(syn):
            try:
                d = syn()
                return {
                    "synops": float(d.get("synops", 0.0)),
                    "ratio": float(d.get("ratio", 0.0)),
                }
            except NotImplementedError:
                pass  # the models/metrics module stub not filled yet — zeros, never fabricated numbers
        return {"synops": 0.0, "ratio": 0.0}

    def _reset_energy(self, model: nn.Module) -> None:
        rst = getattr(model, "reset_counters", None)
        if callable(rst):
            try:
                rst()
            except NotImplementedError:
                pass

    # --------------------------------------------------------------- result

    def _build_result(self, rounds_out: list[dict], final: dict[str, Any]) -> RunResult:
        from fedgrid.metrics import runresult as rr

        config: dict[str, Any] = {
            "k": len(self._federation.clients),
            "rounds": self._rounds,
            "local_epochs": self._local_epochs,
            "batch_size": self._batch_size,
            "lr": self._lr,
            "optimizer": self._optimizer,
            "class_weighting": self._class_weighting,
            "device": self._device,
            "strategy": self._method,
            "mu": getattr(self._strategy, "mu", None),
            "ldp": dataclasses.asdict(self._ldp) if self._ldp is not None else None,
            "attacks": {int(cid): type(a).__name__ for cid, a in self._attacks.items()},
            "validator": self._validator is not None,
            "run_id": self._run_id,
        }
        kwargs: dict[str, Any] = {
            "method": self._method,
            "seed": self._seed,
            "config": config,
            "rounds": rounds_out,
            "final": final,
            "energy": None,  # owned by the models/metrics layer, filled by the runner
            "latency": None,
        }
        # Amendment 2026-07-28d §4 adds git_commit/env_fingerprint to RunResult
        # (the models/metrics module's file, helpers current_git_commit/current_env_fingerprint).
        # Probed defensively so construction works against stub and final forms.
        present = {f.name for f in dataclasses.fields(rr.RunResult)}
        for extra in ("git_commit", "env_fingerprint"):
            if extra in present:
                kwargs[extra] = _provenance(rr, extra)
        return rr.RunResult(**kwargs)
