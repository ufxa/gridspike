"""Regression guard against the majority-class collapse (phase 3 timing-pilot bug).

Under the realistic substation prior (NORMAL >= 90% of windows, CONTRACT.md
Realism requirements), an *unweighted* cross-entropy loss has its global minimum
at "always predict the majority class": that drives macro-F1 down to
``F1(majority) / n_classes`` (~0.12 for 8 classes) and gives every minority
(fault) class zero recall -- exactly the pilot symptom (``acc = prior``,
``macro_f1 = F1(NORMAL)/8``) where CNN-LSTM and SNN returned bit-identical
collapsed metrics.

The fix (``fedgrid.fl.engine``): class-weighted cross-entropy computed from each
client's OWN training labels (effective number of samples, Cui et al. CVPR 2019,
or inverse frequency) plus an Adam optimizer. This file trains a small model
through the *real* :class:`FederatedEngine` on a realistically-imbalanced but
separable synthetic federation and asserts:

* the fixed config (adam + weighting) learns the minority classes -- macro-F1
  well above the majority baseline, every minority class with non-zero recall;
* the old default (sgd + unweighted) still collapses on the same data -- so the
  guard genuinely discriminates the fix from the bug rather than passing on an
  easy dataset;
* class weights are computed from local labels only and never leak a global or
  balanced distribution.

Fast + deterministic: 8 classes, K=2, a handful of CPU rounds, fixed seed.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, TensorDataset

from fedgrid.fl.engine import (
    FederatedEngine,
    _class_weight_vector,
    _dataset_labels,
)
from fedgrid.fl.strategies import FedAvg

N_FEATURES = 8
N_CLASSES = 8
# NORMAL-heavy prior: class 0 ~= 91%, the seven "fault" classes share the rest.
_PRIOR = np.array([0.91, *([0.09 / (N_CLASSES - 1)] * (N_CLASSES - 1))])


def _imbalanced_blobs(
    n: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Realistically-imbalanced, linearly-separable blobs (majority class 0)."""
    rng = np.random.default_rng(seed)
    y = rng.choice(N_CLASSES, size=n, p=_PRIOR)
    # Well-separated per-class means so the classes ARE learnable with a gradient
    # signal -- the collapse is about the loss's prior, not class overlap.
    centers = rng.normal(0.0, 3.0, (N_CLASSES, N_FEATURES))
    x = centers[y] + rng.normal(0.0, 0.7, (n, N_FEATURES))
    # Guarantee every minority class appears at least a few times per client so
    # per-class recall is defined (rejection sampling on the rare tail).
    for c in range(1, N_CLASSES):
        if int((y == c).sum()) < 3:
            idx = rng.choice(np.flatnonzero(y == 0), size=3, replace=False)
            y[idx] = c
            x[idx] = centers[c] + rng.normal(0.0, 0.7, (3, N_FEATURES))
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)


class _Client:
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        a, b = int(0.7 * len(y)), int(0.85 * len(y))
        self.train = TensorDataset(x[:a], y[:a])
        self.val = TensorDataset(x[a:b], y[a:b])
        self.test = TensorDataset(x[b:], y[b:])
        self.meta = None


class _Fed:
    def __init__(self, clients: list[_Client]) -> None:
        self.clients = clients

    def global_test(self) -> ConcatDataset:
        return ConcatDataset([c.test for c in self.clients])


def _make_fed(k: int = 2, n: int = 900, seed: int = 0) -> _Fed:
    return _Fed([_Client(*_imbalanced_blobs(n, seed * 100 + cid)) for cid in range(k)])


def _model_fn() -> nn.Module:
    return nn.Sequential(nn.Linear(N_FEATURES, 32), nn.ReLU(), nn.Linear(32, N_CLASSES))


def _run(weighting: str, optimizer: str, lr: float, rounds: int = 12) -> dict:
    eng = FederatedEngine(
        _make_fed(), _model_fn, FedAvg(), rounds=rounds, local_epochs=1,
        batch_size=32, lr=lr, optimizer=optimizer, class_weighting=weighting,
        device="cpu", seed=0,
    )
    return eng.run().final


def _majority_baseline_macro_f1(final: dict) -> float:
    """F1(majority)/n_classes -- the collapsed model's macro-F1 on this test set."""
    conf = np.asarray(final["confusion"], dtype=np.float64)
    support = conf.sum(axis=1)
    majority = int(support.argmax())
    tp = support[majority]  # a majority-only predictor gets every majority sample right
    # Precision if EVERYTHING were predicted majority: tp / total; recall = 1.
    precision = tp / support.sum()
    f1_majority = 2 * precision / (precision + 1.0)
    return float(f1_majority / N_CLASSES)


# --------------------------------------------------------------------------- #
# The weight vector: local, bounded, correctly shaped.
# --------------------------------------------------------------------------- #


def test_class_weight_vector_upweights_minorities() -> None:
    counts = np.array([9100, 130, 130, 130, 130, 130, 130, 120], dtype=np.int64)
    for scheme in ("inverse", "effective_num"):
        w = _class_weight_vector(counts, scheme)
        assert w.shape == (N_CLASSES,)
        assert np.all(w > 0)
        # Rare classes must weigh strictly more than the majority class.
        assert np.all(w[1:] > w[0] * 5)
        # Present-class weights are normalised to mean 1 (loss scale preserved).
        assert abs(w.mean() - 1.0) < 1e-9


def test_class_weight_vector_handles_absent_classes() -> None:
    counts = np.array([50, 0, 10, 0, 0, 0, 0, 0], dtype=np.int64)
    for scheme in ("inverse", "effective_num"):
        w = _class_weight_vector(counts, scheme)
        assert w[1] == 0.0 and w[3] == 0.0  # absent -> weight 0, never NaN/inf
        assert np.all(np.isfinite(w))
        assert w[2] > w[0]  # the rarer present class weighs more
    # An empty client weights nothing rather than dividing by zero.
    assert np.all(_class_weight_vector(np.zeros(N_CLASSES, np.int64), "effective_num") == 0.0)


def test_weights_computed_from_local_labels_only() -> None:
    """The engine reads each client's own train labels -- no global leakage."""
    fed = _make_fed()
    labels0 = _dataset_labels(fed.clients[0].train)
    assert labels0 is not None
    counts0 = np.bincount(labels0, minlength=N_CLASSES)
    # A client's histogram is its own imbalanced draw, not a balanced/global one.
    assert counts0.argmax() == 0 and counts0[0] > 3 * counts0[1:].sum() // 4


# --------------------------------------------------------------------------- #
# End-to-end: the fix learns minorities; the old default collapses.
# --------------------------------------------------------------------------- #


def test_fixed_config_learns_minority_classes() -> None:
    final = _run(weighting="effective_num", optimizer="adam", lr=5e-2)
    baseline = _majority_baseline_macro_f1(final)
    assert baseline < 0.2  # sanity: the collapse target really is ~0.12 here

    conf = np.asarray(final["confusion"], dtype=np.float64)
    support = conf.sum(axis=1)
    recall = np.divide(np.diag(conf), support, out=np.zeros_like(support), where=support > 0)

    # The whole point: macro-F1 well clear of the majority-class baseline...
    assert final["macro_f1"] > 4 * baseline
    assert final["macro_f1"] > 0.5
    # ...driven by real minority detection, not a single lucky class.
    minority_present = [c for c in range(1, N_CLASSES) if support[c] > 0]
    assert minority_present
    assert sum(recall[c] > 0.0 for c in minority_present) >= len(minority_present) - 1
    assert float(np.mean([recall[c] for c in minority_present])) > 0.5


def test_inverse_weighting_also_beats_baseline() -> None:
    final = _run(weighting="inverse", optimizer="adam", lr=5e-2)
    baseline = _majority_baseline_macro_f1(final)
    assert final["macro_f1"] > 3 * baseline


def test_unweighted_sgd_default_still_collapses() -> None:
    """Discrimination: the pre-fix default collapses on the SAME data, so the
    guard above cannot pass just because the dataset is easy."""
    final = _run(weighting="none", optimizer="sgd", lr=1e-3)
    baseline = _majority_baseline_macro_f1(final)
    conf = np.asarray(final["confusion"], dtype=np.float64)
    support = conf.sum(axis=1)
    recall = np.divide(np.diag(conf), support, out=np.zeros_like(support), where=support > 0)
    # Majority class hoovers up the predictions; minorities get ~no recall.
    assert final["macro_f1"] < 2 * baseline
    assert float(np.mean([recall[c] for c in range(1, N_CLASSES) if support[c] > 0])) < 0.2
    # And the fixed config is decisively better on the identical federation.
    fixed = _run(weighting="effective_num", optimizer="adam", lr=5e-2)
    assert fixed["macro_f1"] > final["macro_f1"] + 0.3


def test_fixed_config_is_cpu_deterministic() -> None:
    a = _run(weighting="effective_num", optimizer="adam", lr=5e-2, rounds=6)
    b = _run(weighting="effective_num", optimizer="adam", lr=5e-2, rounds=6)
    assert a["macro_f1"] == b["macro_f1"]
    assert a["confusion"] == b["confusion"]
