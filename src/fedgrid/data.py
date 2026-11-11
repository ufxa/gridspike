"""Data contract: event taxonomy, per-client datasets and the federation container.

Everything downstream depends on the shapes fixed here: samples are
``x: float32 (C=11, T=64)`` windows with channel order
``va, vb, vc, ia, ib, ic, p, q, v0, i0, fdev`` and labels ``y: int64`` in ``0..7``
(:class:`EventClass`). K = 10 clients, non-IID via feeder identity plus
Dirichlet(alpha) label skew.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset

if TYPE_CHECKING:
    from fedgrid.simulation.grid import FeederSpec

__all__ = [
    "CHANNELS",
    "N_CHANNELS",
    "N_CLASSES",
    "SPLIT_MODES",
    "TEMPORAL_SPLIT_GAP",
    "WINDOW_T",
    "ClientDataset",
    "ClientMeta",
    "EventClass",
    "Federation",
    "Normalizer",
    "apply_split_mode",
    "apply_temporal_split",
    "load_federation",
    "temporal_split_indices",
]

# The fixed data contract (never change without a dated amendment).
CHANNELS: tuple[str, ...] = ("va", "vb", "vc", "ia", "ib", "ic", "p", "q", "v0", "i0", "fdev")
N_CHANNELS: int = 11
WINDOW_T: int = 64
N_CLASSES: int = 8
_SPLIT_FRACS = (0.70, 0.15, 0.15)

#: How a client's windows are cut into train/val/test.
#:
#: ``random`` -- the seeded permutation of :func:`_split_indices`, the split every
#: run before 2026-07-30 used and the one persisted in the ``.npz`` files.
#: ``temporal`` -- contiguous chronological blocks (:func:`temporal_split_indices`).
#: Windows are cut with ``partition.STRIDE = 16`` at ``WINDOW_T = 64``, i.e. 75 %
#: overlap between neighbours, so a permutation split puts a near-copy of almost
#: every held-out window in the training set (measured on the battery data:
#: 99.7 % of test windows share samples with a train window). The temporal mode is
#: the leakage-free protocol -- oldest days train, newest days test, per client.
SPLIT_MODES: tuple[str, ...] = ("random", "temporal")

#: Windows purged at each contiguous-block boundary. Two windows share samples iff
#: their indices differ by less than ``WINDOW_T / STRIDE = 64 / 16 = 4``, so a
#: four-window gap makes the blocks disjoint in *samples*, not just in windows.
TEMPORAL_SPLIT_GAP: int = 4


class EventClass(IntEnum):
    """Label taxonomy for substation events; values are the on-disk ``y`` codes."""

    NORMAL = 0
    FAULT_LG = 1
    FAULT_LL = 2
    FAULT_LLG = 3
    FAULT_3PH = 4
    OVERLOAD = 5
    SAG = 6
    SWELL = 7


@dataclass
class ClientMeta:
    """Static identity of one federated client (substation).

    The phase 1 declared fields (cid, feeder, class_weights) are contract-level;
    the extra fields are the data-simulation module phase 2 extensions (extend, never shrink).
    """

    cid: int
    feeder: FeederSpec
    class_weights: np.ndarray  # (8,) Dirichlet draw driving this client's label skew
    days: int = 0              # simulated days behind this client's windows
    split_seed: int = 0        # seed of the 70/15/15 permutation (reproducibility)
    n_nonconverged: int = 0    # solver steps held (non-convergent) across this client
    n_steps: int = 0           # total quasi-static steps simulated for this client
    split_mode: str = "random"  # which of SPLIT_MODES produced train/val/test


@dataclass
class Normalizer:
    """Per-channel affine normalisation ``(x - mean) / std``.

    Statistics are computed on the *train* split only (:meth:`fit`) and travel
    inside the owning :class:`ClientDataset` - never fitted on val/test.
    """

    mean: np.ndarray  # (C, 1) float32
    std: np.ndarray   # (C, 1) float32

    @classmethod
    def fit(cls, windows: np.ndarray) -> Normalizer:
        """Fit on an (N, C, T) array - the caller passes the train windows only."""
        if windows.ndim != 3 or windows.shape[1] != N_CHANNELS:
            raise ValueError(f"expected (N, {N_CHANNELS}, T), got {windows.shape}")
        mean = windows.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)[:, None]
        std = windows.std(axis=(0, 2), dtype=np.float64).astype(np.float32)[:, None]
        std = np.maximum(std, np.float32(1e-6))
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)


def _split_indices(
    n: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seeded 70/15/15 permutation split; returns sorted index arrays."""
    perm = np.random.default_rng(seed).permutation(n)
    n_tr = int(n * _SPLIT_FRACS[0])
    n_va = int(n * _SPLIT_FRACS[1])
    return (
        np.sort(perm[:n_tr]),
        np.sort(perm[n_tr:n_tr + n_va]),
        np.sort(perm[n_tr + n_va:]),
    )


def temporal_split_indices(
    n: int, gap: int = TEMPORAL_SPLIT_GAP
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contiguous chronological 70/15/15 split; returns sorted index arrays.

    A client's windows are stored in simulated-time order (``make_federation``
    concatenates day 0, day 1, ... and each day is cut left to right), so index
    order *is* time order: the oldest 70 % train, the next 15 % validate and the
    newest 15 % test. No RNG is involved -- the split is a pure function of ``n``.

    Because neighbouring windows overlap by 75 % (stride 16, T = 64), a block
    boundary alone is not enough: the last train window still shares samples with
    the first val window. ``gap`` windows are therefore dropped from the *head* of
    the val and test blocks, so every held-out window is sample-disjoint from
    every train window (``|i - j| >= WINDOW_T / STRIDE``). The train block keeps
    its full 70 %; val/test lose ``gap`` windows each, which at the battery's
    2610 windows per client is 0.15 % of the data.
    """
    n = int(n)
    gap = int(gap)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if gap < 0:
        raise ValueError(f"gap must be >= 0, got {gap}")
    n_tr = int(n * _SPLIT_FRACS[0])
    n_va = int(n * _SPLIT_FRACS[1])
    train = np.arange(0, n_tr, dtype=np.int64)
    val = np.arange(n_tr + gap, n_tr + n_va, dtype=np.int64)
    test = np.arange(n_tr + n_va + gap, n, dtype=np.int64)
    if min(train.size, val.size, test.size) == 0:
        raise ValueError(
            f"temporal split of n={n} windows at gap={gap} leaves an empty block "
            f"({train.size}/{val.size}/{test.size}); need a longer client series"
        )
    return train, val, test


def apply_temporal_split(federation: Federation, gap: int = TEMPORAL_SPLIT_GAP) -> Federation:
    """Re-split every client of ``federation`` into contiguous time blocks, in place.

    Applied *at load* (:func:`load_federation`, :func:`apply_split_mode`), so no
    dataset is regenerated: the persisted windows are already in time order and
    only the index sets change. The train-only normalisation statistics are refitted
    on the new train block -- carrying the ``.npz`` statistics over would leak the
    (now held-out) newest days into the scaling. Returns the same object.
    """
    for client in federation.clients:
        tr, va, te = temporal_split_indices(len(client.labels), gap=gap)
        client._assign_splits(tr, va, te)
        client.meta.split_mode = "temporal"
    return federation


def apply_split_mode(
    federation: Federation, split_mode: str = "random", gap: int = TEMPORAL_SPLIT_GAP
) -> Federation:
    """Dispatch over :data:`SPLIT_MODES`; ``"random"`` returns ``federation`` untouched."""
    if split_mode == "random":
        return federation
    if split_mode == "temporal":
        return apply_temporal_split(federation, gap=gap)
    raise ValueError(f"unknown split_mode {split_mode!r}; expected one of {SPLIT_MODES}")


class ClientDataset(torch.utils.data.Dataset):
    """One substation's windowed time series; yields ``(x (C,T) float32, y int64)``.

    Per-channel affine normalisation statistics are computed on this client's
    *train* split only and travel inside the dataset object - never fitted on
    val/test. Splits are 70/15/15 train/val/test, seeded.
    """

    train: Subset
    val: Subset
    test: Subset
    meta: ClientMeta

    def __init__(self, windows: np.ndarray, labels: np.ndarray, meta: ClientMeta) -> None:
        windows = np.ascontiguousarray(windows, dtype=np.float32)
        labels = np.ascontiguousarray(labels, dtype=np.int64)
        if windows.ndim != 3 or windows.shape[1:] != (N_CHANNELS, WINDOW_T):
            raise ValueError(f"windows must be (N, {N_CHANNELS}, {WINDOW_T}), "
                             f"got {windows.shape}")
        if labels.shape != (windows.shape[0],):
            raise ValueError("labels must be (N,) aligned with windows")
        self.windows = windows
        self.labels = labels
        self.meta = meta
        tr, va, te = _split_indices(len(labels), meta.split_seed)
        self._assign_splits(tr, va, te)

    @classmethod
    def from_splits(
        cls,
        windows: np.ndarray,
        labels: np.ndarray,
        meta: ClientMeta,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        normalizer: Normalizer,
    ) -> ClientDataset:
        """Rebuild from persisted splits/statistics (``load_federation`` path)."""
        obj = cls.__new__(cls)
        obj.windows = np.ascontiguousarray(windows, dtype=np.float32)
        obj.labels = np.ascontiguousarray(labels, dtype=np.int64)
        obj.meta = meta
        obj.normalizer = normalizer
        obj.train = Subset(obj, [int(i) for i in train_idx])
        obj.val = Subset(obj, [int(i) for i in val_idx])
        obj.test = Subset(obj, [int(i) for i in test_idx])
        return obj

    def _assign_splits(self, tr: np.ndarray, va: np.ndarray, te: np.ndarray) -> None:
        # Normalizer sees the raw *train* windows only - the contract's leakage bar.
        self.normalizer = Normalizer.fit(self.windows[tr])
        self.train = Subset(self, [int(i) for i in tr])
        self.val = Subset(self, [int(i) for i in va])
        self.test = Subset(self, [int(i) for i in te])

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.normalizer.transform(self.windows[idx])
        return (
            torch.from_numpy(x),
            torch.tensor(int(self.labels[idx]), dtype=torch.int64),
        )


class Federation:
    """The K client datasets plus the derived global views used by evaluation."""

    clients: list[ClientDataset]

    def __init__(self, clients: list[ClientDataset], alpha: float = 0.5,
                 seed: int = 0, days: int = 0) -> None:
        self.clients = clients
        self.alpha = alpha
        self.seed = seed
        self.days = days

    def global_test(self) -> ConcatDataset:
        """Union of all client test splits - the paper's global test set."""
        return ConcatDataset([c.test for c in self.clients])

    def label_histogram(self) -> np.ndarray:
        """Per-client class counts, shape (K, 8) - feeds the non-IID figure."""
        hist = np.zeros((len(self.clients), N_CLASSES), dtype=np.int64)
        for k, client in enumerate(self.clients):
            hist[k] = np.bincount(client.labels, minlength=N_CLASSES)
        return hist


def load_federation(out_dir: Path | str, split_mode: str = "random") -> Federation:
    """Rebuild a :class:`Federation` from ``fedgrid.simulation.generate`` output.

    Uses the persisted split indices and normaliser statistics, so the loaded
    object is bit-identical to the one built in-process at generation time.

    ``split_mode="temporal"`` re-derives the splits as contiguous time blocks
    (:func:`apply_temporal_split`) *after* loading -- the on-disk data is read, not
    rewritten, so one generated ``data_battery`` serves both protocols.
    """
    from fedgrid.simulation.grid import FeederSpec  # local import: avoids cycle

    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    clients: list[ClientDataset] = []
    for entry in manifest["clients"]:
        with np.load(out_dir / entry["file"]) as z:
            meta = ClientMeta(
                cid=int(entry["cid"]),
                feeder=FeederSpec(**entry["feeder"]),
                class_weights=z["class_weights"],
                days=int(manifest["config"]["days"]),
                split_seed=int(entry["split_seed"]),
            )
            norm = Normalizer(mean=z["norm_mean"], std=z["norm_std"])
            clients.append(ClientDataset.from_splits(
                z["windows"], z["labels"], meta,
                z["train_idx"], z["val_idx"], z["test_idx"], norm,
            ))
    cfg = manifest["config"]
    fed = Federation(clients, alpha=float(cfg["alpha"]), seed=int(cfg["seed"]),
                     days=int(cfg["days"]))
    return apply_split_mode(fed, split_mode)
