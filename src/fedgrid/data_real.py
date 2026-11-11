"""Real-measurement loader: MSU/ORNL Power System Attack Datasets -> ``Federation``.

External-validity companion to :mod:`fedgrid.simulation` (owner: Real-Data
Integration agent). Every number in the paper's main results comes from OpenDSS
quasi-static simulation; the most damaging reviewer objection is "all of this is
synthetic". This module ingests a **real hardware-in-the-loop measurement
dataset** and reshapes it into the exact ``(C=11, T=64)`` tensor and
:class:`~fedgrid.data.Federation` interface the pipeline already consumes, so the
CNN-LSTM, spiking and SSM classifiers run **unmodified** on relay/PMU
measurements under the same federated protocol.

**What the dataset is (state this honestly).** The Mississippi State University /
Oak Ridge National Laboratory *Power System Attack Datasets* (Morris, Adhikari,
Pan, Beaver, Borges, 2014): a 2-generator, 2-line, 4-relay transmission testbed
whose electrical physics is modelled on an **RTDS real-time digital simulator**
while the *measurement and communication chain* is real hardware/software
(commercial protection relays, PMUs, an OpenPDC phasor data concentrator, a Snort
IDS and the substation protocol stack). It is **not field data**: what is real is
the instrumentation, not the grid. Using it moves the paper from "one simulator"
to "our simulator **plus** an independent hardware-in-the-loop testbed built by a
different group for a different purpose" -- a genuine reduction in single-tool
risk, not validation against grid recordings. See ``docs/REAL_DATA.md`` (survey)
and ``docs/REAL_DATA_INTEGRATION.md`` (this integration).

**Channel mapping -- all 11 recovered per relay (9 direct, 2 derived).** For each
relay ``R# in {R1..R4}`` (columns are PMU phasor magnitude ``PM*`` and phase-angle
``PA*`` pairs):

======  ===================================  ==================================
Channel Source column(s)                     Note
======  ===================================  ==================================
va,vb,vc ``R#-PM1:V``,``PM2:V``,``PM3:V``    phase A-C voltage magnitude (V)
ia,ib,ic ``R#-PM4:I``,``PM5:I``,``PM6:I``    phase A-C current magnitude (A)
v0       ``R#-PM9:V``                        zero-sequence voltage magnitude
i0       ``R#-PM12:I``                       zero-sequence current magnitude
fdev     ``R#:F`` - 60                       relay frequency deviation (Hz)
p,q      derived                             per-phase P/Q (see :func:`_derive_pq`)
======  ===================================  ==================================

``p = sum_phi V_phi I_phi cos(thetaV - thetaI)``,
``q = sum_phi V_phi I_phi sin(thetaV - thetaI)``, with the phase angles ``PA1:VH``
.. ``PA6:IH`` in degrees. The derivation is physics-validated: on NORMAL rows it
returns P ~= 173 MW, Q ~= 11 MVAr, pf 0.998 at a 131 kV (line-to-neutral, ~230 kV
line) bus, and on the ground-fault rows the zero-sequence current/voltage rise
(i0 mean 11.5 vs 1.4, v0 mean 86 vs 0) -- the classic single-line-to-ground
signature.

**Label mapping -- only the classes that physically occur.** The testbed's 37
scenarios (the ARFF ``marker``) are, per the dataset README: 1 No-event, 8
Natural (6 short-circuit faults + 2 line-maintenance), 28 cyber-attack. Our fixed
8-class taxonomy (:class:`~fedgrid.data.EventClass`) contains no cyber-attack or
maintenance class, so the honest physical subset is **binary**: ``NORMAL``
(marker 41) vs ``FAULT_LG`` (markers 1-6, single-line-to-ground faults). Every
other scenario (maintenance, all attacks) is dropped -- it is neither NORMAL nor a
taxonomy fault. Two of eight classes survive; the models keep ``n_classes=8`` and
macro-F1 averages over the classes present, exactly as in the synthetic tables.
This does **not** validate the LL/LLG/3PH/OVERLOAD/SAG/SWELL taxonomy (no open
real source provides those labelled at multiple sites) and the prior is
fault-heavy, not the synthetic NORMAL>=90% -- a distinct operating point.

**Windowing.** Rows within a file are contiguous per scenario (37 blocks/file,
553 blocks total, verified). Each contiguous ``(file, marker)`` block is one
quasi-continuous episode; ``T=64`` windows are cut with a documented ``stride``
and **never cross an episode boundary**. The public release is a 1% random
sub-sample with no timestamps, so a window is a ragged ~tens-of-seconds snapshot,
not a uniform grid -- a documented caveat, not hidden.

**Federation / clients.** The natural unit is the ``(relay, file)`` shard: 4
relays x 15 experiment files = 60 shards. ``partition="relay"`` yields the honest
K=4 vantage-point split (one client per measurement location). ``partition=
"shard"`` groups the 60 shards into K contiguous clients (K=10 -> 6 shards each)
so the federated protocol -- committee size, Byzantine budget, aggregation -- is
**identical to the synthetic experiments**; that comparability is the point. The
heterogeneity is the relay vantage point (feature skew); we do **not** impose the
synthetic Dirichlet label skew (only two classes, and every file contains every
scenario, so the split is close to IID in labels -- stated plainly, not dressed up
as naturally-occurring non-IID).

**Normalisation discipline (no leakage).** Absolute per-unit bases are not shipped
with the release, so rather than divide by a fitted base (which would leak
statistics across splits) we feed physical SI units and let the contract's
per-channel :class:`~fedgrid.data.Normalizer` -- fit on each client's *train*
split only -- standardise every channel. This is the same leak-free discipline the
synthetic path uses.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fedgrid.data import (
    N_CHANNELS,
    N_CLASSES,
    WINDOW_T,
    ClientDataset,
    ClientMeta,
    EventClass,
    Federation,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "FAULT_LG_MARKERS",
    "NORMAL_MARKERS",
    "RELAYS",
    "load_msu_federation",
    "msu_shard_windows",
]

# Relay identifiers (measurement vantage points): R1/R2 on line 1, R3/R4 on line 2.
RELAYS: tuple[str, ...] = ("R1", "R2", "R3", "R4")

# marker -> our EventClass. The physical subset only: single-line-to-ground faults
# (scenarios 1-6, validated by the zero-sequence signature) and normal operation
# (scenario 41). Maintenance (13,14) and every cyber-attack scenario are dropped --
# they are outside the fixed 8-class taxonomy and must not be mislabelled.
NORMAL_MARKERS: frozenset[int] = frozenset({41})
FAULT_LG_MARKERS: frozenset[int] = frozenset({1, 2, 3, 4, 5, 6})

# Per-relay source columns for the derivation. Voltage/current magnitude+angle for
# the three phases, plus zero-sequence magnitudes and frequency.
_V_MAG = ("PM1:V", "PM2:V", "PM3:V")
_V_ANG = ("PA1:VH", "PA2:VH", "PA3:VH")
_I_MAG = ("PM4:I", "PM5:I", "PM6:I")
_I_ANG = ("PA4:IH", "PA5:IH", "PA6:IH")
_V0 = "PM9:V"
_I0 = "PM12:I"
_FREQ = "F"
_NOMINAL_FREQ_HZ = 60.0

# Cache/format tag: bump if the parse or windowing semantics change so a stale
# shard cache is never silently reused.
_CACHE_VERSION = 1


def _marker_label(marker: int) -> int | None:
    """Map an ARFF ``marker`` scenario id to an EventClass code, or ``None`` to drop."""
    if marker in NORMAL_MARKERS:
        return int(EventClass.NORMAL)
    if marker in FAULT_LG_MARKERS:
        return int(EventClass.FAULT_LG)
    return None


def _read_arff(path: Path) -> tuple[list[str], np.ndarray]:
    """Parse a Weka ARFF into (attribute names, ``(rows, cols)`` string array).

    Minimal reader: ``@attribute <name> ...`` lines give the column order and the
    ``@data`` section is comma-separated rows. No numeric coercion here -- the
    caller converts only the columns it uses, so a non-numeric cell in an unused
    column (e.g. the ``PA:Z`` impedance's ``Infinity`` when current ~= 0) never
    breaks the parse.
    """
    names: list[str] = []
    data_rows: list[list[str]] = []
    in_data = False
    with path.open("r", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            if in_data:
                data_rows.append(line.split(","))
                continue
            low = line.lower()
            if low.startswith("@attribute"):
                names.append(line.split()[1])
            elif low.startswith("@data"):
                in_data = True
    if not names or not data_rows:
        raise ValueError(f"{path}: no @attribute/@data content parsed")
    width = len(names)
    clean = [r for r in data_rows if len(r) == width]  # guard ragged trailing lines
    return names, np.array(clean, dtype=object)


def _col(rows: np.ndarray, index: dict[str, int], name: str) -> np.ndarray:
    """One column as float64; non-numeric tokens ('Infinity', '?') become NaN."""
    raw = rows[:, index[name]]
    out = np.empty(raw.shape[0], dtype=np.float64)
    for i, tok in enumerate(raw):
        try:
            out[i] = float(tok)
        except (TypeError, ValueError):
            out[i] = np.nan
    return out


def _derive_pq(
    v_mag: np.ndarray, v_ang_deg: np.ndarray, i_mag: np.ndarray, i_ang_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Three-phase active/reactive power from phasor magnitudes and angles.

    ``P = sum_phi V_phi I_phi cos(thetaV_phi - thetaI_phi)`` and the ``sin`` form
    for ``Q`` (angles in degrees). Arrays are ``(N, 3)`` (three phases); returns
    ``(N,)`` each in W / VAr. Physics-validated (see module docstring).
    """
    theta = np.deg2rad(v_ang_deg - i_ang_deg)
    p = (v_mag * i_mag * np.cos(theta)).sum(axis=1)
    q = (v_mag * i_mag * np.sin(theta)).sum(axis=1)
    return p, q


def _relay_matrix(rows: np.ndarray, index: dict[str, int], relay: str) -> np.ndarray:
    """Build the ``(11, N)`` contract-channel matrix for one relay over ``rows``.

    Channel order is the data contract's exactly:
    ``va,vb,vc, ia,ib,ic, p,q, v0,i0, fdev``.
    """
    v_mag = np.stack([_col(rows, index, f"{relay}-{c}") for c in _V_MAG], axis=1)
    v_ang = np.stack([_col(rows, index, f"{relay}-{c}") for c in _V_ANG], axis=1)
    i_mag = np.stack([_col(rows, index, f"{relay}-{c}") for c in _I_MAG], axis=1)
    i_ang = np.stack([_col(rows, index, f"{relay}-{c}") for c in _I_ANG], axis=1)
    p, q = _derive_pq(v_mag, v_ang, i_mag, i_ang)
    v0 = _col(rows, index, f"{relay}-{_V0}")
    i0 = _col(rows, index, f"{relay}-{_I0}")
    fdev = _col(rows, index, f"{relay}:{_FREQ}") - _NOMINAL_FREQ_HZ
    # Column order MUST match fedgrid.data.CHANNELS.
    return np.stack(
        [v_mag[:, 0], v_mag[:, 1], v_mag[:, 2],
         i_mag[:, 0], i_mag[:, 1], i_mag[:, 2],
         p, q, v0, i0, fdev],
        axis=0,
    ).astype(np.float32)


def _episode_blocks(markers: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous ``(start, stop, marker)`` runs of equal marker (one per episode)."""
    if markers.size == 0:
        return []
    change = np.flatnonzero(markers[1:] != markers[:-1]) + 1
    bounds = [0, *change.tolist(), markers.size]
    return [(bounds[i], bounds[i + 1], int(markers[bounds[i]]))
            for i in range(len(bounds) - 1)]


def _window(mat: np.ndarray, stride: int, window_t: int) -> np.ndarray:
    """Cut ``(11, L)`` into ``(n, 11, window_t)`` windows at ``stride`` (no wrap)."""
    length = mat.shape[1]
    if length < window_t:
        return np.empty((0, mat.shape[0], window_t), dtype=np.float32)
    starts = range(0, length - window_t + 1, stride)
    return np.stack([mat[:, s:s + window_t] for s in starts]).astype(np.float32)


def _find_arff_files(data_dir: Path) -> list[Path]:
    """Locate the 15 multiclass ARFF files under ``data_dir`` (several layouts)."""
    for pattern in ("extracted/multiclass/*.arff", "multiclass/*.arff", "*.arff", "**/*.arff"):
        found = sorted(data_dir.glob(pattern))
        if found:
            return found
    raise FileNotFoundError(f"no .arff files found under {data_dir}")


def _file_sort_key(path: Path) -> int:
    """Numeric order of ``dataN ...arff`` so file index is stable (data1..data15)."""
    digits = "".join(ch for ch in path.stem.split()[0] if ch.isdigit())
    return int(digits) if digits else 0


def _shard_id(relay_idx: int, file_idx: int, n_files: int) -> int:
    """Deterministic 0-based shard id ordered by (relay, file)."""
    return relay_idx * n_files + file_idx


def msu_shard_windows(
    data_dir: str | Path,
    *,
    stride: int = 8,
    window_t: int = WINDOW_T,
    cache: bool = True,
) -> dict[str, np.ndarray]:
    """Parse the ARFF files into per-``(relay, file)`` shard windows (cached).

    Returns a dict with concatenated arrays independent of K/seed/partition, so a
    federation at any K reuses one parse:

    * ``windows`` ``(N, 11, T)`` float32, ``labels`` ``(N,)`` int64,
    * ``shard`` ``(N,)`` int64 (relay_idx * n_files + file_idx),
    * ``relay`` ``(N,)`` int64, ``file_idx`` ``(N,)`` int64, ``n_files`` scalar.

    The expensive ARFF parse + windowing is memoised to
    ``<data_dir>/cache/shards_v{V}_s{stride}_t{T}.npz``; delete that file to force
    a rebuild. Set ``cache=False`` to bypass it (tests use a tiny fixture dir).
    """
    data_dir = Path(data_dir)
    cache_path = data_dir / "cache" / f"shards_v{_CACHE_VERSION}_s{stride}_t{window_t}.npz"
    if cache and cache_path.exists():
        with np.load(cache_path) as z:
            return {k: z[k] for k in z.files}

    files = _find_arff_files(data_dir)
    n_files = len(files)
    win_parts: list[np.ndarray] = []
    lab_parts: list[np.ndarray] = []
    shard_parts: list[np.ndarray] = []
    relay_parts: list[np.ndarray] = []
    file_parts: list[np.ndarray] = []

    for file_idx, path in enumerate(sorted(files, key=_file_sort_key)):
        names, rows = _read_arff(path)
        index = {n: i for i, n in enumerate(names)}
        markers = np.array([int(float(m)) for m in rows[:, index["marker"]]], dtype=np.int64)
        blocks = _episode_blocks(markers)
        for relay_idx, relay in enumerate(RELAYS):
            mat = _relay_matrix(rows, index, relay)  # (11, file_len)
            sid = _shard_id(relay_idx, file_idx, n_files)
            for start, stop, marker in blocks:
                label = _marker_label(marker)
                if label is None:
                    continue  # maintenance / attack -> outside taxonomy, dropped
                w = _window(mat[:, start:stop], stride, window_t)
                if w.shape[0] == 0:
                    continue
                win_parts.append(w)
                lab_parts.append(np.full(w.shape[0], label, dtype=np.int64))
                shard_parts.append(np.full(w.shape[0], sid, dtype=np.int64))
                relay_parts.append(np.full(w.shape[0], relay_idx, dtype=np.int64))
                file_parts.append(np.full(w.shape[0], file_idx, dtype=np.int64))

    if not win_parts:
        raise ValueError(f"no windows produced from {data_dir} (stride={stride}, T={window_t})")

    out = {
        "windows": np.concatenate(win_parts).astype(np.float32),
        "labels": np.concatenate(lab_parts),
        "shard": np.concatenate(shard_parts),
        "relay": np.concatenate(relay_parts),
        "file_idx": np.concatenate(file_parts),
        "n_files": np.asarray(n_files, dtype=np.int64),
    }
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **out)
    return out


def _client_shard_groups(
    shard_ids: Sequence[int], relay_ids: Sequence[int], k: int, partition: str
) -> list[list[int]]:
    """Partition the present shard ids into K client groups.

    * ``relay``: one client per relay (K forced to 4) -- the honest vantage-point
      split.
    * ``shard``: the shards, ordered by (relay, file), split into K contiguous
      groups (K=10 -> 6 shards each) for protocol parity with the synthetic K.
    """
    unique = sorted(set(int(s) for s in shard_ids))
    if partition == "relay":
        groups: dict[int, list[int]] = {}
        relay_of = dict(zip((int(s) for s in shard_ids), (int(r) for r in relay_ids),
                            strict=True))
        for s in unique:
            groups.setdefault(relay_of[s], []).append(s)
        return [groups[r] for r in sorted(groups)]
    if partition == "shard":
        return [list(part) for part in np.array_split(np.asarray(unique), k) if len(part)]
    raise ValueError(f"unknown partition {partition!r}; expected 'relay' or 'shard'")


def _feeder_name(relay_ids: Sequence[int]) -> str:
    """Human label for a client's relay composition, e.g. ``msu_R1`` or ``msu_R1R2``."""
    present = sorted(set(int(r) for r in relay_ids))
    return "msu_" + "".join(RELAYS[r] for r in present)


def load_msu_federation(
    data_dir: str | Path,
    *,
    k: int = 10,
    seed: int = 0,
    stride: int = 8,
    window_t: int = WINDOW_T,
    partition: str = "shard",
    alpha: float = 0.5,
    cache: bool = True,
) -> Federation:
    """Build a :class:`~fedgrid.data.Federation` from the MSU/ORNL dataset.

    ``data_dir`` holds the extracted multiclass ARFF files (see
    :func:`msu_shard_windows`). ``partition="shard"`` (default) gives the K-client
    protocol-parity split; ``partition="relay"`` gives the honest K=4
    vantage-point split. ``seed`` drives only the per-client 70/15/15 split
    permutation (the parse and windowing are seed-independent), so the object is
    bit-reproducible. ``alpha`` is recorded on the federation for bookkeeping; no
    Dirichlet label skew is imposed (module docstring).

    The per-client :class:`~fedgrid.data.Normalizer` is fit on each client's train
    split only -- the contract's leakage discipline, unchanged.
    """
    from fedgrid.simulation.grid import FeederSpec  # local: avoids import cycle

    if partition == "relay":
        k = len(RELAYS)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    data = msu_shard_windows(data_dir, stride=stride, window_t=window_t, cache=cache)
    windows, labels = data["windows"], data["labels"]
    shard, relay = data["shard"], data["relay"]
    if windows.shape[1:] != (N_CHANNELS, window_t):
        raise ValueError(f"windows have shape {windows.shape}, expected (*, {N_CHANNELS}, "
                         f"{window_t})")

    groups = _client_shard_groups(shard, relay, k, partition)
    clients: list[ClientDataset] = []
    for cid, shard_group in enumerate(groups):
        mask = np.isin(shard, np.asarray(shard_group, dtype=np.int64))
        cw = windows[mask]
        cl = labels[mask]
        if cw.shape[0] < 3:  # need at least one window per 70/15/15 split slice
            raise ValueError(f"client {cid} has only {cw.shape[0]} windows; "
                             "reduce K, reduce stride, or check the data dir")
        hist = np.bincount(cl, minlength=N_CLASSES).astype(np.float64)
        meta = ClientMeta(
            cid=cid,
            feeder=FeederSpec(name=_feeder_name(relay[mask]), load_shape_seed=seed,
                              pv_penetration=0.0),
            class_weights=(hist / hist.sum()),
            days=0,
            # Deterministic per-client split seed, well separated across clients.
            split_seed=seed * 100_003 + 7919 * (cid + 1),
            n_steps=int(cw.shape[0]),
        )
        clients.append(ClientDataset(cw, cl, meta))
    return Federation(clients, alpha=float(alpha), seed=seed, days=0)


def dataset_report(data_dir: str | Path, *, stride: int = 8) -> dict[str, object]:
    """Summary stats (windows, class counts, per-shard yield) for the docs/tables.

    Pure measurement helper -- no training. Returns a JSON-native dict so the
    integration doc quotes numbers this repo produced, not the paper's.
    """
    data = msu_shard_windows(data_dir, stride=stride, cache=True)
    labels, shard = data["labels"], data["shard"]
    n_files = int(data["n_files"])
    counts = {EventClass(int(c)).name: int((labels == c).sum())
              for c in sorted(np.unique(labels))}
    per_shard = np.bincount(shard.astype(np.int64))
    digest = hashlib.sha256(data["windows"].tobytes()).hexdigest()[:16]
    return {
        "n_windows": int(labels.shape[0]),
        "n_shards": int(np.unique(shard).size),
        "n_files": n_files,
        "stride": stride,
        "class_counts": counts,
        "normal_share": float((labels == int(EventClass.NORMAL)).mean()),
        "windows_per_shard_median": int(np.median(per_shard[per_shard > 0])),
        "windows_sha256_16": digest,
    }


def _cli() -> int:
    """`python -m fedgrid.data_real <data_dir>` prints the dataset report."""
    import argparse

    p = argparse.ArgumentParser(description="Summarise the MSU/ORNL real-data windows.")
    p.add_argument("data_dir")
    p.add_argument("--stride", type=int, default=8)
    args = p.parse_args()
    print(json.dumps(dataset_report(args.data_dir, stride=args.stride), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
