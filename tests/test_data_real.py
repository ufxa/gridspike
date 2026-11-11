"""Tests for the MSU/ORNL real-data loader (:mod:`fedgrid.data_real`).

Uses a tiny synthetic fixture that mimics the dataset's ARFF schema (per-relay
``PM*``/``PA*`` phasor columns + ``marker`` scenario id), so the suite needs no
60 MB download and runs in a second on CPU. Covers: the marker->EventClass
mapping, the physics-checked P/Q derivation, window shapes and the ``(11, 64)``
contract, that out-of-taxonomy scenarios (attacks/maintenance) are dropped, the
relay and shard partitions, the train-only normaliser discipline, and that the
loader plugs into ``experiments.build.build_run`` end to end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fedgrid.data import CHANNELS, N_CHANNELS, WINDOW_T, EventClass
from fedgrid.data_real import (
    FAULT_LG_MARKERS,
    NORMAL_MARKERS,
    RELAYS,
    _derive_pq,
    _episode_blocks,
    _marker_label,
    load_msu_federation,
    msu_shard_windows,
)

# Columns the loader reads for every relay, plus a decoy impedance column carrying
# 'Infinity' (as the real release does) to prove unused non-numeric cells are safe.
_PER_RELAY = [
    "PA1:VH", "PM1:V", "PA2:VH", "PM2:V", "PA3:VH", "PM3:V",
    "PA4:IH", "PM4:I", "PA5:IH", "PM5:I", "PA6:IH", "PM6:I",
    "PM9:V", "PM12:I",
]


def _relay_cols() -> list[str]:
    cols: list[str] = []
    for r in RELAYS:
        cols += [f"{r}-{c}" for c in _PER_RELAY]
        cols += [f"{r}-PA:Z", f"{r}:F"]  # decoy (Infinity) + frequency
    return cols


def _block_rows(rng: np.random.Generator, n: int, marker: int, cols: list[str]) -> list[str]:
    """One contiguous scenario block: n rows, faults get a zero-sequence signature."""
    rows: list[str] = []
    fault = marker in FAULT_LG_MARKERS
    for _ in range(n):
        vals: list[str] = []
        for name in cols:
            if name.endswith("-PA:Z"):
                vals.append("Infinity")  # unused, non-numeric -> must not break parsing
            elif name.endswith(":F"):
                vals.append(f"{60.0 + rng.normal(0, 0.005):.5f}")
            elif ":V" in name and "PM9" in name:      # zero-seq voltage
                vals.append(f"{(80.0 if fault else 0.0) + abs(rng.normal(0, 1)):.3f}")
            elif ":I" in name and "PM12" in name:     # zero-seq current
                vals.append(f"{(12.0 if fault else 0.2) + abs(rng.normal(0, 0.5)):.3f}")
            elif name.endswith(":V"):                 # phase voltage magnitude
                vals.append(f"{131000.0 + rng.normal(0, 300):.2f}")
            elif name.endswith(":I"):                 # phase current magnitude
                vals.append(f"{440.0 + rng.normal(0, 5):.3f}")
            else:                                     # phase angles (VH/IH), degrees
                vals.append(f"{rng.normal(0, 3):.3f}")
        vals.append(str(marker))
        rows.append(",".join(vals))
    return rows


def _write_fixture(root: Path, n_files: int = 2, seed: int = 0) -> Path:
    """Write ``n_files`` ARFF files under ``root/extracted/multiclass``.

    Each file holds: a NORMAL block (41), two SLG-fault blocks (1, 2), one attack
    block (7) and one maintenance block (13). The last two must be dropped.
    """
    mc = root / "extracted" / "multiclass"
    mc.mkdir(parents=True, exist_ok=True)
    cols = _relay_cols()
    rng = np.random.default_rng(seed)
    header = ["@relation power"]
    header += [f"@attribute {c} numeric" for c in cols]
    markers_decl = ",".join(str(m) for m in (1, 2, 7, 13, 41))
    header.append(f"@attribute marker {{{markers_decl}}}")
    header.append("@data")
    for fi in range(n_files):
        body: list[str] = []
        body += _block_rows(rng, 200, 41, cols)   # NORMAL  -> ~9 windows @stride16
        body += _block_rows(rng, 150, 1, cols)    # FAULT_LG
        body += _block_rows(rng, 150, 2, cols)    # FAULT_LG
        body += _block_rows(rng, 150, 7, cols)    # attack     -> dropped
        body += _block_rows(rng, 120, 13, cols)   # maintenance -> dropped
        text = "\n".join([*header, *body]) + "\n"
        (mc / f"data{fi + 1} Sampled Scenarios.csv.arff").write_text(text, encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_fixture(tmp_path_factory.mktemp("msu"))


# --------------------------------------------------------------------------- #
# Unit: mapping + derivation.
# --------------------------------------------------------------------------- #


def test_marker_label_mapping() -> None:
    assert _marker_label(41) == int(EventClass.NORMAL)
    for m in (1, 2, 3, 4, 5, 6):
        assert _marker_label(m) == int(EventClass.FAULT_LG)
    for m in (7, 13, 14, 30, 40):  # attacks + maintenance -> dropped
        assert _marker_label(m) is None
    assert NORMAL_MARKERS == {41} and FAULT_LG_MARKERS == {1, 2, 3, 4, 5, 6}


def test_derive_pq_known_values() -> None:
    # Balanced unit phasors, zero angle diff -> P = 3, Q = 0.
    vmag = np.ones((1, 3))
    imag = np.ones((1, 3))
    zero = np.zeros((1, 3))
    p, q = _derive_pq(vmag, zero, imag, zero)
    assert p[0] == pytest.approx(3.0) and q[0] == pytest.approx(0.0)
    # 30 degree lagging current -> P = 3 cos30, Q = 3 sin30.
    p2, q2 = _derive_pq(vmag, zero, imag, np.full((1, 3), -30.0))
    assert p2[0] == pytest.approx(3 * np.cos(np.deg2rad(30)))
    assert q2[0] == pytest.approx(3 * np.sin(np.deg2rad(30)))


def test_episode_blocks() -> None:
    m = np.array([41, 41, 1, 1, 1, 2, 2])
    assert _episode_blocks(m) == [(0, 2, 41), (2, 5, 1), (5, 7, 2)]
    assert _episode_blocks(np.array([], dtype=np.int64)) == []


# --------------------------------------------------------------------------- #
# Windowing + shards.
# --------------------------------------------------------------------------- #


def test_shard_windows_shape_and_labels(fixture_dir: Path) -> None:
    data = msu_shard_windows(fixture_dir, stride=16, cache=False)
    w, y = data["windows"], data["labels"]
    assert w.dtype == np.float32 and w.shape[1:] == (N_CHANNELS, WINDOW_T)
    assert set(np.unique(y).tolist()) == {int(EventClass.NORMAL), int(EventClass.FAULT_LG)}
    # 4 relays x 2 files = 8 shards, all present.
    assert set(np.unique(data["shard"]).tolist()) == set(range(8))
    assert int(data["n_files"]) == 2
    # No window carries a dropped (attack/maintenance) scenario: only 2 classes exist.
    assert w.shape[0] == y.shape[0] > 0


def test_channels_finite_and_ordered(fixture_dir: Path) -> None:
    data = msu_shard_windows(fixture_dir, stride=16, cache=False)
    w = data["windows"]
    assert np.isfinite(w).all()  # derived + direct channels, no Infinity leaks in
    assert len(CHANNELS) == N_CHANNELS  # zero-seq current channel (i0) is index 9
    i0 = w[:, CHANNELS.index("i0"), :]
    fault = data["labels"] == int(EventClass.FAULT_LG)
    # Faults carry the elevated zero-sequence current signature vs normal.
    assert i0[fault].mean() > i0[~fault].mean()


def test_cache_roundtrip(fixture_dir: Path, tmp_path: Path) -> None:
    root = _write_fixture(tmp_path / "cached")
    first = msu_shard_windows(root, stride=16, cache=True)
    assert (root / "cache").exists()
    second = msu_shard_windows(root, stride=16, cache=True)  # served from cache
    assert np.array_equal(first["windows"], second["windows"])
    assert np.array_equal(first["labels"], second["labels"])


# --------------------------------------------------------------------------- #
# Federation.
# --------------------------------------------------------------------------- #


def test_relay_partition_k4(fixture_dir: Path) -> None:
    fed = load_msu_federation(fixture_dir, partition="relay", stride=16, cache=False)
    assert len(fed.clients) == len(RELAYS)  # one client per relay vantage point
    for c in fed.clients:
        assert len(c.train) > 0 and len(c.val) > 0 and len(c.test) > 0
        assert c.windows.shape[1:] == (N_CHANNELS, WINDOW_T)
    # label_histogram only lights up the two present classes.
    hist = fed.label_histogram()
    assert hist.shape == (len(RELAYS), 8)
    assert hist[:, 2:].sum() == 0  # nothing beyond NORMAL(0)/FAULT_LG(1)


def test_shard_partition_k(fixture_dir: Path) -> None:
    fed = load_msu_federation(fixture_dir, partition="shard", k=2, stride=16, cache=False)
    assert len(fed.clients) == 2
    total = sum(len(c) for c in fed.clients)
    assert total == msu_shard_windows(fixture_dir, stride=16, cache=False)["labels"].shape[0]


def test_normalizer_is_train_only(fixture_dir: Path) -> None:
    fed = load_msu_federation(fixture_dir, partition="relay", stride=16, cache=False)
    c = fed.clients[0]
    # The travelling normaliser is fit on the train windows, not all windows.
    tr_idx = np.asarray(c.train.indices, dtype=np.int64)
    expected_mean = c.windows[tr_idx].mean(axis=(0, 2)).astype(np.float32)
    assert np.allclose(c.normalizer.mean.ravel(), expected_mean, atol=1e-4)
    all_mean = c.windows.mean(axis=(0, 2)).astype(np.float32)
    assert not np.allclose(c.normalizer.mean.ravel(), all_mean)  # would signal leakage


def test_reproducible(fixture_dir: Path) -> None:
    a = load_msu_federation(fixture_dir, partition="relay", stride=16, seed=3, cache=False)
    b = load_msu_federation(fixture_dir, partition="relay", stride=16, seed=3, cache=False)
    assert np.array_equal(a.clients[0].windows, b.clients[0].windows)
    assert list(a.clients[0].train.indices) == list(b.clients[0].train.indices)


# --------------------------------------------------------------------------- #
# End-to-end wiring through experiments.build.
# --------------------------------------------------------------------------- #


def test_build_run_uses_msu_source(fixture_dir: Path) -> None:
    from experiments.build import build_run
    from experiments.manifest_real import make_real_config

    cfg = make_real_config(
        model="cnn_lstm", arm="fedavg", seed=0, real_data_dir=str(fixture_dir),
        partition="relay", stride=16, rounds=2,
    )
    cfg["model"]["kwargs"] = {"conv_channels": [8, 16], "lstm_hidden": 16, "lstm_layers": 1}
    built = build_run(cfg, data_dir=None, device="cpu", limit_rounds=2)
    assert len(built.federation.clients) == len(RELAYS)
    result = built.run()
    assert isinstance(result.final["macro_f1"], float)
    assert 0.0 <= result.final["acc"] <= 1.0
    assert len(result.rounds) == 2
