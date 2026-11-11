"""Contiguous temporal split: the leakage-free protocol and its wiring.

Windows are cut with ``partition.STRIDE = 16`` at ``WINDOW_T = 64``, i.e. 75 %
overlap between neighbours, so the seeded permutation split puts a near-copy of
almost every held-out window in the training set (99.7 % of test windows on the
shipped battery data). ``federation.split_mode = "temporal"`` recomputes the split
*at load* as contiguous chronological blocks with a four-window purge at each
boundary -- no dataset is regenerated.

Two contracts are tested here:

* the temporal split itself -- contiguous, ordered, sample-disjoint across blocks,
  right fractions, deterministic;
* that the default (``"random"``) path is **bit-identical** to before: a loaded
  federation reproduces exactly the indices and normaliser statistics persisted in
  the ``.npz``, so every historical run stays valid.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fedgrid.data import (
    N_CHANNELS,
    TEMPORAL_SPLIT_GAP,
    WINDOW_T,
    Normalizer,
    _split_indices,
    apply_split_mode,
    load_federation,
    temporal_split_indices,
)

#: Windows per client in the paper battery (30 days x 87 windows/day).
BATTERY_N = 2610


def _min_gap(a: np.ndarray, b: np.ndarray) -> int:
    """Smallest ``|i - j|`` over the two index sets (the overlap criterion)."""
    return int(np.abs(np.asarray(a)[:, None] - np.asarray(b)[None, :]).min())


# --------------------------------------------------------------------------- #
# The split function.
# --------------------------------------------------------------------------- #


def test_temporal_blocks_are_contiguous_and_chronological() -> None:
    tr, va, te = temporal_split_indices(BATTERY_N)
    for block in (tr, va, te):
        assert block.size > 0
        assert np.array_equal(block, np.arange(block[0], block[-1] + 1))  # contiguous
    assert tr[0] == 0 and te[-1] == BATTERY_N - 1  # oldest windows train, newest test
    assert tr[-1] < va[0] < va[-1] < te[0]  # strictly increasing in time


def test_temporal_split_has_no_window_overlap_across_blocks() -> None:
    """No held-out window shares a sample with a train window (|i-j| >= T/stride)."""
    tr, va, te = temporal_split_indices(BATTERY_N)
    assert WINDOW_T // 16 == TEMPORAL_SPLIT_GAP  # the gap IS one full window
    assert _min_gap(tr, te) >= TEMPORAL_SPLIT_GAP
    assert _min_gap(tr, va) >= TEMPORAL_SPLIT_GAP
    assert _min_gap(va, te) >= TEMPORAL_SPLIT_GAP


def test_random_split_leaks_almost_every_test_window() -> None:
    """The baseline this replaces: the permutation split leaks >95 % of the test set."""
    tr, _va, te = _split_indices(BATTERY_N, seed=7)
    leaked = sum(1 for t in te if np.abs(tr - t).min() < TEMPORAL_SPLIT_GAP)
    assert leaked / te.size > 0.95


def test_temporal_split_fractions_are_the_battery_fractions() -> None:
    tr, va, te = temporal_split_indices(BATTERY_N)
    assert tr.size / BATTERY_N == pytest.approx(0.70, abs=0.005)
    assert va.size / BATTERY_N == pytest.approx(0.15, abs=0.005)
    assert te.size / BATTERY_N == pytest.approx(0.15, abs=0.005)
    # Exactly the purged windows are missing; nothing else is dropped or duplicated.
    assert tr.size + va.size + te.size == BATTERY_N - 2 * TEMPORAL_SPLIT_GAP
    assert len(set(tr) | set(va) | set(te)) == BATTERY_N - 2 * TEMPORAL_SPLIT_GAP


def test_temporal_split_is_deterministic() -> None:
    a = temporal_split_indices(BATTERY_N)
    b = temporal_split_indices(BATTERY_N)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)


def test_temporal_split_rejects_a_series_too_short_for_the_gap() -> None:
    with pytest.raises(ValueError, match="empty block"):
        temporal_split_indices(20)
    with pytest.raises(ValueError):
        temporal_split_indices(0)


@settings(max_examples=50, deadline=None)
@given(n=st.integers(40, 5000))
def test_property_temporal_split_disjoint_ordered_for_every_n(n: int) -> None:
    tr, va, te = temporal_split_indices(n)
    assert tr.size and va.size and te.size
    assert tr[-1] < va[0] and va[-1] < te[0]
    assert te[-1] == n - 1
    assert _min_gap(tr, va) >= TEMPORAL_SPLIT_GAP
    assert _min_gap(va, te) >= TEMPORAL_SPLIT_GAP
    assert tr.size == int(n * 0.70)


# --------------------------------------------------------------------------- #
# Applying it to a federation (load path, no regeneration).
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def npz_federation(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A generator-shaped federation dir (2 clients) written without OpenDSS.

    Mirrors ``fedgrid.simulation.generate`` exactly in the fields
    ``load_federation`` reads, so the load path is exercised for real while the
    test stays fast and hermetic.
    """
    out = tmp_path_factory.mktemp("npzfed")
    rng = np.random.default_rng(0)
    n = 400
    clients = []
    for cid in range(2):
        windows = rng.normal(1.0, 0.2, size=(n, N_CHANNELS, WINDOW_T)).astype(np.float32)
        labels = rng.integers(0, 8, size=n).astype(np.int64)
        split_seed = 100_003 * cid + 7919
        tr, va, te = _split_indices(n, split_seed)
        norm = Normalizer.fit(windows[tr])
        fname = f"client_{cid:02d}.npz"
        np.savez_compressed(
            out / fname, windows=windows, labels=labels,
            class_weights=rng.dirichlet(0.5 * np.ones(8)),
            train_idx=tr, val_idx=va, test_idx=te,
            norm_mean=norm.mean, norm_std=norm.std,
        )
        clients.append({
            "cid": cid, "file": fname,
            "feeder": {"name": "ieee13", "load_shape_seed": cid, "pv_penetration": 0.15},
            "split_seed": split_seed, "n_windows": n,
        })
    (out / "manifest.json").write_text(json.dumps(
        {"config": {"k": 2, "days": 1, "alpha": 0.5, "seed": 0}, "clients": clients},
    ), encoding="utf-8")
    return out


def test_default_split_mode_reproduces_the_persisted_indices(npz_federation: Path) -> None:
    """The default is bit-identical to before: persisted indices AND statistics."""
    fed = load_federation(npz_federation)
    for client in fed.clients:
        with np.load(npz_federation / f"client_{client.meta.cid:02d}.npz") as z:
            assert client.train.indices == [int(i) for i in z["train_idx"]]
            assert client.val.indices == [int(i) for i in z["val_idx"]]
            assert client.test.indices == [int(i) for i in z["test_idx"]]
            assert np.array_equal(client.normalizer.mean, z["norm_mean"])
            assert np.array_equal(client.normalizer.std, z["norm_std"])
        assert client.meta.split_mode == "random"
    # Explicitly asking for the default changes nothing.
    again = load_federation(npz_federation, split_mode="random")
    for a, b in zip(fed.clients, again.clients, strict=True):
        assert a.train.indices == b.train.indices and a.test.indices == b.test.indices


def test_temporal_split_mode_rebuilds_every_client_contiguously(npz_federation: Path) -> None:
    fed = load_federation(npz_federation, split_mode="temporal")
    for client in fed.clients:
        n = len(client.labels)
        tr, va, te = temporal_split_indices(n)
        assert client.train.indices == [int(i) for i in tr]
        assert client.val.indices == [int(i) for i in va]
        assert client.test.indices == [int(i) for i in te]
        assert client.meta.split_mode == "temporal"
        assert _min_gap(client.train.indices, client.test.indices) >= TEMPORAL_SPLIT_GAP
        # The normaliser follows the new train block -- the persisted statistics were
        # fitted on windows that are now held out.
        expected = Normalizer.fit(client.windows[tr])
        assert np.array_equal(client.normalizer.mean, expected.mean)
        assert np.array_equal(client.normalizer.std, expected.std)
        with np.load(npz_federation / f"client_{client.meta.cid:02d}.npz") as z:
            assert not np.array_equal(client.normalizer.mean, z["norm_mean"])
            # The on-disk data itself is untouched (loaded, never rewritten).
            assert np.array_equal(z["train_idx"], _split_indices(n, client.meta.split_seed)[0])


def test_unknown_split_mode_is_a_loud_error(npz_federation: Path) -> None:
    with pytest.raises(ValueError, match="unknown split_mode"):
        load_federation(npz_federation, split_mode="chronological")
    with pytest.raises(ValueError, match="unknown split_mode"):
        apply_split_mode(load_federation(npz_federation), "shuffle")


# --------------------------------------------------------------------------- #
# Config wiring: experiments.build honours federation.split_mode.
# --------------------------------------------------------------------------- #


def _config(npz_federation: Path, **fed_kwargs: object) -> dict:
    from experiments.manifest import make_config

    cfg = make_config(model="cnn_lstm", arm="fedavg", seed=0, k=2, days=1)
    cfg["federation"].update(fed_kwargs)
    return cfg


def test_build_defaults_to_the_persisted_random_split(npz_federation: Path) -> None:
    from experiments.build import load_or_make_federation

    cfg = _config(npz_federation)
    assert "split_mode" not in cfg["federation"]  # the manifest default is unchanged
    fed = load_or_make_federation(cfg, npz_federation, log=lambda _m: None)
    for client in fed.clients:
        with np.load(npz_federation / f"client_{client.meta.cid:02d}.npz") as z:
            assert client.train.indices == [int(i) for i in z["train_idx"]]
        assert client.meta.split_mode == "random"


def test_build_honours_temporal_split_mode(npz_federation: Path) -> None:
    from experiments.build import load_or_make_federation

    cfg = _config(npz_federation, split_mode="temporal")
    logs: list[str] = []
    fed = load_or_make_federation(cfg, npz_federation, log=logs.append)
    assert any("split_mode=temporal" in m for m in logs)  # never silent
    for client in fed.clients:
        tr, _va, te = temporal_split_indices(len(client.labels))
        assert client.train.indices == [int(i) for i in tr]
        assert client.test.indices == [int(i) for i in te]
        assert _min_gap(client.train.indices, client.test.indices) >= TEMPORAL_SPLIT_GAP


# --------------------------------------------------------------------------- #
# The temporal battery manifest.
# --------------------------------------------------------------------------- #


def test_temporal_rows_rewrite_block_id_and_split_mode(tmp_path: Path) -> None:
    from experiments.gen_temporal_battery import RUN_ID_PREFIX, temporal_rows
    from experiments.manifest import battery_runs, config_hash

    runs = battery_runs()
    src = tmp_path / "src.jsonl"
    keep = [r for r in runs if r.block in ("t1a", "t1b")][:4]
    drop = [r for r in runs if r.block == "t2a"][:2]
    src.write_text("".join(r.to_json_line() + "\n" for r in keep + drop), encoding="utf-8")

    rows = temporal_rows((src,))
    assert len(rows) == len(keep)  # Tier-2 blocks are not replicated
    for row, orig in zip(rows, keep, strict=True):
        assert row["block"] == f"{orig.block}_temporal"
        assert row["run_id"] == f"{RUN_ID_PREFIX}{orig.run_id}"
        assert row["config"]["federation"]["split_mode"] == "temporal"
        assert row["config_hash"] == config_hash(row["config"]) != orig.config_hash
        # Top-level metadata copied verbatim, so a temporal row pairs with its twin.
        for key in ("arm", "model", "seed", "tier"):
            assert row[key] == getattr(orig, key)
        # ... and nothing else in the config moved.
        without = {k: v for k, v in row["config"]["federation"].items() if k != "split_mode"}
        assert without == orig.config["federation"]


def test_temporal_battery_covers_the_headline_390_runs() -> None:
    from experiments.gen_temporal_battery import DEFAULT_MANIFESTS, temporal_rows

    if not all(p.exists() for p in DEFAULT_MANIFESTS):
        pytest.skip("source battery manifests not present")
    rows = temporal_rows()
    assert len(rows) == 390
    blocks = {r["block"] for r in rows}
    assert blocks == {"t1a_temporal", "t1b_temporal"}
    assert sum(1 for r in rows if r["block"] == "t1a_temporal") == 90
    assert sum(1 for r in rows if r["block"] == "t1b_temporal") == 300
    assert len({r["run_id"] for r in rows}) == 390
    assert {r["model"] for r in rows} == {"cnn_lstm", "snn", "ssm"}
