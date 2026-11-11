"""the data-simulation module correctness bar: OpenDSS co-sim, events, windowing, federation, realism.

Contract items covered (CONTRACT.md, the data-simulation module section + amendment 2026-07-28b):
every feeder solves (converged); event windows carry the right label and
unaffected windows stay NORMAL; per-class counts match the Dirichlet plan
within tolerance; ``make_federation(seed=0)`` is bit-reproducible; the
normaliser never sees val/test; ``label_histogram`` rows sum to client sample
counts; CT/PT noise stays within the declared accuracy class over 1e4 samples;
the event-coupled frequency deviation steps in, decays over its fixed 8-sample
tail and stays inside the +/-0.3 Hz clip;
the NORMAL >= 90 % prior is achieved; a PV cloud-transient scenario provably
yields SAG-labelled windows; every REALISM.md parameter row has a non-empty
source. Plus Hypothesis property tests (event-plan invariants, split
partitioning, measurement bounds, window finiteness/label range for all seeds).

OpenDSS-backed fixtures are module-scoped and small (a few client-days) so the
file runs in well under a minute.
"""

from __future__ import annotations

import faulthandler
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# The FPC-built DSS C-API DLL raises a benign, internally-handled SEH exception
# (code 0xe0465043, "FPC") while loading on Windows; pytest's faulthandler would
# log a scary-but-harmless stack dump. Import the engine once with the handler
# paused so the suite output stays clean.
_was_enabled = faulthandler.is_enabled()
faulthandler.disable()
import opendssdirect  # noqa: E402,F401

if _was_enabled:
    faulthandler.enable()

from fedgrid.data import CHANNELS, EventClass, _split_indices  # noqa: E402
from fedgrid.simulation.events import EventSpec, sample_event_plan  # noqa: E402
from fedgrid.simulation.grid import (  # noqa: E402
    NONCONV_ABORT_CONSEC,
    NONCONV_ABORT_FRAC,
    FeederSim,
    FeederSpec,
    _should_abort_day,
    available_feeders,
)
from fedgrid.simulation.measurement import (  # noqa: E402
    MeasurementChain,
    MeasurementSpec,
    frequency_deviation,
)
from fedgrid.simulation.partition import (  # noqa: E402
    STRIDE,
    client_feeder_spec,
    make_federation,
    sim_calendar_day,
    windows_from_frame,
)

REPO = Path(__file__).resolve().parents[1]
T = 64


# --------------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def fed2():
    """2 clients x 1 day (ieee13 + ieee34) - the in-process federation fixture."""
    return make_federation(k=2, alpha=0.5, days=1, seed=0)


@pytest.fixture(scope="module")
def fed1x3():
    """1 client x 3 days - extra physical-sim mass for the prior check."""
    return make_federation(k=1, alpha=0.5, days=3, seed=1)


DAY13_EVENTS = [
    EventSpec(EventClass.FAULT_LG, 3600.0, 240.0,
              {"r_ohm": 5.0, "loc_frac": 0.4, "phase": 1}),
    EventSpec(EventClass.OVERLOAD, 10800.0, 1800.0, {"mult": 2.0}),
    EventSpec(EventClass.SAG, 43200.0, 300.0, {"depth_pu": 0.80, "mech_pv": 0.0}),
]


@pytest.fixture(scope="module")
def day13():
    """One ieee13 day with a mid-feeder LG fault, an overload and a source sag."""
    sim = FeederSim(FeederSpec("ieee13", load_shape_seed=7, pv_penetration=0.15))
    return sim.run_day(0, DAY13_EVENTS)


# ------------------------------------------------------------------ feeders/day

def test_all_feeders_solve():
    names = available_feeders()
    assert names == ["ieee13", "ieee34", "ieee37", "ieee123", "eulv"]
    for name in names:
        assert FeederSim(FeederSpec(name, load_shape_seed=0)).solve_baseline(), name


def test_run_day_contract(day13):
    assert list(day13.columns) == list(CHANNELS) + ["label"]
    assert len(day13) == 1440
    assert np.isfinite(day13.to_numpy(dtype=np.float64)).all()
    assert set(np.unique(day13["label"])) <= set(range(8))
    # sample-accurate labels: exactly the scheduled intervals
    lbl = day13["label"].to_numpy()
    assert (lbl[60:64] == int(EventClass.FAULT_LG)).all()
    assert (lbl[180:210] == int(EventClass.OVERLOAD)).all()
    assert (lbl[720:725] == int(EventClass.SAG)).all()
    covered = np.zeros(1440, bool)
    covered[60:64] = covered[180:210] = covered[720:725] = True
    assert (lbl[~covered] == 0).all()


def test_event_physics(day13):
    lbl = day13["label"].to_numpy()
    norm = day13[lbl == 0]
    lg = day13[lbl == int(EventClass.FAULT_LG)]
    # LG fault: zero-sequence current is the signature at the substation CT
    assert lg["i0"].mean() > 4.0 * norm["i0"].mean()
    assert lg["ia"].mean() > 1.2 * norm["ia"].mean()
    # overload: active power well above the surrounding night hours
    ovl = day13["p"].iloc[180:210].mean()
    night = day13["p"].iloc[np.r_[120:180, 210:270]].mean()
    assert ovl > 1.4 * night
    # source sag: voltage depressed vs pre-event, into the undervoltage regime
    sag_va = day13["va"].iloc[720:725].mean()
    pre_va = day13["va"].iloc[700:715].mean()
    assert sag_va < 0.95 and sag_va < pre_va - 0.05


def test_windows_match_reference_rule(day13):
    xs, ys = windows_from_frame(day13)
    assert xs.shape == (87, 11, T) and xs.dtype == np.float32
    assert ys.shape == (87,) and ys.dtype == np.int64
    assert np.isfinite(xs).all()
    # independent reference for the labelling rule: modal non-NORMAL class of
    # the window if it covers >= 2 samples, else NORMAL
    lbl = day13["label"].to_numpy()
    for w, s in enumerate(range(0, 1440 - T + 1, STRIDE)):
        counts = np.bincount(lbl[s:s + T], minlength=8)
        counts[0] = 0
        expected = int(counts.argmax()) if counts.max() >= 2 else 0
        assert ys[w] == expected
    # every scheduled class shows up; windows without event samples stay NORMAL
    assert {int(e.cls) for e in DAY13_EVENTS} <= set(ys.tolist())


def test_windows_per_client_budget():
    per_day = (1440 - T) // STRIDE + 1
    assert per_day == 87
    assert 1500 <= 30 * per_day <= 3000  # contract: ~1.5-3k windows/client at 30 days


def test_pv_cloud_transient_yields_sag_windows():
    """A PV-mechanism sag on a high-penetration feeder produces SAG windows.

    The event couples an area-wide deep cloud front (irradiance x0.1) with the
    upstream depression it causes system-wide; the metering bus must show a
    real voltage dip below 0.9 pu and a *rise* in imported power (PV lost),
    and the windowed labels must contain SAG.
    """
    ev = EventSpec(EventClass.SAG, 43200.0, 360.0, {"depth_pu": 0.78, "mech_pv": 1.0})
    sim = FeederSim(FeederSpec("ieee13", load_shape_seed=11, pv_penetration=0.40))
    df = sim.run_day(4, [ev])
    lbl = df["label"].to_numpy()
    ev_idx = slice(720, 726)
    pre_idx = slice(690, 715)
    assert (lbl[ev_idx] == int(EventClass.SAG)).all()
    assert df["va"].iloc[ev_idx].mean() < 0.90
    assert df["p"].iloc[ev_idx].mean() > df["p"].iloc[pre_idx].mean()  # PV output lost
    _, ys = windows_from_frame(df)
    assert int(EventClass.SAG) in set(ys.tolist())


def test_eulv_fault_converges_with_signature():
    ev = EventSpec(EventClass.FAULT_LG, 30000.0, 240.0,
                   {"r_ohm": 0.05, "loc_frac": 0.6, "phase": 3})
    df = FeederSim(FeederSpec("eulv", load_shape_seed=3, pv_penetration=0.0)).run_day(1, [ev])
    lbl = df["label"].to_numpy()
    assert df[lbl == 1]["i0"].mean() > 4.0 * df[lbl == 0]["i0"].mean()


# ------------------------------------------------------ solver non-convergence

def _battery_case(bseed: int, cid: int, sim_day: int, alpha: float = 0.5, k: int = 10):
    """Reconstruct the exact (spec, plan, calendar_day) a battery run would use.

    Mirrors ``make_federation``'s seeding via the module's own public/internal
    seeding surface, so these are genuine battery cases - not look-alikes.
    """
    from fedgrid.data import N_CLASSES
    from fedgrid.simulation import partition as P

    rng = np.random.default_rng(np.random.SeedSequence([P._FED_TAG, bseed]))
    weights = rng.dirichlet(alpha * np.ones(N_CLASSES), size=k)
    ev_rng = np.random.default_rng(np.random.SeedSequence([P._FED_TAG, bseed, cid, 1]))
    plan = None
    for _ in range(sim_day + 1):
        plan = sample_event_plan(ev_rng, 86400.0, weights[cid])
    return client_feeder_spec(bseed, cid), plan, sim_calendar_day(sim_day)


def test_should_abort_day_policy():
    """The abort predicate: tolerate noise, abort a genuine divergence."""
    n = 1440
    assert not _should_abort_day(n_nonconv=4, consec=4, n_steps=n)     # observed noise
    assert not _should_abort_day(n_nonconv=15, consec=1, n_steps=n)    # 1.04 % < 5 % (ieee37 case)
    assert not _should_abort_day(n_nonconv=72, consec=1, n_steps=n)    # 5.00 % == threshold, ok
    assert _should_abort_day(n_nonconv=73, consec=1, n_steps=n)        # 5.07 % > 5 %: abort
    assert not _should_abort_day(n_nonconv=9, consec=9, n_steps=n)     # 9 in a row ok
    assert _should_abort_day(n_nonconv=10, consec=10, n_steps=n)       # 10 in a row: abort
    assert (NONCONV_ABORT_FRAC, NONCONV_ABORT_CONSEC) == (0.05, 10)


def test_ieee34_known_divergence_recovers_deterministically():
    """The exact case that aborted a full battery run now completes gracefully.

    IEEE 34-bus, federation seed 0, client 1, simulated day 10 (calendar 123):
    a few early-morning steps are ill-conditioned; they must be held (not
    abort), stay under the tolerated rate, keep the data finite, and reproduce
    bit-for-bit including the held rows and the held-step count.
    """
    spec, plan, cal = _battery_case(bseed=0, cid=1, sim_day=10)
    assert spec.name == "ieee34" and cal == 123
    df = FeederSim(spec).run_day(cal, plan)
    n_bad, n = df.attrs["n_nonconverged"], df.attrs["n_steps"]
    assert n == 1440
    assert 0 < n_bad <= NONCONV_ABORT_FRAC * n          # held, not aborted, in policy
    assert np.isfinite(df.to_numpy(dtype=np.float64)).all()
    # windows still assemble and carry the scheduled fault label
    _, ys = windows_from_frame(df)
    assert int(EventClass.FAULT_LG) in set(ys.tolist())
    # deterministic under seed: identical frame (incl. held rows) and count
    df2 = FeederSim(spec).run_day(cal, plan)
    assert df.equals(df2)
    assert df2.attrs["n_nonconverged"] == n_bad


def test_battery_stiff_slice_completes_under_threshold():
    """A convergence-exercising slice across battery seeds and alphas completes.

    Sweeps the stiff IEEE 34-bus client across the five alpha=0.5 battery seeds
    plus the alpha=0.1 and IID(alpha=100) corners at the known-stiff day. Each
    run_day must finish (no abort) with a held-rate under the tolerated fraction
    and finite data. (The full 14-federation pre-generation is the
    the overnight batch job; this proves the mechanism holds beyond seed 0.)
    """
    cases = [(s, 1, 10, 0.5) for s in range(5)]
    cases += [(0, 1, 10, 0.1), (0, 1, 10, 100.0), (0, 6, 10, 0.5)]
    total_bad = total = 0
    for bseed, cid, sim_day, alpha in cases:
        spec, plan, cal = _battery_case(bseed, cid, sim_day, alpha=alpha)
        assert spec.name == "ieee34"
        df = FeederSim(spec).run_day(cal, plan)   # must not raise
        n_bad, n = df.attrs["n_nonconverged"], df.attrs["n_steps"]
        assert n_bad <= NONCONV_ABORT_FRAC * n, (bseed, cid, alpha, n_bad)
        assert np.isfinite(df.to_numpy(dtype=np.float64)).all()
        total_bad += n_bad
        total += n
    assert total_bad / total < NONCONV_ABORT_FRAC  # aggregate stays tiny


# ------------------------------------------------------------- event statistics

def test_event_plan_matches_dirichlet_weights():
    """Realised event-class mix tracks class_weights[1:] (the Dirichlet plan)."""
    rng = np.random.default_rng(123)
    w = np.array([0.30, 0.20, 0.05, 0.05, 0.10, 0.10, 0.15, 0.05])
    target = w[1:] / w[1:].sum()
    counts = np.zeros(7)
    for _ in range(300):
        for ev in sample_event_plan(rng, 86400.0, w):
            counts[int(ev.cls) - 1] += 1
    assert counts.sum() > 250  # Poisson(1.4) x 300 days, minus rare placement skips
    realised = counts / counts.sum()
    assert np.abs(realised - target).sum() < 0.15


def test_class_prior_normal_at_least_90pct(fed2, fed1x3):
    """NORMAL >= 90 % of windows (contract realism bar).

    Binding check at paper scale: the window-level prior over a 200-day plan
    ensemble (the same sampler that drives every generated day). The physical
    OpenDSS fixtures add an end-to-end aggregate; their per-client-day variance
    is Poisson, so the small-fixture floor is looser.
    """
    rng = np.random.default_rng(0)
    weights = np.random.default_rng(1).dirichlet(0.5 * np.ones(8), size=10)
    win_starts = range(0, 1440 - T + 1, STRIDE)
    n_win = ev_win = 0
    for d in range(200):
        plan = sample_event_plan(rng, 86400.0, weights[d % 10])
        lbl = np.zeros(1440, np.int64)
        for ev in plan:
            s = int(ev.start_s // 60)
            e = min(1440, max(s + 1, int(np.ceil((ev.start_s + ev.duration_s) / 60.0))))
            lbl[s:e] = int(ev.cls)
        for s in win_starts:
            n_win += 1
            c = np.bincount(lbl[s:s + T], minlength=8)
            c[0] = 0
            ev_win += int(c.max() >= 2)
    ensemble_normal = 1.0 - ev_win / n_win
    assert 0.90 <= ensemble_normal <= 0.975
    # end-to-end physical aggregate over the OpenDSS fixtures (5 client-days)
    hist = np.vstack([fed2.label_histogram(), fed1x3.label_histogram()])
    assert hist[:, 0].sum() / hist.sum() >= 0.90


# ------------------------------------------------------------------- federation

def test_federation_structure(fed2):
    assert len(fed2.clients) == 2
    for cid, c in enumerate(fed2.clients):
        assert c.meta.cid == cid
        assert c.meta.feeder.name == available_feeders()[cid]
        assert c.meta.class_weights.shape == (8,)
        n = len(c)
        assert n == 87
        assert (len(c.train), len(c.val), len(c.test)) == (60, 13, 14)
        idx = np.concatenate([c.train.indices, c.val.indices, c.test.indices])
        assert np.array_equal(np.sort(idx), np.arange(n))  # disjoint cover
        x, y = c[0]
        assert x.shape == (11, T) and x.dtype.__str__() == "torch.float32"
        assert y.dtype.__str__() == "torch.int64" and 0 <= int(y) <= 7
    assert len(fed2.global_test()) == sum(len(c.test) for c in fed2.clients)
    hist = fed2.label_histogram()
    assert hist.shape == (2, 8)
    assert np.array_equal(hist.sum(axis=1), np.array([len(c) for c in fed2.clients]))


def test_make_federation_bit_reproducible(fed2):
    again = make_federation(k=2, alpha=0.5, days=1, seed=0)
    for a, b in zip(fed2.clients, again.clients, strict=True):
        assert np.array_equal(a.windows, b.windows)
        assert np.array_equal(a.labels, b.labels)
        assert np.array_equal(a.meta.class_weights, b.meta.class_weights)
        assert np.array_equal(a.normalizer.mean, b.normalizer.mean)
        assert np.array_equal(a.normalizer.std, b.normalizer.std)
        assert a.train.indices == b.train.indices
        assert a.test.indices == b.test.indices


def test_normalizer_fitted_on_train_only(fed2):
    for c in fed2.clients:
        tr = np.asarray(c.train.indices)
        train_mean = c.windows[tr].mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
        train_std = c.windows[tr].std(axis=(0, 2), dtype=np.float64).astype(np.float32)
        assert np.array_equal(c.normalizer.mean[:, 0], train_mean)
        assert np.array_equal(c.normalizer.std[:, 0], np.maximum(train_std, np.float32(1e-6)))
        full_mean = c.windows.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
        assert not np.array_equal(c.normalizer.mean[:, 0], full_mean)  # val/test unseen


def test_generate_cli_and_load_roundtrip(tmp_path):
    from fedgrid.data import load_federation
    from fedgrid.simulation.generate import main

    assert main(["--k", "1", "--days", "1", "--seed", "0", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "client_00.npz").is_file()
    manifest = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    assert "windows_per_min" in manifest and "sha256_windows" in manifest
    ref = make_federation(k=1, alpha=0.5, days=1, seed=0)
    loaded = load_federation(tmp_path)
    a, b = ref.clients[0], loaded.clients[0]
    assert np.array_equal(a.windows, b.windows)
    assert np.array_equal(a.labels, b.labels)
    assert a.train.indices == b.train.indices
    assert np.array_equal(a.normalizer.mean, b.normalizer.mean)


# ------------------------------------------------------------ measurement chain

def test_ct_pt_noise_within_class_over_1e4_samples():
    """Instrument-stage error bounded by the IEC 61869 class-0.5 limit (0.5 %).

    The instrument stage is isolated with a fine ADC (24-bit); the shipped
    16-bit ADC adds at most ~1 LSB on top, checked separately below.
    """
    spec = MeasurementSpec(adc_bits=24)
    chain = MeasurementChain(np.random.default_rng(2), spec)
    rng = np.random.default_rng(3)
    n = 10_000
    clean = np.column_stack([rng.uniform(0.5, 1.5, n) for _ in range(10)]
                            + [rng.uniform(-0.05, 0.05, n)])
    meas = chain.apply(clean, np.random.default_rng(4), dropout=False)
    rel = np.abs(meas[:, :10] / clean[:, :10] - 1.0)
    assert rel.max() <= spec.class_limit + 1e-9
    assert rel.std() > 0  # noise actually injected
    assert np.abs(meas[:, 10] - clean[:, 10]).max() <= spec.freq_noise_clip + 1e-9


def test_default_chain_error_budget_at_rated():
    spec = MeasurementSpec()  # shipped 16-bit config
    chain = MeasurementChain(np.random.default_rng(5), spec)
    n = 10_000
    clean = np.ones((n, 11))
    clean[:, 10] = 0.0
    meas = chain.apply(clean, np.random.default_rng(6), dropout=False)
    from fedgrid.simulation.measurement import _ADC_RANGE

    for j, ch in enumerate(CHANNELS[:10]):
        lo, hi = _ADC_RANGE[ch]
        lsb = (hi - lo) / 2**16
        assert np.abs(meas[:, j] - clean[:, j]).max() <= spec.class_limit + lsb + 1e-12


def test_adc_quantisation_grid():
    from fedgrid.simulation.measurement import _ADC_RANGE

    spec = MeasurementSpec()
    chain = MeasurementChain(np.random.default_rng(7), spec)
    rng = np.random.default_rng(8)
    clean = np.column_stack([rng.uniform(0.2, 1.8, 500) for _ in range(10)]
                            + [rng.uniform(-0.4, 0.4, 500)])
    meas = chain.apply(clean, np.random.default_rng(9), dropout=False)
    for j, ch in enumerate(CHANNELS):
        lo, hi = _ADC_RANGE[ch]
        lsb = (hi - lo) / 2**16
        steps = (meas[:, j] - lo) / lsb
        assert np.abs(steps - np.round(steps)).max() < 1e-6  # on the 16-bit grid


def test_dropout_holds_last_value():
    spec = MeasurementSpec(dropout_per_day=80.0)
    chain = MeasurementChain(np.random.default_rng(10), spec)
    rng = np.random.default_rng(11)
    clean = np.abs(rng.normal(1.0, 0.2, size=(1440, 11)))
    meas = chain.apply(clean, np.random.default_rng(12), dropout=True)
    held = (np.diff(meas, axis=0) == 0.0).all(axis=1)
    frac = held.mean()
    assert 0.02 < frac < 0.6  # episodes present at roughly the configured rate
    # default rate ~0.1 % of samples: same mechanism, just rarer (REALISM.md row)


def test_frequency_deviation_step_tail_clip_and_determinism():
    """Event coupling: step in, fixed 8-sample tail, +/-0.3 Hz clip, seeded.

    The AR(1) wander is drawn from ``rng`` before any event is consumed, so the
    same seed run with and without ``events`` isolates the event-coupled term
    exactly - no re-implementation of the wander process needed here.
    """
    n = 200
    labels = np.zeros(n, np.int64)
    start, end = 50, 60
    labels[start:end] = int(EventClass.FAULT_3PH)
    ev = [(start, end, int(EventClass.FAULT_3PH))]
    coupled = (frequency_deviation(np.random.default_rng(0), labels, 60.0, ev)
               - frequency_deviation(np.random.default_rng(0), labels, 60.0, []))

    # STEP, not a ramp: the full deviation is present at the first in-event
    # sample, held flat to end-1, with nothing at all before the window
    dev = coupled[start]
    assert -0.12 - 1e-12 <= dev <= -0.04 + 1e-12          # FAULT_3PH band
    assert np.allclose(coupled[start:end], dev)
    assert np.abs(coupled[:start]).max() == 0.0

    # TAIL: exactly 8 samples of dev * exp(-(t - end + 1)/2), then hard zero
    k = np.arange(8)
    assert np.allclose(coupled[end:end + 8], dev * np.exp(-(k + 1) / 2.0))
    assert np.abs(coupled[end + 8:]).max() == 0.0
    assert abs(coupled[end] / dev - np.exp(-0.5)) < 1e-12  # clearance steps down

    # TAIL truncated at the array end (no wrap-around, no padding): an event
    # clearing 3 samples from the end keeps 3 tail samples, not 8
    short = np.zeros(40, np.int64)
    short[30:37] = int(EventClass.FAULT_LG)
    ev_late = [(30, 37, int(EventClass.FAULT_LG))]
    tail = (frequency_deviation(np.random.default_rng(5), short, 60.0, ev_late)
            - frequency_deviation(np.random.default_rng(5), short, 60.0, []))
    assert np.count_nonzero(tail[37:]) == 3
    assert np.abs(tail[:30]).max() == 0.0

    # CLIP at +/-0.3 Hz. No single class can reach the rail (the largest band is
    # 120 mHz against 18 mHz-sigma wander), so it is driven the only way the
    # public API allows: overlapping windows, whose deviations accumulate. Ten
    # stacked FAULT_3PH events sum to <= -0.4 Hz whatever the draws.
    stacked = np.zeros(100, np.int64)
    stacked[40:50] = int(EventClass.FAULT_3PH)
    fd = frequency_deviation(np.random.default_rng(3), stacked, 60.0,
                             [(40, 50, int(EventClass.FAULT_3PH))] * 10)
    assert (fd[40:50] == -0.3).all()                       # saturated, not -1.2
    assert fd.min() >= -0.3 and fd.max() <= 0.3

    # DETERMINISM: same seed -> identical array, different seed -> different
    a = frequency_deviation(np.random.default_rng(11), labels, 60.0, ev)
    b = frequency_deviation(np.random.default_rng(11), labels, 60.0, ev)
    c = frequency_deviation(np.random.default_rng(12), labels, 60.0, ev)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)

    # the documented magnitude range: 5-120 mHz across the tabulated bands
    from fedgrid.simulation.measurement import _EVENT_FDEV

    mags = np.abs(np.asarray(list(_EVENT_FDEV.values())))
    assert (mags.min(), mags.max()) == (0.005, 0.12)


def test_wan_latency_distribution():
    from fedgrid.simulation.network import utility_wan_net_ms

    rng = np.random.default_rng(13)
    draws = np.array([utility_wan_net_ms(rng) for _ in range(4000)])
    assert draws.min() >= 5.0 and draws.max() <= 200.0
    assert 20.0 < np.median(draws) < 30.0
    assert np.percentile(draws, 99) < 110.0  # inside the 61850-90-12 100 ms class


# ----------------------------------------------------------------- realism docs

def test_realism_md_every_row_has_source():
    text = (REPO / "docs" / "REALISM.md").read_text(encoding="utf-8")
    rows = 0
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4 or cells[0] in ("Parameter", "") or set(cells[0]) <= {"-", " "}:
            continue
        rows += 1
        assert cells[-1], f"REALISM.md row without source: {cells[0]}"
        assert len(cells[-1]) > 10, f"suspiciously empty source: {cells[0]}"
    assert rows >= 35  # the full parameter table is present
    for needle in ("IEC 61869", "IEEE Std 1159", "Richardson", "Haurwitz",
                   "Dagenhart", "1909.06335", "C37.118", "Nature Communications"):
        assert needle in text, f"expected source citation missing: {needle}"


# ------------------------------------------------------------------- properties

@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 2**32 - 1))
def test_property_event_plan_invariants(seed):
    """For all seeds: classes in 1..7, durations in class bounds, no overlap."""
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(0.5 * np.ones(8))
    plan = sample_event_plan(np.random.default_rng(seed + 1), 86400.0, w)
    bounds = {1: (120, 900), 2: (120, 600), 3: (120, 600), 4: (120, 600),
              5: (300, 2700), 6: (120, 600), 7: (120, 600)}
    prev_end = -1e9
    for ev in plan:
        assert 1 <= int(ev.cls) <= 7
        lo, hi = bounds[int(ev.cls)]
        assert lo - 1e-6 <= ev.duration_s <= hi + 1e-6
        assert 0.0 <= ev.start_s and ev.start_s + ev.duration_s <= 86400.0
        assert ev.start_s >= prev_end  # sorted and non-overlapping
        prev_end = ev.start_s + ev.duration_s


@settings(max_examples=50, deadline=None)
@given(n=st.integers(10, 5000), seed=st.integers(0, 2**32 - 1))
def test_property_split_partitions_every_n_and_seed(n, seed):
    tr, va, te = _split_indices(n, seed)
    assert len(tr) + len(va) + len(te) == n
    assert np.array_equal(np.sort(np.concatenate([tr, va, te])), np.arange(n))
    assert len(tr) == int(n * 0.70) and len(va) == int(n * 0.15)


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 2**32 - 1))
def test_property_measurement_bounded_for_all_seeds(seed):
    spec = MeasurementSpec(adc_bits=24)
    chain = MeasurementChain(np.random.default_rng(seed), spec)
    rng = np.random.default_rng(seed + 1)
    clean = np.column_stack([rng.uniform(0.5, 1.5, 200) for _ in range(10)]
                            + [rng.uniform(-0.05, 0.05, 200)])
    meas = chain.apply(clean, np.random.default_rng(seed + 2), dropout=False)
    assert np.abs(meas[:, :10] / clean[:, :10] - 1.0).max() <= spec.class_limit + 1e-9


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 2**32 - 1))
def test_property_windows_finite_labels_in_range(seed):
    """windows_from_frame invariants on synthetic frames, for all seeds."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    n = 256
    df = pd.DataFrame(rng.normal(1.0, 0.3, size=(n, 11)), columns=list(CHANNELS))
    lbl = np.zeros(n, np.int64)
    for _ in range(rng.integers(0, 4)):
        cls = int(rng.integers(1, 8))
        s = int(rng.integers(0, n - 2))
        lbl[s:s + int(rng.integers(2, 30))] = cls
    df["label"] = lbl
    xs, ys = windows_from_frame(df)
    assert xs.shape == ((n - T) // STRIDE + 1, 11, T)
    assert np.isfinite(xs).all()
    assert ys.min() >= 0 and ys.max() <= 7
    if (lbl == 0).all():
        assert (ys == 0).all()
