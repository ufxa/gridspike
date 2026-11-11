"""phase 2 tests for the models/metrics module: models (CNN-LSTM, spiking) and metrics.

Covers the contract correctness bar (CONTRACT.md §models/metrics): forward
shapes, surrogate gradient flow, a hand-computed SynOps check, rate-coding
reproducibility, a hand-counted MAC total, latency-percentile monotonicity and
the RunResult round-trip (incl. the amendment 2026-07-28d §4 provenance
fields) — plus Hypothesis property tests and a learnability sanity loop.
"""

from __future__ import annotations

import json
import time

import pytest
import torch
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import nn

from fedgrid.metrics.energy import E_AC_PJ, E_MAC_PJ, energy_uj
from fedgrid.metrics.latency import measure_inference_latency
from fedgrid.metrics.runresult import (
    RunResult,
    current_env_fingerprint,
    current_git_commit,
    load_run,
    save_run,
)
from fedgrid.models.cnn_lstm import CNNLSTM
from fedgrid.models.snn import LIF, SpikingClassifier

# --------------------------------------------------------------------- CNN-LSTM


def test_cnn_lstm_forward_shape() -> None:
    torch.manual_seed(0)
    model = CNNLSTM()
    out = model(torch.randn(4, 11, 64))
    assert out.shape == (4, 8)
    # Non-default config must reshape accordingly.
    small = CNNLSTM(in_ch=3, n_classes=5, conv_channels=(8,), lstm_hidden=16)
    assert small(torch.randn(2, 3, 64)).shape == (2, 5)
    # Eval mode is deterministic (dropout off).
    model.eval()
    x = torch.randn(2, 11, 64)
    assert torch.equal(model(x), model(x))


def test_cnn_lstm_gradients_flow() -> None:
    torch.manual_seed(0)
    model = CNNLSTM()
    loss = F.cross_entropy(model(torch.randn(8, 11, 64)), torch.randint(0, 8, (8,)))
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_cnn_lstm_macs_hand_count_small_config() -> None:
    # Hand count, T = 64 (data contract), in_ch=2, conv=(3,), k=3, lstm_hidden=4,
    # lstm_layers=1, n_classes=2:
    #   conv : 64 * 3 * 2 * 3            = 1,152   (same padding: T_out = T_in)
    #   pool halves T: 64 -> 32
    #   lstm : 32 * 4 * 4 * (3 + 4)      = 3,584   (4 gates, W_ih + W_hh matmuls)
    #   head : 4 * 2                     = 8
    #   total                            = 4,744
    model = CNNLSTM(in_ch=2, n_classes=2, conv_channels=(3,), kernel=3, lstm_hidden=4)
    assert model.macs_per_inference() == 4744


def test_cnn_lstm_macs_default_config() -> None:
    # conv1 64*32*11*5 = 112,640; conv2 32*64*32*5 = 327,680;
    # lstm 16*4*128*(64+128) = 1,572,864; head 128*8 = 1,024 -> 2,014,208.
    assert CNNLSTM().macs_per_inference() == 2_014_208


# -------------------------------------------------------------------- SNN model


@pytest.mark.parametrize("coding", ["rate", "delta"])
def test_snn_forward_shape(coding: str) -> None:
    torch.manual_seed(0)
    model = SpikingClassifier(coding=coding)
    out = model(torch.randn(4, 11, 64))
    assert out.shape == (4, 8)
    assert torch.isfinite(out).all()


def test_snn_input_validation() -> None:
    with pytest.raises(ValueError, match="expected"):
        SpikingClassifier()(torch.randn(2, 11, 32))  # wrong T
    with pytest.raises(ValueError, match="expected"):
        SpikingClassifier()(torch.randn(2, 5, 64))  # wrong C
    with pytest.raises(ValueError, match="coding"):
        SpikingClassifier(coding="ttfs")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="divisible"):
        SpikingClassifier(coding="delta", T_steps=7)  # 64 % 7 != 0


def test_surrogate_gradients_flow() -> None:
    """The arctan surrogate must deliver non-zero gradients to every LIF parameter."""
    torch.manual_seed(0)
    model = SpikingClassifier(hidden=32, layers=2, T_steps=8)
    loss = F.cross_entropy(model(torch.randn(8, 11, 64)), torch.randint(0, 8, (8,)))
    loss.backward()
    for i, lif in enumerate(model.lifs):
        assert lif._beta_raw.grad is not None and lif._beta_raw.grad.abs().sum() > 0, f"lif{i} beta"
        assert lif.threshold.grad is not None and lif.threshold.grad.abs().sum() > 0, f"lif{i} thr"
    for name in ("fc_in", "readout"):
        w = getattr(model, name).weight
        assert w.grad is not None and w.grad.abs().sum() > 0, name


def test_lif_beta_stays_bounded() -> None:
    lif = LIF(4, beta_init=0.9)
    assert torch.allclose(lif.beta, torch.full((4,), 0.9), atol=1e-6)
    with torch.no_grad():
        lif._beta_raw.fill_(10.0)  # strong upward drift under SGD
    assert (lif.beta < 1.0).all()
    with torch.no_grad():
        lif._beta_raw.fill_(1e6)  # fp32 sigmoid saturates to exactly 1.0 here —
    assert (lif.beta <= 1.0).all()  # still never explosive (beta > 1 impossible)
    with torch.no_grad():
        lif._beta_raw.fill_(-10.0)
    assert (lif.beta > 0.0).all()
    with pytest.raises(ValueError, match="beta_init"):
        LIF(4, beta_init=1.5)


def test_synops_matches_hand_computed_value() -> None:
    """SynOps hand check on a tiny fixed net (spikes forced deterministic).

    Net: in_ch=1, window_t=2 (F_in = 1*2 = 2), hidden=2, layers=2, T_steps=2,
    n_classes=2, rate coding. Input is +40 everywhere so sigmoid(40) == 1.0 in
    float32 and every coding neuron fires at every step (torch.rand samples
    [0, 1), so u < 1 always). All Linear weights = 1, biases = 0, so each LIF
    neuron receives current 2.0 > threshold 1.0 and fires every step (hard reset
    to 0, then again 2.0 next step).

    Hand arithmetic (B=1), SynOps = spikes_in x fan_out per layer per timestep:
      coding -> fc_in : 2 feat * 2 steps = 4 spikes, fan_out 2 ->  8
      lif1  -> fc_h   : 2 nrn  * 2 steps = 4 spikes, fan_out 2 ->  8
      lif2  -> readout: 2 nrn  * 2 steps = 4 spikes, fan_out 2 ->  8
      total SynOps                                             -> 24
    dense_macs = T_steps * (F_in*h + h*h + h*n_cls) = 2 * (4+4+4) = 24
    spike_rate = (4+4+4) spikes / (2 steps * (2+2+2) sources)    = 1.0
    ratio      = 24 / 24                                          = 1.0
    """
    tiny = SpikingClassifier(
        in_ch=1, n_classes=2, hidden=2, layers=2, T_steps=2, coding="rate", seed=0, window_t=2
    )
    with torch.no_grad():
        for lin in (tiny.fc_in, tiny.fc_hidden[0], tiny.readout):
            lin.weight.fill_(1.0)
            lin.bias.zero_()
    x = torch.full((1, 1, 2), 40.0)
    tiny(x)
    s = tiny.synops()
    assert s == {"synops": 24.0, "dense_macs": 24.0, "spike_rate": 1.0, "ratio": 1.0}
    # Per-layer spike-count buffers agree with the same arithmetic.
    assert [float(lif.spike_count) for lif in tiny.lifs] == [4.0, 4.0]

    # Counters accumulate across forwards until reset.
    tiny(x)
    assert tiny.synops()["synops"] == 48.0
    tiny.reset_counters()
    assert tiny.synops() == {"synops": 0.0, "dense_macs": 0.0, "spike_rate": 0.0, "ratio": 0.0}
    assert all(float(lif.spike_count) == 0.0 for lif in tiny.lifs)


def test_rate_coding_reproduces_under_fixed_seed() -> None:
    a = SpikingClassifier(seed=7)
    b = SpikingClassifier(seed=7)
    b.load_state_dict(a.state_dict())  # same weights; coding seed matches
    x = torch.randn(2, 11, 64, generator=torch.Generator().manual_seed(3))
    out_a = a(x)
    assert torch.equal(out_a, b(x))  # across instances
    assert torch.equal(out_a, a(x))  # across calls (coding is a fixed dither)
    # A different coding seed draws different spikes -> different logits.
    c = SpikingClassifier(seed=8)
    c.load_state_dict(a.state_dict())
    c.seed = 8  # load_state_dict does not carry the coding seed; be explicit
    assert not torch.equal(out_a, c(x))


def test_delta_coding_is_deterministic_and_sees_edges() -> None:
    model = SpikingClassifier(coding="delta", delta_threshold=1.0)
    x = torch.zeros(1, 11, 64)
    x[0, 3, 16:48] += 2.5  # square pulse: ON edge at t=16, OFF edge at t=48
    assert torch.equal(model(x), model(x))
    spikes = model._encode(x)  # (T_steps, B, F)
    assert float(spikes.sum()) == 2.0  # exactly the two edges, nothing else
    flat = torch.zeros(1, 11, 64)  # constant input -> contrast coding stays silent
    assert float(model._encode(flat).sum()) == 0.0


# ---------------------------------------------------------------------- metrics


def test_energy_uj_reference_values() -> None:
    out = energy_uj(synops=1e6, macs=1e6)
    assert out["snn_uJ"] == pytest.approx(1e6 * E_AC_PJ * 1e-6)  # 0.9 uJ
    assert out["ann_uJ"] == pytest.approx(1e6 * E_MAC_PJ * 1e-6)  # 4.6 uJ
    assert out["ratio"] == pytest.approx(0.9 / 4.6)
    with pytest.raises(ValueError):
        energy_uj(synops=-1.0, macs=1.0)
    with pytest.raises(ValueError):
        energy_uj(synops=1.0, macs=0.0)


@given(
    synops=st.floats(min_value=0.0, max_value=1e15, allow_nan=False, allow_infinity=False),
    macs=st.floats(min_value=1e-3, max_value=1e15, allow_nan=False, allow_infinity=False),
)
def test_energy_uj_ratio_consistent(synops: float, macs: float) -> None:
    """For all positive inputs: components non-negative and ratio == snn/ann."""
    out = energy_uj(synops, macs)
    assert set(out) == {"snn_uJ", "ann_uJ", "ratio"}
    assert out["snn_uJ"] >= 0.0 and out["ann_uJ"] > 0.0
    assert out["ratio"] == pytest.approx(out["snn_uJ"] / out["ann_uJ"], rel=1e-9)
    assert out["snn_uJ"] == pytest.approx(synops * E_AC_PJ * 1e-6, rel=1e-9)


@settings(max_examples=12, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    scale=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
    coding=st.sampled_from(["rate", "delta"]),
)
def test_synops_nonnegative_and_zero_after_reset(seed: int, scale: float, coding: str) -> None:
    """Property: counters are non-negative, ratio <= 1, and reset really zeroes."""
    model = SpikingClassifier(
        in_ch=2, n_classes=3, hidden=4, layers=2, T_steps=2, coding=coding, seed=seed, window_t=8
    )
    x = scale * torch.randn(3, 2, 8, generator=torch.Generator().manual_seed(seed))
    model(x)
    s = model.synops()
    assert all(v >= 0.0 for v in s.values())
    assert s["dense_macs"] > 0.0
    # SynOps can never exceed the fully-active dense cost of the same topology.
    assert s["ratio"] <= 1.0 + 1e-12
    assert s["spike_rate"] <= 1.0 + 1e-12
    model.reset_counters()
    assert model.synops() == {"synops": 0.0, "dense_macs": 0.0, "spike_rate": 0.0, "ratio": 0.0}


def test_latency_stats_monotone() -> None:
    model = nn.Linear(16, 4)
    model.train()  # prove mode save/restore
    stats = measure_inference_latency(model, torch.randn(1, 16), n=50, device="cpu", warmup=5)
    assert stats.mean_ms > 0.0
    assert stats.p50_ms <= stats.p95_ms <= stats.p99_ms
    assert model.training  # restored
    # Works on the real SNN too (counter accounting must survive no_grad eval).
    snn = SpikingClassifier(hidden=16, layers=2)
    s2 = measure_inference_latency(snn, torch.randn(1, 11, 64), n=10, device="cpu", warmup=2)
    assert s2.p50_ms <= s2.p95_ms <= s2.p99_ms
    with pytest.raises(ValueError):
        measure_inference_latency(model, torch.randn(1, 16), n=0)


def _sample_run() -> RunResult:
    return RunResult(
        method="fedavg+krum",
        seed=42,
        config={"rounds": 3, "alpha": 0.5, "model": "snn"},
        rounds=[{"round": 1, "acc": 0.71}, {"round": 2, "acc": 0.83}],
        final={"acc": 0.91, "macro_f1": 0.89, "confusion": [[5, 0], [1, 4]]},
        energy={"synops": 1.9e6, "ratio": 0.11},
        latency={"p50_ms": 0.8, "p99_ms": 1.9},
        git_commit="deadbeefcafe",
        env_fingerprint="py3.14.3 | torch 2.11.0+cu128 | cuda 12.8 | RTX 3060",
    )


def test_runresult_roundtrip(tmp_path) -> None:
    r = _sample_run()
    p = tmp_path / "runs" / "r0.json"  # nested dir: save_run must create it
    save_run(r, p)
    assert load_run(p) == r  # dataclass equality covers every field incl. provenance
    # Byte-stable serialisation (the determinism test elsewhere hashes this).
    save_run(r, tmp_path / "again.json")
    assert p.read_bytes() == (tmp_path / "again.json").read_bytes()


def test_runresult_loads_pre_amendment_files(tmp_path) -> None:
    """Files written before amendment 2026-07-28d §4 lack the provenance fields."""
    old = {
        "method": "fedavg",
        "seed": 1,
        "config": {},
        "rounds": [],
        "final": {"acc": 0.5},
        "energy": None,
        "latency": None,
    }
    p = tmp_path / "old.json"
    p.write_text(json.dumps(old), encoding="utf-8")
    r = load_run(p)
    assert (r.git_commit, r.env_fingerprint) == ("unknown", "unknown")
    assert r.final == {"acc": 0.5}


def test_provenance_helpers() -> None:
    # Repo has `git init` but zero commits (amendment 2026-07-28c) -> "unknown"
    # is legitimate; once a commit exists, a 40-hex hash appears.
    commit = current_git_commit("E:/paper7/fedgrid")
    is_hash = len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
    assert commit == "unknown" or is_hash
    assert current_git_commit("E:/paper7/definitely-not-a-repo-xyz") == "unknown"
    fp = current_env_fingerprint()
    assert fp.startswith("py") and "torch" in fp


# ------------------------------------------------------- learnability sanity loop


def _toy_data(n: int = 200, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Separable 8-class toy: class k puts a square pulse on channel k.

    The pulse gives every coding something to see: an amplitude offset for rate
    coding, ON/OFF edges (t=16 rising, t=48 falling) for delta coding.
    """
    gen = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % 8
    x = 0.3 * torch.randn(n, 11, 64, generator=gen)
    for i in range(n):
        x[i, int(y[i]), 16:48] += 2.5
    return x, y


def _train_accuracy(model: nn.Module, lr: float, epochs: int) -> float:
    x, y = _toy_data()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(ep))
        for i in range(0, len(y), 64):
            idx = perm[i : i + 64]
            opt.zero_grad()
            F.cross_entropy(model(x[idx]), y[idx]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return float((model(x).argmax(1) == y).float().mean())


@pytest.mark.parametrize(
    ("name", "factory", "lr"),
    [
        ("cnn_lstm", lambda: CNNLSTM(), 1e-3),
        ("snn_rate", lambda: SpikingClassifier(seed=0), 5e-3),
        ("snn_delta", lambda: SpikingClassifier(coding="delta", seed=0, delta_threshold=1.0), 5e-3),
    ],
)
def test_toy_task_learns(name: str, factory, lr: float) -> None:
    """200 synthetic samples, separable task: >70% train accuracy in <60 s CPU.

    Proves learning actually works through the surrogate gradient / conv-LSTM
    stack, not just that shapes line up. Empirically all three arms reach 100%
    in ~1 s; the 60 s ceiling is the contract's CI bound, not the expectation.
    """
    torch.manual_seed(0)
    t0 = time.perf_counter()
    acc = _train_accuracy(factory(), lr=lr, epochs=15)
    elapsed = time.perf_counter() - t0
    assert acc > 0.70, f"{name}: train accuracy {acc:.3f} <= 0.70"
    assert elapsed < 60.0, f"{name}: took {elapsed:.1f}s on CPU"
