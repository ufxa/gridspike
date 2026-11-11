"""phase 2 tests for ``fedgrid.fl``: engine, strategies, LDP.

Every cross-module dependency (Federation/ClientDataset, models, attacks,
validators) is a tiny in-file fake satisfying the same contract interface, so
this file stays green independently of Agents S/M/C progress. The contract
correctness bar covered here: centralised equivalence, exact FedAvg weighting,
FedProx(0) == FedAvg, clip/noise behaviour over 10^4 draws, the published
epsilon reference value, bit-reproducibility on CPU, val-loss improvement on a
tiny 2-round federation, plus Hypothesis property tests (FedAvg convex hull,
clipping never increases norm).
"""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from fedgrid.fl.engine import (
    ClientUpdate,
    FederatedEngine,
    ValidationReport,
    _client_torch_seed,
    _confusion,
    _f1_from_confusion,
    _flip_labels,
)
from fedgrid.fl.ldp import LDPConfig, clip_and_noise, epsilon
from fedgrid.fl.strategies import FedAvg, FedProx

# --------------------------------------------------------------------------- #
# Fakes: a synthetic federation of TensorDatasets satisfying the data contract
# interface the engine consumes (clients / train / val / test / global_test).
# --------------------------------------------------------------------------- #

N_FEATURES = 6
N_CLASSES = 3


class _Client:
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        n = len(y)
        a, b = int(0.7 * n), int(0.85 * n)
        self.train = TensorDataset(x[:a], y[:a])
        self.val = TensorDataset(x[a:b], y[a:b])
        self.test = TensorDataset(x[b:], y[b:])
        self.meta = None


class _Fed:
    def __init__(self, clients: list[_Client]) -> None:
        self.clients = clients

    def global_test(self) -> ConcatDataset:
        return ConcatDataset([c.test for c in self.clients])


def _blobs(n: int, seed: int, spread: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, N_CLASSES, n)
    centers = rng.normal(0.0, 2.0, (N_CLASSES, N_FEATURES))
    x = centers[y] + rng.normal(0.0, spread, (n, N_FEATURES))
    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.int64),
    )


def _make_fed(k: int = 3, n: int = 60, seed: int = 0) -> _Fed:
    return _Fed([_Client(*_blobs(n, seed * 1000 + cid)) for cid in range(k)])


def _model_fn() -> nn.Module:
    return nn.Sequential(nn.Linear(N_FEATURES, 16), nn.Tanh(), nn.Linear(16, N_CLASSES))


def _dropout_model_fn() -> nn.Module:
    # Dropout on purpose: proves the per-(round, client) torch re-seeding tames
    # stochastic layers, not just shuffle order.
    return nn.Sequential(
        nn.Linear(N_FEATURES, 16), nn.Tanh(), nn.Dropout(0.25), nn.Linear(16, N_CLASSES)
    )


def _l2(params: list[np.ndarray], ref: list[np.ndarray]) -> float:
    return math.sqrt(
        sum(
            float(((np.asarray(p, np.float64) - np.asarray(r, np.float64)) ** 2).sum())
            for p, r in zip(params, ref, strict=True)
        )
    )


def _stable_json(rounds: list[dict], drop: tuple[str, ...] = ("ts",)) -> str:
    """Serialised rounds with wall-clock metadata removed — the determinism unit."""
    return json.dumps(
        [{k: v for k, v in r.items() if k not in drop} for r in rounds], sort_keys=True
    )


def _accept_all(updates: list[ClientUpdate], ref: list[np.ndarray]) -> ValidationReport:
    return ValidationReport(
        accepted=[u.cid for u in updates],
        scores=np.zeros(len(updates)),
        consensus=None,
    )


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #


def test_fedavg_weighting_exact_on_toy_vectors() -> None:
    u1 = ClientUpdate(
        cid=0,
        n_samples=1,
        params=[np.array([1.0, 0.0], np.float32), np.array([[2.0]], np.float32)],
        train_loss=0.0,
    )
    u2 = ClientUpdate(
        cid=1,
        n_samples=3,
        params=[np.array([5.0, 4.0], np.float32), np.array([[6.0]], np.float32)],
        train_loss=0.0,
    )
    current = [np.zeros(2, np.float32), np.zeros((1, 1), np.float32)]
    out = FedAvg().aggregate([u1, u2], current)
    # (1*1 + 3*5)/4 = 4, (1*0 + 3*4)/4 = 3, (1*2 + 3*6)/4 = 5 — exact in float32.
    assert out[0].dtype == np.float32
    np.testing.assert_array_equal(out[0], np.array([4.0, 3.0], np.float32))
    np.testing.assert_array_equal(out[1], np.array([[5.0]], np.float32))


def test_fedavg_edge_cases() -> None:
    current = [np.array([1.5, -2.0], np.float32)]
    # Empty admitted set: current returned unchanged, as a copy.
    out = FedAvg().aggregate([], current)
    np.testing.assert_array_equal(out[0], current[0])
    out[0][0] = 99.0
    assert current[0][0] == 1.5
    # Mismatched structure is a hard error.
    bad = ClientUpdate(cid=0, n_samples=1, params=[np.zeros(2), np.zeros(3)], train_loss=0.0)
    with pytest.raises(ValueError):
        FedAvg().aggregate([bad], current)
    # All-zero sample counts fall back to a uniform mean.
    z1 = ClientUpdate(cid=0, n_samples=0, params=[np.array([2.0, 0.0])], train_loss=0.0)
    z2 = ClientUpdate(cid=1, n_samples=0, params=[np.array([4.0, 2.0])], train_loss=0.0)
    np.testing.assert_allclose(FedAvg().aggregate([z1, z2], current)[0], [3.0, 1.0])


def test_fedprox_hook_value_and_mu_zero() -> None:
    model = nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0]]))
        model.bias.copy_(torch.tensor([3.0]))
    ref = [torch.zeros_like(model.weight), torch.zeros_like(model.bias)]
    hook = FedProx(mu=0.1).client_loss_hook(model, ref)
    assert hook is not None
    # mu/2 * (1^2 + 2^2 + 3^2) = 0.05 * 14 = 0.7
    assert torch.isclose(hook, torch.tensor(0.7))
    assert FedProx(mu=0.0).client_loss_hook(model, ref) is None
    with pytest.raises(ValueError):
        FedProx(mu=-0.1)


@settings(max_examples=60, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.integers(min_value=1, max_value=10**6),
            hnp.arrays(
                np.float32,
                (3, 2),
                elements=st.floats(-100, 100, width=32, allow_nan=False),
            ),
            hnp.arrays(
                np.float32,
                (4,),
                elements=st.floats(-100, 100, width=32, allow_nan=False),
            ),
        ),
        min_size=1,
        max_size=5,
    )
)
def test_fedavg_output_within_convex_hull(raw) -> None:
    """FedAvg is a convex combination: every output coordinate stays within the
    per-coordinate [min, max] of the inputs, for any updates and weights."""
    updates = [
        ClientUpdate(cid=i, n_samples=w, params=[a.copy(), b.copy()], train_loss=0.0)
        for i, (w, a, b) in enumerate(raw)
    ]
    current = [np.zeros((3, 2), np.float32), np.zeros(4, np.float32)]
    out = FedAvg().aggregate(updates, current)
    tol = 1e-3
    for i in range(2):
        stack = np.stack([u.params[i] for u in updates]).astype(np.float64)
        assert np.all(out[i] >= stack.min(axis=0) - tol)
        assert np.all(out[i] <= stack.max(axis=0) + tol)


# --------------------------------------------------------------------------- #
# LDP: clip_and_noise
# --------------------------------------------------------------------------- #


def test_clip_noop_inside_ball_and_exact_on_boundary() -> None:
    rng = np.random.default_rng(0)
    ref = [np.zeros(4, np.float32), np.zeros((2, 2), np.float32)]
    small = [np.full(4, 0.1, np.float32), np.full((2, 2), 0.1, np.float32)]  # norm ~0.283
    out = clip_and_noise(small, ref, LDPConfig(clip=1.0, sigma=0.0), rng)
    for o, p in zip(out, small, strict=True):
        np.testing.assert_array_equal(o, p)  # inside the ball + sigma=0 -> identity

    big = [np.full(4, 3.0, np.float32), np.full((2, 2), 3.0, np.float32)]  # norm ~8.49
    out = clip_and_noise(big, ref, LDPConfig(clip=1.0, sigma=0.0), rng)
    assert abs(_l2(out, ref) - 1.0) < 1e-6  # projected exactly onto the clip sphere


def test_noise_matches_sigma_empirically_over_1e4_draws() -> None:
    clip, sigma = 2.0, 0.5
    cfg = LDPConfig(clip=clip, sigma=sigma, seed=0)
    ref = [np.zeros(100, np.float64)]
    rng = np.random.default_rng(1234)
    draws = np.stack(
        [clip_and_noise([ref[0].copy()], ref, cfg, rng)[0] for _ in range(10_000)]
    )
    # params == ref, so the output is pure N(0, (sigma*clip)^2): 1e6 samples.
    assert abs(draws.std() - sigma * clip) < 0.01 * sigma * clip
    assert abs(draws.mean()) < 5 * (sigma * clip) / math.sqrt(draws.size)


def test_clip_and_noise_deterministic_and_validated() -> None:
    params = [np.arange(6, dtype=np.float32).reshape(2, 3)]
    ref = [np.zeros((2, 3), np.float32)]
    cfg = LDPConfig(clip=1.0, sigma=0.3)
    a = clip_and_noise(params, ref, cfg, np.random.default_rng(7))
    b = clip_and_noise(params, ref, cfg, np.random.default_rng(7))
    np.testing.assert_array_equal(a[0], b[0])
    with pytest.raises(ValueError):
        clip_and_noise(params, ref, LDPConfig(clip=0.0, sigma=0.1), np.random.default_rng(0))
    with pytest.raises(ValueError):
        clip_and_noise(params, ref, LDPConfig(clip=1.0, sigma=-0.1), np.random.default_rng(0))
    with pytest.raises(ValueError):
        clip_and_noise(params, [], cfg, np.random.default_rng(0))


@settings(max_examples=60, deadline=None)
@given(
    hnp.arrays(np.float64, (5,), elements=st.floats(-1e3, 1e3, allow_nan=False)),
    hnp.arrays(np.float64, (5,), elements=st.floats(-1e3, 1e3, allow_nan=False)),
    hnp.arrays(np.float64, (3, 2), elements=st.floats(-1e3, 1e3, allow_nan=False)),
    hnp.arrays(np.float64, (3, 2), elements=st.floats(-1e3, 1e3, allow_nan=False)),
    st.floats(min_value=0.01, max_value=50.0, allow_nan=False),
)
def test_clip_never_increases_norm(p1, r1, p2, r2, clip) -> None:
    """For any update and clip radius: post-clip norm <= clip AND <= pre-clip norm.

    float64 end-to-end: the property under test is the projection math; the
    (float32-cast) round-trip behaviour is covered by the unit tests above.
    """
    params, ref = [p1, p2], [r1, r2]
    out = clip_and_noise(params, ref, LDPConfig(clip=clip, sigma=0.0), np.random.default_rng(0))
    norm_in = _l2(params, ref)
    norm_out = _l2(out, ref)
    assert norm_out <= clip * (1 + 1e-5) + 1e-6
    assert norm_out <= norm_in * (1 + 1e-5) + 1e-6


# --------------------------------------------------------------------------- #
# LDP: epsilon accounting
# --------------------------------------------------------------------------- #


def test_epsilon_matches_published_reference_value() -> None:
    """Anchor: Dwork & Roth 2014, Thm. A.1 (DOI 10.1561/0400000042).

    sigma = sqrt(2 ln(1.25/delta)) / eps0 makes the Gaussian mechanism
    (eps0, delta)-DP. The RDP accountant must land next to that published
    calibration: at eps0 = 0.5 it returns 0.5005... (0.11 % above), at
    eps0 = 1.0 it returns 1.0117... (1.2 % above) — both valid upper bounds.
    The exact digits are pinned as a regression guard.
    """
    delta = 1e-5
    s_half = math.sqrt(2 * math.log(1.25 / delta)) / 0.5  # 9.689610525210778
    eps_half = epsilon(s_half, 1, delta)
    assert abs(eps_half - 0.5) < 0.005  # published value: 0.5
    assert abs(eps_half - 0.5005492801617069) < 1e-12  # pinned exact digits

    s_one = math.sqrt(2 * math.log(1.25 / delta))  # 4.844805262605389
    eps_one = epsilon(s_one, 1, delta)
    assert abs(eps_one - 1.0) < 0.02  # published value: 1.0
    assert abs(eps_one - 1.011749486099886) < 1e-12


def test_epsilon_closed_form_equals_grid_minimum() -> None:
    """The closed-form alpha* minimisation must match an independent numeric
    minimisation of Mironov Prop. 3's conversion over a dense alpha grid."""
    for sigma, rounds in [(1.0, 1), (2.0, 50), (9.6896105, 1), (0.7, 200)]:
        delta = 1e-5
        c = rounds / (2 * sigma * sigma)
        # alpha* = 1 + sqrt(log(1/delta)/c) < 100 for every case above: sample
        # that region densely, keep a coarse tail as an escape check.
        alphas = np.concatenate(
            [np.linspace(1 + 1e-9, 100, 2_000_000), np.linspace(100, 50_000, 100_000)]
        )
        grid = float((c * alphas + math.log(1 / delta) / (alphas - 1)).min())
        assert abs(epsilon(sigma, rounds, delta) - grid) < 1e-5 * max(1.0, grid)
        assert epsilon(sigma, rounds, delta) <= grid + 1e-12  # closed form is the true min


def test_epsilon_composition_identity_and_monotonicity() -> None:
    # Gaussian RDP composition is exact: T rounds at sigma == 1 round at sigma/sqrt(T).
    assert math.isclose(epsilon(9.6896105, 4), epsilon(9.6896105 / 2, 1), rel_tol=1e-12)
    assert math.isclose(epsilon(6.0, 100), epsilon(0.6, 1), rel_tol=1e-12)
    assert epsilon(2.0, 10) < epsilon(1.0, 10)  # more noise, less loss
    assert epsilon(1.0, 20) > epsilon(1.0, 10)  # more rounds, more loss
    assert epsilon(0.0, 5) == math.inf
    with pytest.raises(ValueError):
        epsilon(1.0, 0)
    with pytest.raises(ValueError):
        epsilon(1.0, 5, delta=1.5)


# --------------------------------------------------------------------------- #
# Engine: centralised equivalence (K=1, FedAvg, 1 local epoch == plain SGD)
# --------------------------------------------------------------------------- #


def test_centralized_equivalence_k1() -> None:
    seed, rounds, bs, lr = 7, 2, 8, 0.05
    fed = _make_fed(k=1, n=40, seed=3)
    train_ds = fed.clients[0].train

    snaps: list[list[np.ndarray]] = []

    def cb(_event: dict) -> None:
        snaps.append([p.copy() for p in eng.global_params])

    eng = FederatedEngine(
        fed, _model_fn, FedAvg(), rounds=rounds, local_epochs=1, batch_size=bs,
        lr=lr, device="cpu", seed=seed, on_round=cb,
    )
    eng.run()

    # Plain SGD, constructed explicitly: same init, same per-round shuffled
    # batch order (derived from the engine's public seed tree), no federation.
    torch.manual_seed(seed)
    net = _model_fn()
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    for rnd in range(rounds):
        g = torch.Generator().manual_seed(_client_torch_seed(seed, rnd, 0))
        loader = DataLoader(train_ds, batch_size=bs, shuffle=True, generator=g)
        for xb, yb in loader:
            opt.zero_grad()
            F.cross_entropy(net(xb), yb).backward()
            opt.step()
        ref = [p.detach().numpy() for p in net.parameters()]
        for a, b in zip(snaps[rnd], ref, strict=True):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-7)


# --------------------------------------------------------------------------- #
# Engine: FedProx(0) == FedAvg, FedProx(mu>0) != FedAvg
# --------------------------------------------------------------------------- #


def _run(strategy, seed=5, model_fn=_model_fn, fed=None, **kw):
    params = dict(rounds=2, local_epochs=1, batch_size=16, lr=0.1, device="cpu", seed=seed)
    params.update(kw)
    eng = FederatedEngine(fed or _make_fed(k=3, n=60, seed=1), model_fn, strategy, **params)
    return eng, eng.run()


def test_fedprox_mu_zero_equals_fedavg_bitwise() -> None:
    eng_a, res_a = _run(FedAvg())
    eng_p, res_p = _run(FedProx(mu=0.0))
    # run_id/method legitimately differ; every numeric field must be identical.
    assert _stable_json(res_a.rounds, drop=("ts", "run_id")) == _stable_json(
        res_p.rounds, drop=("ts", "run_id")
    )
    for a, b in zip(eng_a.global_params, eng_p.global_params, strict=True):
        np.testing.assert_array_equal(a, b)


def test_fedprox_mu_positive_changes_trajectory() -> None:
    eng_a, _ = _run(FedAvg())
    eng_p, _ = _run(FedProx(mu=1.0))
    diff = max(
        float(np.abs(a - b).max())
        for a, b in zip(eng_a.global_params, eng_p.global_params, strict=True)
    )
    assert diff > 1e-6  # the proximal term must actually bite


# --------------------------------------------------------------------------- #
# Engine: bit-reproducibility on CPU
# --------------------------------------------------------------------------- #


def test_engine_bit_reproducible_same_seed_cpu() -> None:
    eng1, res1 = _run(FedAvg(), seed=3, model_fn=_dropout_model_fn)
    eng2, res2 = _run(FedAvg(), seed=3, model_fn=_dropout_model_fn)
    assert _stable_json(res1.rounds) == _stable_json(res2.rounds)
    for a, b in zip(eng1.global_params, eng2.global_params, strict=True):
        np.testing.assert_array_equal(a, b)  # bitwise, not just close
    assert res1.final == res2.final

    # Re-running the SAME engine reproduces the run: state re-derives from seed.
    res1b = eng1.run()
    assert _stable_json(res1b.rounds) == _stable_json(res1.rounds)

    # A different seed must actually change the result (the test can fail).
    _, res3 = _run(FedAvg(), seed=4, model_fn=_dropout_model_fn)
    assert _stable_json(res3.rounds) != _stable_json(res1.rounds)


def test_seed_tree_unique_and_stable() -> None:
    seeds = [_client_torch_seed(0, r, c) for r in range(3) for c in range(3)]
    assert len(set(seeds)) == 9
    assert _client_torch_seed(0, 1, 2) == _client_torch_seed(0, 1, 2)
    assert _client_torch_seed(1, 1, 2) != _client_torch_seed(0, 1, 2)


# --------------------------------------------------------------------------- #
# Engine: learning happens
# --------------------------------------------------------------------------- #


def test_two_round_federation_improves_val_loss() -> None:
    _, res = _run(FedAvg(), seed=0, rounds=2, lr=0.2)
    losses = [r["global"]["loss"] for r in res.rounds]
    assert losses[1] < losses[0]
    assert res.final["acc"] > 1.0 / N_CLASSES  # better than chance on global test


# --------------------------------------------------------------------------- #
# Engine: LDP/attack ordering (threat model) and validator plumbing
# --------------------------------------------------------------------------- #


class _RecordingValidator:
    def __init__(self) -> None:
        self.updates: list[list[ClientUpdate]] = []
        self.refs: list[list[np.ndarray]] = []

    def __call__(self, updates: list[ClientUpdate], ref: list[np.ndarray]) -> ValidationReport:
        self.updates.append(
            [
                ClientUpdate(u.cid, u.n_samples, [a.copy() for a in u.params], u.train_loss,
                             u.attack)
                for u in updates
            ]
        )
        self.refs.append([a.copy() for a in ref])
        return _accept_all(updates, ref)


class _RecordingAttack:
    """Replaces the update with a constant far outside any clip ball."""

    def __init__(self) -> None:
        self.received: list[list[np.ndarray]] = []
        self.ctx_rounds: list[int] = []
        self.ctx_refs: list[list[np.ndarray]] = []
        self.rngs_ok: list[bool] = []

    def poison(self, update: ClientUpdate, ctx) -> ClientUpdate:
        self.received.append([a.copy() for a in update.params])
        self.ctx_rounds.append(ctx.round)
        self.ctx_refs.append([a.copy() for a in ctx.ref])
        self.rngs_ok.append(isinstance(ctx.rng, np.random.Generator))
        update.params = [np.full_like(a, 1000.0) for a in update.params]
        return update


def test_ldp_then_attack_ordering() -> None:
    clip = 0.05
    atk = _RecordingAttack()
    val = _RecordingValidator()
    fed = _make_fed(k=2, n=40, seed=2)
    eng = FederatedEngine(
        fed, _model_fn, FedAvg(), rounds=1, local_epochs=1, batch_size=8,
        lr=2.0,  # huge steps so the raw update norm far exceeds the clip
        ldp=LDPConfig(clip=clip, sigma=0.0),
        attacks={1: atk}, validator=val, device="cpu", seed=11,
    )
    eng.run()

    ref = val.refs[0]
    submitted = {u.cid: u for u in val.updates[0]}

    # Honest client: LDP clipped its submission onto/inside the clip ball.
    assert _l2(submitted[0].params, ref) <= clip * (1 + 1e-5)

    # Attacked client: the attack received the LDP-SANITISED update (clipping
    # had engaged: norm sits on the sphere, not at the raw lr=2.0 magnitude)...
    received_norm = _l2(atk.received[0], ref)
    assert clip * 0.99 <= received_norm <= clip * (1 + 1e-5)
    # ...but its SUBMITTED update is the poison, verbatim: nothing re-clips it.
    assert all(np.all(a == 1000.0) for a in submitted[1].params)
    assert _l2(submitted[1].params, ref) > clip * 100
    assert submitted[1].attack == "_RecordingAttack"

    # AttackContext plumbing: 0-based round, round-start global as ref, np rng.
    assert atk.ctx_rounds == [0]
    assert all(atk.rngs_ok)
    for a, b in zip(atk.ctx_refs[0], ref, strict=True):
        np.testing.assert_array_equal(a, b)


def test_validator_filtering_and_consensus_mapping() -> None:
    from fedgrid.consensus.pbft import ConsensusResult

    class _RejectCid1:
        def __call__(self, updates, ref):
            accepted = [u.cid for u in updates if u.cid != 1]
            return ValidationReport(
                accepted=accepted,
                scores=np.arange(len(updates), dtype=np.float64),
                consensus=ConsensusResult(
                    committed=True, accepted=accepted, n_messages=42,
                    sim_latency_ms=7.5, rejected=[1],
                ),
            )

    atk = _RecordingAttack()  # poisons cid 1 with 1000.0 everywhere
    eng, res = _run(FedAvg(), seed=6, rounds=1, attacks={1: atk}, validator=_RejectCid1())

    # The poisoned update was excluded from aggregation: global stays sane.
    assert max(float(np.abs(p).max()) for p in eng.global_params) < 100.0

    ev = res.rounds[0]
    assert ev["clients"][1]["accepted"] is False
    assert ev["clients"][1]["attack"] == "_RecordingAttack"
    assert ev["clients"][1]["fidelity"] == 1.0  # scores[1] from arange
    assert ev["clients"][0]["accepted"] is True
    assert ev["consensus"] == {"n_messages": 42, "latency_ms": 7.5, "rejected": [1]}


def test_robust_validator_aggregate_bypasses_strategy() -> None:
    """Handshake with consensus/pipeline.py: a report carrying ``aggregate``
    replaces the round's global params verbatim — the Strategy never runs."""

    @dataclasses.dataclass
    class _RobustReport(ValidationReport):
        aggregate: list[np.ndarray] | None = None

    class _RobustValidator:
        def __call__(self, updates, ref):
            return _RobustReport(
                accepted=[updates[0].cid],
                scores=np.zeros(len(updates)),
                consensus=None,
                aggregate=[np.full_like(a, 0.123) for a in ref],
            )

    class _MustNotAggregate(FedAvg):
        def aggregate(self, updates, current):
            raise AssertionError("Strategy.aggregate must be bypassed")

    eng, res = _run(_MustNotAggregate(), seed=9, rounds=1, validator=_RobustValidator())
    for p in eng.global_params:
        assert np.all(p == np.float32(0.123))
    # Per-client accepted flags still reflect the report, not the bypass.
    flags = [c["accepted"] for c in res.rounds[0]["clients"]]
    assert flags == [True, False, False]


def test_fail_closed_validator_keeps_global_unchanged() -> None:
    class _RejectAll:
        def __call__(self, updates, ref):
            return ValidationReport(accepted=[], scores=np.zeros(len(updates)), consensus=None)

    events: list[dict] = []
    fed = _make_fed(k=2, n=40, seed=5)
    eng = FederatedEngine(
        fed, _model_fn, FedAvg(), rounds=1, local_epochs=1, batch_size=8, lr=0.1,
        validator=_RejectAll(), device="cpu", seed=12, on_round=events.append,
    )
    torch.manual_seed(12)
    init = [p.detach().numpy().copy() for p in _model_fn().parameters()]
    eng.run()
    for a, b in zip(eng.global_params, init, strict=True):
        np.testing.assert_array_equal(a, b)  # nothing admitted -> global untouched
    assert events[0]["consensus"]["rejected"] == [0, 1]


# --------------------------------------------------------------------------- #
# Engine: data-level (label-flip) attacks
# --------------------------------------------------------------------------- #


def test_flip_labels_helper() -> None:
    y = torch.arange(10, dtype=torch.int64) % N_CLASSES
    rng = np.random.default_rng(0)
    assert torch.equal(_flip_labels(y, 0.0, N_CLASSES, rng), y)
    full = _flip_labels(y, 1.0, N_CLASSES, rng)
    assert torch.all(full != y)  # offset in [1, n) can never map to itself
    assert full.min() >= 0 and full.max() < N_CLASSES
    half = _flip_labels(y, 0.5, N_CLASSES, rng)
    assert int((half != y).sum()) == 5  # exactly round(frac * n) flips


class _FlipAll:
    """Data-level attack per contract LabelFlip shape: frac + seed, no poison()."""

    frac = 1.0
    seed = 9


def test_data_level_attack_consumed_pre_train() -> None:
    # Enough local optimisation that the label signal dominates init noise:
    # clean training reaches ~0.24 CE while 100%-flipped labels stay ~0.92.
    kw = dict(seed=8, rounds=2, local_epochs=3, lr=0.3)
    _, clean = _run(FedAvg(), **kw)
    _, flipped = _run(FedAvg(), attacks={0: _FlipAll()}, **kw)
    assert flipped.rounds[0]["clients"][0]["attack"] == "_FlipAll"
    assert flipped.rounds[0]["clients"][1]["attack"] is None
    # Training against wrong labels can't fit: a wide, structural loss gap...
    assert (
        flipped.rounds[-1]["clients"][0]["loss"]
        > clean.rounds[-1]["clients"][0]["loss"] + 0.3
    )
    # ...and the poisoned federation ends with worse global accuracy.
    assert flipped.rounds[-1]["global"]["acc"] < clean.rounds[-1]["global"]["acc"]


# --------------------------------------------------------------------------- #
# Engine: events match the API schema verbatim; RunResult is well-formed
# --------------------------------------------------------------------------- #


def test_round_events_validate_against_api_schema() -> None:
    from fedgrid.api.schemas import RoundEvent

    events: list[dict] = []
    fed = _make_fed(k=2, n=40, seed=4)
    eng = FederatedEngine(
        fed, _model_fn, FedAvg(), rounds=2, local_epochs=1, batch_size=8, lr=0.1,
        validator=_RecordingValidator(), device="cpu", seed=1, on_round=events.append,
    )
    res = eng.run()
    assert len(events) == 2
    assert events == res.rounds
    for i, ev in enumerate(events):
        parsed = RoundEvent.model_validate(ev)  # streams over /ws/live verbatim
        assert parsed.round == i + 1
        assert parsed.run_id == "fedavg-seed1"
        assert len(parsed.clients) == 2
    # Everything the engine stores must be JSON-native already.
    json.dumps(res.rounds)


def test_runresult_shape_and_serialisability() -> None:
    from fedgrid.metrics.runresult import RunResult

    _, res = _run(
        FedProx(mu=0.01), seed=2, rounds=2,
        ldp=LDPConfig(clip=1.0, sigma=0.1, seed=3),
    )
    assert isinstance(res, RunResult)
    assert res.method == "fedprox"
    assert res.seed == 2
    assert len(res.rounds) == 2
    for key in ("acc", "macro_f1", "per_class_f1", "confusion", "loss"):
        assert key in res.final
    conf = np.asarray(res.final["confusion"])
    assert conf.shape == (N_CLASSES, N_CLASSES)
    assert res.final["split"] == "global_test"
    assert res.config["ldp"] == {"clip": 1.0, "sigma": 0.1, "seed": 3}
    assert res.config["mu"] == 0.01
    json.dumps(dataclasses.asdict(res))
    # Amendment 2026-07-28d §4 fields, when the models/metrics module has added them, are strings;
    # and when the models/metrics module's helper is implemented, the fingerprint really flows in.
    import fedgrid.metrics.runresult as rrmod

    for extra in ("git_commit", "env_fingerprint"):
        if hasattr(res, extra):
            assert isinstance(getattr(res, extra), str)
    if hasattr(res, "env_fingerprint") and callable(
        getattr(rrmod, "current_env_fingerprint", None)
    ):
        assert res.env_fingerprint != "unknown"


def test_engine_rejects_bad_config() -> None:
    fed = _make_fed(k=1, n=20, seed=0)
    with pytest.raises(ValueError):
        FederatedEngine(fed, _model_fn, FedAvg(), rounds=0, device="cpu")
    with pytest.raises(ValueError):
        FederatedEngine(fed, _model_fn, FedAvg(), rounds=1, lr=0.0, device="cpu")
    with pytest.raises(ValueError):
        FederatedEngine(fed, _model_fn, FedAvg(), rounds=1, local_epochs=0, device="cpu")
    with pytest.raises(ValueError):
        FederatedEngine(fed, _model_fn, FedAvg(), rounds=1, batch_size=0, device="cpu")
    with pytest.raises(ValueError):
        FederatedEngine(_Fed([]), _model_fn, FedAvg(), rounds=1, device="cpu")
    # A federation with zero validation samples cannot be evaluated per round.
    x, y = _blobs(20, 0)
    empty_val = _Client(x, y)
    empty_val.val = TensorDataset(x[:0], y[:0])
    with pytest.raises(ValueError):
        FederatedEngine(_Fed([empty_val]), _model_fn, FedAvg(), rounds=1, device="cpu").run()


def test_attack_tag_restored_when_poison_drops_it() -> None:
    class _ReplacingAttack:
        def poison(self, update: ClientUpdate, ctx) -> ClientUpdate:
            # Fresh update object with no tag: the engine must restore the
            # ground-truth attack label for evaluation.
            return ClientUpdate(
                cid=update.cid,
                n_samples=update.n_samples,
                params=[-a for a in update.params],
                train_loss=update.train_loss,
                attack=None,
            )

    _, res = _run(FedAvg(), seed=10, rounds=1, attacks={2: _ReplacingAttack()})
    assert res.rounds[0]["clients"][2]["attack"] == "_ReplacingAttack"


# --------------------------------------------------------------------------- #
# Engine: energy plumbing (SynOps hooks) and provenance fallbacks
# --------------------------------------------------------------------------- #


class _FakeSpikingNet(nn.Module):
    """SpikingClassifier-shaped fake: synops()/reset_counters() per contract."""

    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(N_FEATURES, 8), nn.Tanh(), nn.Linear(8, N_CLASSES))
        self.resets = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)

    def synops(self) -> dict[str, float]:
        return {"synops": 123.0, "dense_macs": 999.0, "spike_rate": 0.1, "ratio": 0.25}

    def reset_counters(self) -> None:
        self.resets += 1


class _StubSynNet(_FakeSpikingNet):
    """the models/metrics module stub shape: hooks exist but raise NotImplementedError."""

    def synops(self) -> dict[str, float]:
        raise NotImplementedError

    def reset_counters(self) -> None:
        raise NotImplementedError


def test_round_energy_reads_model_synops() -> None:
    eng, res = _run(FedAvg(), seed=1, rounds=1, model_fn=_FakeSpikingNet)
    assert res.rounds[0]["energy"] == {"synops": 123.0, "ratio": 0.25}
    # Counters are reset before the dry shape probe and before each eval, so
    # the reported SynOps belong to this round only.
    assert eng._global_model.resets >= 2

    _, stub = _run(FedAvg(), seed=1, rounds=1, model_fn=_StubSynNet)
    assert stub.rounds[0]["energy"] == {"synops": 0.0, "ratio": 0.0}


def test_provenance_helper_fallbacks() -> None:
    from fedgrid.fl.engine import _provenance

    class _FakeRR:
        @staticmethod
        def current_env_fingerprint() -> str:
            raise NotImplementedError

        @staticmethod
        def current_git_commit() -> str:  # no repo_root parameter -> TypeError path
            return "abc123"

    class _EmptyRR:
        pass

    assert _provenance(_FakeRR, "env_fingerprint") == "unknown"
    assert _provenance(_FakeRR, "git_commit") == "abc123"
    assert _provenance(_EmptyRR, "git_commit") == "unknown"


# --------------------------------------------------------------------------- #
# Metrics helpers (numpy-only, no sklearn)
# --------------------------------------------------------------------------- #


def test_confusion_and_f1_hand_computed() -> None:
    y_true = np.array([0, 0, 1, 2, 1, 0])
    y_pred = np.array([0, 1, 1, 2, 0, 0])
    conf = _confusion(y_true, y_pred, 3)
    np.testing.assert_array_equal(conf, [[2, 1, 0], [1, 1, 0], [0, 0, 1]])
    f1, macro = _f1_from_confusion(conf)
    np.testing.assert_allclose(f1, [2 / 3, 1 / 2, 1.0])
    assert math.isclose(macro, (2 / 3 + 1 / 2 + 1.0) / 3)
    with pytest.raises(ValueError):
        _confusion(np.array([0, 5]), np.array([0, 1]), 3)
    with pytest.raises(ValueError):
        _confusion(np.array([0]), np.array([0, 1]), 3)


def test_macro_f1_averages_over_present_classes_only() -> None:
    # Class 2 absent from y_true: macro averages classes 0 and 1 only.
    conf = _confusion(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 0]), 3)
    f1, macro = _f1_from_confusion(conf)
    assert f1[2] == 0.0
    assert math.isclose(macro, (4 / 5 + 2 / 3) / 2)
