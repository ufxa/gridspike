"""Correctness bar for `consensus/` and `attacks/` (phase 2).

Covers the contract's bar plus the Hypothesis properties assigned to this module
by `docs/VALIDATION_STANDARD.md` §3.8: fidelity in [0, 2] for every update set,
PBFT never commits above its fault bound for every sampled topology, and the
ledger detects every single-field tamper. The ledger's one documented blind
spot -- tail truncation, which a bare hash chain cannot see -- is pinned by its
own test rather than left implicit.

The attack tests deliberately run a *two-round* scene: `AdaptiveCosine` has no
public aggregate direction to aim at in round 0 and submits an honest update
then, so a single-round evaluation would measure nothing (see its docstring).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from fedgrid.attacks.base import AttackContext
from fedgrid.attacks.suite import AdaptiveCosine, GaussianNoise, LabelFlip, SignFlip
from fedgrid.consensus.aggregation import geometric_median, krum, krum_scores, multi_krum
from fedgrid.consensus.fidelity import fidelity_scores, flatten, unflatten
from fedgrid.consensus.ledger import GENESIS_HASH, Ledger
from fedgrid.consensus.pbft import ConsensusResult, PBFTSim, default_net_ms
from fedgrid.consensus.pipeline import ConsensusValidator, make_validator
from fedgrid.fl.engine import ClientUpdate

SHAPES = [(20, 10), (10,)]
K = 10
N_ADV = 3
TAU = 0.15  # admits honest non-IID clients (~0.09 here); the battery sweeps it


# --------------------------------------------------------------------------- #
# scene builders (helpers live in this file only -- house rule: no utils.py)
# --------------------------------------------------------------------------- #
def make_update(cid: int, params: list[np.ndarray], n_samples: int = 100) -> ClientUpdate:
    return ClientUpdate(cid=cid, n_samples=n_samples, params=params, train_loss=0.4)


def gaussian_cloud(
    n: int, d: int, seed: int = 0, spread: float = 0.1, centre: np.ndarray | None = None
) -> list[ClientUpdate]:
    """`n` updates scattered around one direction -- the honest-majority stand-in."""
    rng = np.random.default_rng(seed)
    if centre is None:
        centre = rng.normal(size=d)
    return [
        make_update(i, [centre + spread * rng.normal(size=d)])
        for i in range(n)
    ]


def federated_scene(
    seed: int,
    attack: object | None,
    rounds: int = 2,
    n_adv: int = N_ADV,
    k: int = K,
    spread: float = 0.4,
    step: float = 0.05,
) -> tuple[list[ClientUpdate], list[np.ndarray], list[ClientUpdate]]:
    """Run `rounds` honest aggregation rounds, poisoning the last `n_adv` clients.

    Returns (updates of the final round with the attack applied, that round's
    global parameters, the honest subset of the same round). The honest subset is
    the *counterfactual a perfect defense would aggregate*, which is what the
    damage measurements compare against -- comparing against "all ten clients
    clean" would charge the defense for correctly dropping the attackers.
    """
    rng = np.random.default_rng(seed)
    ref = [rng.normal(size=s).astype(np.float32) for s in SHAPES]
    direction = [rng.normal(size=s).astype(np.float32) for s in SHAPES]
    mixed: list[ClientUpdate] = []
    honest: list[ClientUpdate] = []
    for rnd in range(rounds):
        crng = np.random.default_rng(4_000 + 13 * seed + rnd)
        updates = [
            make_update(
                i,
                [
                    (b + step * d + step * spread * crng.normal(size=b.shape)).astype(np.float32)
                    for b, d in zip(ref, direction, strict=True)
                ],
            )
            for i in range(k)
        ]
        ctx = AttackContext(round=rnd, ref=[r.copy() for r in ref], rng=np.random.default_rng(5))
        poisoned = updates[: k - n_adv] + (
            [attack.poison(u, ctx) for u in updates[k - n_adv :]]  # type: ignore[attr-defined]
            if attack is not None
            else updates[k - n_adv :]
        )
        mixed, honest = poisoned, updates[: k - n_adv]
        if rnd < rounds - 1:
            # Honest server step, so the attacker can observe the public direction.
            ref = [
                np.mean([u.params[layer] for u in updates], axis=0) for layer in range(len(ref))
            ]
    return mixed, ref, honest


def constant_net(ms: float):
    def _net(_rng: np.random.Generator) -> float:
        return ms

    return _net


class CountingNet:
    """`net_ms` wrapper that records how many messages the simulator actually sent."""

    def __init__(self, ms: float = 1.0) -> None:
        self.ms = ms
        self.calls = 0

    def __call__(self, _rng: np.random.Generator) -> float:
        self.calls += 1
        return self.ms


# --------------------------------------------------------------------------- #
# fidelity
# --------------------------------------------------------------------------- #
def test_flatten_and_unflatten_round_trip() -> None:
    params = [np.arange(6, dtype=np.float32).reshape(2, 3), np.array([9.0], dtype=np.float32)]
    flat = flatten(params)
    assert flat.tolist() == [0, 1, 2, 3, 4, 5, 9]
    back = unflatten(flat, params)
    assert [b.shape for b in back] == [(2, 3), (1,)]
    assert all(b.dtype == np.float32 for b in back)
    assert all(np.array_equal(a, b) for a, b in zip(params, back, strict=True))


def test_flatten_handles_empty_parameter_list() -> None:
    assert flatten([]).shape == (0,)


def test_fidelity_flags_planted_sign_flip_outlier() -> None:
    """The contract's headline fidelity bar: the planted outlier gets the max score."""
    updates = gaussian_cloud(9, d=64, seed=1, spread=0.1)
    centre = updates[0].params[0]
    updates.append(make_update(9, [-3.0 * centre]))

    scores = fidelity_scores(updates)
    assert int(np.argmax(scores)) == 9
    assert scores[9] > 1.5, "a reversed update must land in the obtuse half"
    assert scores[:9].max() < 0.5, "honest scatter must stay near zero"
    assert scores[9] > 10 * scores[:9].max()


def test_fidelity_is_bounded_and_never_nan_with_degenerate_updates() -> None:
    updates = gaussian_cloud(4, d=16, seed=2)
    updates.append(make_update(4, [np.zeros(16)]))
    scores = fidelity_scores(updates)
    assert np.all(np.isfinite(scores))
    assert scores[4] == pytest.approx(1.0), "a zero update has no direction: neutral score"
    assert np.all((scores >= 0.0) & (scores <= 2.0))


def test_fidelity_zero_median_is_neutral_not_nan() -> None:
    """An even federation that cancels coordinate-wise has a zero median."""
    v = np.array([1.0, -2.0, 0.5])
    updates = [make_update(0, [v]), make_update(1, [-v])]
    assert np.allclose(fidelity_scores(updates), 1.0)


def test_fidelity_ref_relative_form_separates_what_the_raw_form_cannot() -> None:
    """Full parameters hide the update; deltas expose it (see fidelity_scores docs)."""
    rng = np.random.default_rng(3)
    ref = [rng.normal(size=200) * 10.0]  # a trained model: large shared component
    honest = [make_update(i, [ref[0] + rng.normal(size=200) * 0.01 + 0.05]) for i in range(5)]
    attacker = make_update(5, [ref[0] - 0.05])  # exactly reversed contribution

    raw = fidelity_scores(honest + [attacker])
    rel = fidelity_scores(honest + [attacker], ref=ref)
    assert raw.max() < 1e-3, "raw full-parameter cosines collapse into the noise floor"
    assert rel[5] > 1.5 and rel[:5].max() < 0.5
    assert int(np.argmax(rel)) == 5


def test_fidelity_rejects_mismatched_parameter_shapes() -> None:
    updates = [make_update(0, [np.zeros(4)]), make_update(1, [np.zeros(5)])]
    with pytest.raises(ValueError, match="client 1"):
        fidelity_scores(updates)


def test_fidelity_of_empty_update_list_is_empty() -> None:
    assert fidelity_scores([]).shape == (0,)


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def byzantine_scene(
    seed: int = 0, n_honest: int = 7, n_byz: int = 3, d: int = 64
) -> tuple[list[ClientUpdate], np.ndarray]:
    rng = np.random.default_rng(seed)
    honest_mean = rng.normal(size=d)
    updates = [make_update(i, [honest_mean + 0.05 * rng.normal(size=d)]) for i in range(n_honest)]
    for j in range(n_byz):
        updates.append(make_update(n_honest + j, [-8.0 * honest_mean + rng.normal(size=d)]))
    return updates, honest_mean


def test_krum_selects_an_honest_update_under_f_byzantine() -> None:
    updates, honest_mean = byzantine_scene(seed=4)
    idx = krum(updates, f=3)
    assert idx < 7, "Krum must not select from the Byzantine tail"
    selected = updates[idx].params[0]
    cos = selected @ honest_mean / (np.linalg.norm(selected) * np.linalg.norm(honest_mean))
    assert cos > 0.95


def test_krum_scores_rank_byzantine_updates_last() -> None:
    updates, _ = byzantine_scene(seed=5)
    scores = krum_scores(updates, f=3)
    assert scores[:7].max() < scores[7:].min()


def test_multi_krum_returns_m_honest_indices() -> None:
    updates, _ = byzantine_scene(seed=6)
    picks = multi_krum(updates, f=3, m=7)
    assert len(picks) == 7
    assert all(i < 7 for i in picks)


def test_multi_krum_clamps_m_to_the_available_updates() -> None:
    updates, _ = byzantine_scene(seed=6)
    assert len(multi_krum(updates, f=3, m=99)) == len(updates)
    assert len(multi_krum(updates, f=3, m=0)) == 1


def test_krum_is_permutation_invariant() -> None:
    updates, _ = byzantine_scene(seed=7)
    chosen = updates[krum(updates, f=3)].cid
    order = np.random.default_rng(0).permutation(len(updates))
    shuffled = [updates[i] for i in order]
    assert shuffled[krum(shuffled, f=3)].cid == chosen


def test_geometric_median_recovers_the_honest_direction_under_f_byzantine() -> None:
    updates, honest_mean = byzantine_scene(seed=8)
    gm = flatten(geometric_median(updates))
    naive = flatten([np.mean([u.params[0] for u in updates], axis=0)])
    cos = gm @ honest_mean / (np.linalg.norm(gm) * np.linalg.norm(honest_mean))
    assert cos > 0.95
    assert np.linalg.norm(gm - honest_mean) < 0.1 * np.linalg.norm(naive - honest_mean)


def test_geometric_median_hits_the_exact_optimum_on_a_repeated_point() -> None:
    """A strict majority of identical points *is* the median -- the degenerate case
    the plain Weiszfeld step divides by zero on."""
    p = np.array([1.0, 2.0, 3.0])
    updates = [make_update(i, [p.copy()]) for i in range(3)]
    updates.append(make_update(3, [np.array([50.0, -60.0, 70.0])]))
    gm = geometric_median(updates)
    assert np.allclose(gm[0], p, atol=1e-9)


def test_geometric_median_preserves_shapes_and_float_dtype() -> None:
    updates = [
        make_update(i, [np.full((2, 3), float(i), dtype=np.float32), np.zeros(4, dtype=np.float32)])
        for i in range(5)
    ]
    gm = geometric_median(updates)
    assert [g.shape for g in gm] == [(2, 3), (4,)]
    assert all(g.dtype == np.float32 for g in gm)


def test_geometric_median_of_a_single_update_is_that_update() -> None:
    p = [np.array([3.0, -1.0])]
    assert np.allclose(geometric_median([make_update(0, p)])[0], p[0])


# --------------------------------------------------------------------------- #
# PBFT
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("m", [4, 7, 10, 13, 16])
def test_pbft_message_count_matches_the_documented_formula(m: int) -> None:
    """n_messages == (M-1) + (M-1)^2 + S(M-1); S = M on commit, S = f on stall.

    The counter is the `net_ms` callable itself: the simulator draws exactly one
    latency per message, so an off-by-one in the formula and an off-by-one in the
    simulation cannot cancel each other out.
    """
    scores = np.zeros(5)
    for f in range(0, m):
        net = CountingNet()
        sim = PBFTSim(m, f, net, seed=f)
        res = sim.run_round(scores, threshold=0.5)
        senders = m if res.committed else f
        expected = (m - 1) + (m - 1) ** 2 + senders * (m - 1)
        assert res.n_messages == expected, f"m={m} f={f}"
        assert net.calls == res.n_messages, "one RNG draw per message, exactly"
        if res.committed:
            assert res.n_messages == 2 * m * (m - 1)


def test_pbft_safety_over_200_seeded_trials() -> None:
    """Commits with <= f faulty and *never* with more, over 200 seeded topologies."""
    trials = 0
    committed_below, committed_above = 0, 0
    for m in (4, 7, 10, 13):
        for f in range(0, m + 1):
            for seed in range(6):  # 38 (m, f) topologies x 6 seeds = 228 trials
                trials += 1
                sim = PBFTSim(m, f, default_net_ms, seed=seed)
                res = sim.run_round(np.array([0.0, 0.1, 9.0]), threshold=0.5)
                within = f <= (m - 1) // 3
                assert res.committed is within, f"m={m} f={f} seed={seed}"
                if within:
                    committed_below += 1
                    assert res.accepted == [0, 1] and res.rejected == [2]
                else:
                    committed_above += 1
                    assert res.accepted == [] and res.rejected == [0, 1, 2]
    assert trials >= 200
    assert committed_below > 0 and committed_above > 0


def test_pbft_critical_path_is_exactly_three_hops() -> None:
    """With a constant 1 ms link the round costs pre-prepare + prepare + commit."""
    sim = PBFTSim(10, 3, constant_net(1.0), seed=0)
    res = sim.run_round(np.zeros(4), threshold=1.0)
    assert res.committed
    assert res.sim_latency_ms == pytest.approx(3.0)


def test_pbft_without_byzantine_tolerance_costs_one_hop() -> None:
    """f = 0 asks for no tolerance: the decision is the pre-prepare broadcast."""
    sim = PBFTSim(6, 0, constant_net(1.0), seed=0)
    res = sim.run_round(np.zeros(3), threshold=1.0)
    assert res.committed and res.sim_latency_ms == pytest.approx(1.0)


def test_pbft_stalled_round_still_reports_the_time_it_burnt() -> None:
    sim = PBFTSim(4, 3, constant_net(2.0), seed=0)
    res = sim.run_round(np.zeros(3), threshold=1.0)
    assert not res.committed
    assert res.sim_latency_ms > 0.0


def test_pbft_latency_grows_with_the_network() -> None:
    fast = PBFTSim(10, 3, constant_net(1.0), seed=0).run_round(np.zeros(3), 1.0)
    slow = PBFTSim(10, 3, constant_net(20.0), seed=0).run_round(np.zeros(3), 1.0)
    assert slow.sim_latency_ms > fast.sim_latency_ms


def test_pbft_partition_follows_the_threshold() -> None:
    scores = np.array([0.01, 0.2, 0.05, 1.9])
    res = PBFTSim(10, 3, default_net_ms, seed=0).run_round(scores, threshold=0.1)
    assert res.accepted == [0, 2] and res.rejected == [1, 3]


def test_pbft_is_reproducible_from_its_seed() -> None:
    a = PBFTSim(10, 3, default_net_ms, seed=42).run_round(np.zeros(3), 1.0)
    b = PBFTSim(10, 3, default_net_ms, seed=42).run_round(np.zeros(3), 1.0)
    assert a == b
    c = PBFTSim(10, 3, default_net_ms, seed=43).run_round(np.zeros(3), 1.0)
    assert c.sim_latency_ms != a.sim_latency_ms


def test_pbft_primary_is_honest_and_faulty_set_is_the_tail() -> None:
    sim = PBFTSim(10, 3, default_net_ms, seed=0)
    assert sim.faulty == frozenset({7, 8, 9})
    assert 0 in sim.honest and sim.tolerates


@pytest.mark.parametrize("m,f", [(0, 0), (4, 5), (4, -1)])
def test_pbft_rejects_impossible_topologies(m: int, f: int) -> None:
    with pytest.raises(ValueError):
        PBFTSim(m, f, default_net_ms)


def test_pbft_rejects_a_negative_latency_model() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        PBFTSim(4, 1, constant_net(-1.0), seed=0).run_round(np.zeros(2), 1.0)


# --------------------------------------------------------------------------- #
# ledger
# --------------------------------------------------------------------------- #
def filled_ledger(n: int = 4) -> Ledger:
    led = Ledger()
    for i in range(n):
        led.append({"round": i, "accepted": [0, 1, 2], "rejected": [9], "latency_ms": 1.5 * i})
    return led


def test_ledger_chains_and_verifies() -> None:
    led = filled_ledger()
    blocks = led.to_list()
    assert led.verify()
    assert len(blocks) == 4
    assert blocks[0]["prev_hash"] == GENESIS_HASH
    assert [b["index"] for b in blocks] == [0, 1, 2, 3]
    assert all(blocks[i]["prev_hash"] == blocks[i - 1]["hash"] for i in range(1, 4))
    assert led.head == blocks[-1]["hash"]


def test_ledger_hashes_numpy_scalars_and_key_order_identically() -> None:
    a, b = Ledger(), Ledger()
    a.append({"score": np.float32(0.25), "cid": np.int64(3)})
    b.append({"cid": 3, "score": 0.25})
    assert a.head == b.head, "canonical JSON must not depend on dtype or key order"


def test_ledger_rejects_unserialisable_and_nan_records() -> None:
    with pytest.raises(TypeError):
        Ledger().append({"model": object()})
    with pytest.raises(ValueError):
        Ledger().append({"score": float("nan")})


@pytest.mark.parametrize("field", ["round", "accepted", "rejected", "latency_ms"])
def test_ledger_verify_fails_after_tampering_with_any_record_field(field: str) -> None:
    blocks = filled_ledger().to_list()
    blocks[2]["record"][field] = "tampered"
    assert Ledger.from_list(blocks).verify() is False


@pytest.mark.parametrize("field", ["index", "prev_hash", "hash"])
def test_ledger_verify_fails_after_tampering_with_the_chain_metadata(field: str) -> None:
    blocks = filled_ledger().to_list()
    blocks[2][field] = 99 if field == "index" else "0" * 64
    assert Ledger.from_list(blocks).verify() is False


def test_ledger_verify_fails_when_a_block_is_dropped_or_reordered() -> None:
    blocks = filled_ledger().to_list()
    assert Ledger.from_list(blocks[:2] + blocks[3:]).verify() is False
    assert Ledger.from_list([blocks[1], blocks[0]] + blocks[2:]).verify() is False


def test_ledger_verify_does_NOT_detect_tail_truncation() -> None:
    """The documented limit of a bare hash chain, pinned so it cannot be overclaimed.

    Dropping the last k blocks leaves a valid shorter chain: every link, digest and
    index still recomputes. Nothing stored commits to the chain's length, so
    ``verify()`` cannot tell a truncated chain from a chain that is simply shorter.
    """
    blocks = filled_ledger().to_list()
    assert Ledger.from_list(blocks[:-1]).verify() is True
    assert Ledger.from_list(blocks[:2]).verify() is True
    assert Ledger.from_list([]).verify() is True

    # truncate-then-rewrite: the discarded rounds are replaced by different ones
    forked = Ledger.from_list(blocks[:2])
    forked.append({"round": 2, "accepted": [0], "rejected": [1, 2, 9], "latency_ms": 0.0})
    assert forked.verify() is True
    assert forked.head != blocks[2]["hash"]


def test_ledger_verify_expected_len_and_head_reject_a_truncated_tail() -> None:
    """The optional out-of-band commitment is what makes truncation detectable."""
    led = filled_ledger()
    n, head = len(led), led.head
    assert led.verify(expected_len=n, expected_head=head) is True

    truncated = Ledger.from_list(led.to_list()[:-1])
    assert truncated.verify() is True, "unchanged default behaviour: not caught"
    assert truncated.verify(expected_len=n) is False
    assert truncated.verify(expected_head=head) is False

    # the optional checks do not weaken the structural ones
    tampered = led.to_list()
    tampered[1]["record"]["round"] = 999
    assert Ledger.from_list(tampered).verify(expected_len=n) is False


def test_ledger_rehashing_a_tampered_block_still_breaks_the_next_link() -> None:
    """The point of a chain: fixing a block's own digest does not fix its successor."""
    led = filled_ledger()
    blocks = led.to_list()
    from fedgrid.consensus.ledger import _block_hash, _canonical

    blocks[1]["record"]["round"] = 999
    blocks[1]["hash"] = _block_hash(blocks[1]["prev_hash"], _canonical(blocks[1]["record"]))
    assert Ledger.from_list(blocks).verify() is False


def test_ledger_to_list_hands_out_copies() -> None:
    led = filled_ledger()
    exported = led.to_list()
    exported[0]["record"]["round"] = "tampered"
    assert led.verify(), "mutating an exported block must not reach the live chain"


def test_empty_ledger_verifies_and_heads_at_genesis() -> None:
    led = Ledger()
    assert led.verify() and led.head == GENESIS_HASH and led.to_list() == []


# --------------------------------------------------------------------------- #
# attacks
# --------------------------------------------------------------------------- #
def test_sign_flip_reverses_the_delta_not_the_model() -> None:
    ref = [np.zeros(8) + 5.0]
    delta = np.arange(8, dtype=float)
    upd = make_update(0, [ref[0] + delta])
    ctx = AttackContext(round=0, ref=ref, rng=np.random.default_rng(0))
    out = SignFlip(scale=2.0).poison(upd, ctx)
    assert np.allclose(out.params[0], ref[0] - 2.0 * delta)
    assert out.attack == "sign_flip" and out.cid == 0 and out.n_samples == 100


def test_gaussian_noise_matches_its_declared_sigma() -> None:
    ref = [np.zeros(20_000)]
    upd = make_update(0, [np.zeros(20_000)])
    ctx = AttackContext(round=0, ref=ref, rng=np.random.default_rng(0))
    out = GaussianNoise(sigma=0.5).poison(upd, ctx)
    assert out.params[0].std() == pytest.approx(0.5, rel=0.05)
    assert out.attack == "gaussian_noise"


def test_label_flip_flips_the_declared_fraction_to_other_classes() -> None:
    labels = np.zeros(1000, dtype=np.int64)
    flipped = LabelFlip(frac=0.3, seed=1).flip(labels, n_classes=8, cid=0)
    changed = flipped != labels
    assert changed.sum() == 300
    assert set(np.unique(flipped[changed])).issubset(set(range(1, 8)))
    assert 0 not in flipped[changed], "a flip must never land back on the true class"


def test_label_flip_is_deterministic_per_client_and_differs_across_clients() -> None:
    labels = np.arange(200, dtype=np.int64) % 8
    a = LabelFlip(frac=0.25, seed=3)
    assert np.array_equal(a.flip(labels, 8, cid=1), a.flip(labels, 8, cid=1))
    assert not np.array_equal(a.flip(labels, 8, cid=1), a.flip(labels, 8, cid=2))
    assert np.array_equal(labels, np.arange(200, dtype=np.int64) % 8), "input untouched"


def test_label_flip_is_not_an_update_level_attack() -> None:
    """The engine discriminates on `poison`; LabelFlip must not offer one."""
    assert not hasattr(LabelFlip(0.1), "poison")
    with pytest.raises(ValueError):
        LabelFlip(frac=1.5)


def test_adaptive_cosine_meets_its_cone_by_construction() -> None:
    """cos(delta, estimate) == cos(theta_use) exactly -- no rejection sampling."""
    rng = np.random.default_rng(0)
    ref = [rng.normal(size=500)]
    prev = [ref[0] - 0.1 * rng.normal(size=500)]
    atk = AdaptiveCosine(target_score=0.2, safety_margin=0.0, n_peers=1)
    honest = make_update(0, [ref[0] + 0.05 * rng.normal(size=500)])

    atk.poison(honest, AttackContext(round=0, ref=prev, rng=np.random.default_rng(1)))
    out = atk.poison(honest, AttackContext(round=1, ref=ref, rng=np.random.default_rng(1)))

    u = flatten(ref) - flatten(prev)
    u /= np.linalg.norm(u)
    delta = flatten(out.params) - flatten(ref)
    realised = 1.0 - float(delta @ u / np.linalg.norm(delta))
    assert realised <= 0.2 + 1e-9
    assert out.attack == "adaptive_cosine"


def test_adaptive_cosine_is_honest_on_round_zero() -> None:
    """No observed aggregate step yet, so it does not fire blind (see docstring)."""
    rng = np.random.default_rng(0)
    ref = [rng.normal(size=100)]
    upd = make_update(0, [ref[0] + 0.1 * rng.normal(size=100)])
    out = AdaptiveCosine(target_score=0.3).poison(
        upd, AttackContext(round=0, ref=ref, rng=np.random.default_rng(0))
    )
    assert np.allclose(out.params[0], upd.params[0], atol=1e-9)


def test_adaptive_cosine_preserves_the_update_norm_by_default() -> None:
    rng = np.random.default_rng(2)
    ref = [rng.normal(size=300)]
    prev = [ref[0] - 0.2 * rng.normal(size=300)]
    upd = make_update(0, [ref[0] + 0.1 * rng.normal(size=300)])
    atk = AdaptiveCosine(target_score=0.2)
    atk.poison(upd, AttackContext(round=0, ref=prev, rng=np.random.default_rng(0)))
    out = atk.poison(upd, AttackContext(round=1, ref=ref, rng=np.random.default_rng(0)))
    honest_norm = np.linalg.norm(flatten(upd.params) - flatten(ref))
    assert np.linalg.norm(flatten(out.params) - flatten(ref)) == pytest.approx(honest_norm)


def test_adaptive_cosine_leaves_a_converged_client_alone() -> None:
    ref = [np.ones(50)]
    upd = make_update(0, [np.ones(50)])
    out = AdaptiveCosine(target_score=0.5).poison(
        upd, AttackContext(round=3, ref=ref, rng=np.random.default_rng(0))
    )
    assert np.allclose(out.params[0], upd.params[0])


# --------------------------------------------------------------------------- #
# pipeline: the contract's asymmetry bar, against BOTH aggregators
# (amendment 2026-07-28 §3)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("agg", ["krum", "multi_krum", "geomedian"])
def test_adaptive_evades_the_threshold_that_sign_flip_trips(agg: str) -> None:
    target = 0.1
    for seed in range(5):
        for attack, expect_admitted in (
            (SignFlip(1.0), False),
            (AdaptiveCosine(target_score=target, collude_seed=7, n_peers=K), True),
        ):
            mixed, ref, _ = federated_scene(seed, attack)
            report = make_validator(K, N_ADV, TAU, agg, seed=0)(mixed, ref)
            adv_scores = report.scores[K - N_ADV :]
            admitted = [c for c in report.consensus.accepted if c >= K - N_ADV]
            assert report.consensus.committed
            if expect_admitted:
                assert adv_scores.max() <= target, f"{agg} seed={seed}: {adv_scores}"
                assert len(admitted) == N_ADV
            else:
                assert adv_scores.min() > TAU
                assert admitted == []


@pytest.mark.parametrize("agg", ["krum", "geomedian"])
def test_accuracy_damage_asymmetry_against_both_aggregators(agg: str) -> None:
    """The paper's point, measured: the loud attack is neutralised (zero shift),
    the defense-aware one is admitted and still moves the aggregate.

    Damage is `||aggregate(attacked federation) - aggregate(honest subset)||`, so
    the baseline is what a *perfect* defense would have produced; an attack that
    the pipeline fully rejects scores exactly zero by construction.
    """
    seeds = range(20)
    measured: dict[str, list[float]] = {}
    byzantine_selected = 0
    for name, attack_fn in (
        ("sign_flip", lambda: SignFlip(1.0)),
        ("adaptive", lambda: AdaptiveCosine(target_score=0.1, collude_seed=7, n_peers=K)),
    ):
        shifts = []
        for seed in seeds:
            mixed, ref, honest = federated_scene(seed, attack_fn())
            attacked = make_validator(K, N_ADV, TAU, agg, seed=0)(mixed, ref)
            clean = make_validator(K, N_ADV, TAU, agg, seed=0)(honest, ref)
            assert attacked.aggregate is not None and clean.aggregate is not None
            drift = flatten(attacked.aggregate) - flatten(clean.aggregate)
            shifts.append(float(np.linalg.norm(drift)))
            if name == "adaptive":
                byzantine_selected += any(c >= K - N_ADV for c in attacked.accepted)
        measured[name] = shifts

    print(
        f"\n[damage/{agg}] sign_flip mean={np.mean(measured['sign_flip']):.5f} "
        f"adaptive mean={np.mean(measured['adaptive']):.5f} "
        f"adaptive nonzero={sum(s > 0 for s in measured['adaptive'])}/{len(seeds)} "
        f"byzantine-in-aggregate={byzantine_selected}/{len(seeds)}"
    )
    assert max(measured["sign_flip"]) == 0.0, "the loud attack must be fully neutralised"
    assert min(measured["adaptive"]) > 0.0, "the adaptive attack must still shift the aggregate"
    assert np.mean(measured["adaptive"]) > np.mean(measured["sign_flip"])


def test_cosine_defense_is_blind_to_the_magnitude_lever() -> None:
    """Scaling the same admissible direction multiplies the damage and moves no score."""
    scores, shifts = {}, {}
    for scale in (1.0, 3.0):
        s, d = [], []
        for seed in range(10):
            atk = AdaptiveCosine(target_score=0.1, scale=scale, collude_seed=7, n_peers=K)
            mixed, ref, honest = federated_scene(seed, atk)
            attacked = make_validator(K, N_ADV, TAU, "geomedian", seed=0)(mixed, ref)
            clean = make_validator(K, N_ADV, TAU, "geomedian", seed=0)(honest, ref)
            s.append(float(attacked.scores[K - N_ADV :].max()))
            d.append(float(np.linalg.norm(flatten(attacked.aggregate) - flatten(clean.aggregate))))
        scores[scale], shifts[scale] = float(np.mean(s)), float(np.mean(d))
    assert scores[3.0] <= 0.1 and scores[1.0] <= 0.1
    assert abs(scores[3.0] - scores[1.0]) < 0.02, "cosine is scale-free: the score barely moves"
    assert shifts[3.0] > 1.3 * shifts[1.0]


# --------------------------------------------------------------------------- #
# pipeline plumbing
# --------------------------------------------------------------------------- #
def test_validator_report_carries_consensus_stats_for_the_round_event() -> None:
    mixed, ref, _ = federated_scene(0, SignFlip(1.0))
    validator = make_validator(K, N_ADV, TAU, "geomedian", seed=0)
    report = validator(mixed, ref)

    assert isinstance(validator, ConsensusValidator)
    assert isinstance(report.consensus, ConsensusResult)
    assert report.scores.shape == (K,)
    assert report.consensus.n_messages == 2 * K * (K - 1)
    assert report.consensus.sim_latency_ms > 0.0
    assert report.consensus.rejected == [7, 8, 9], "cids, not positions"
    assert report.aggregate is not None
    assert [a.shape for a in report.aggregate] == SHAPES
    assert validator.ledger.verify() and len(validator.ledger) == 1
    block = validator.ledger.to_list()[0]
    assert block["record"]["consensus_rejected"] == [7, 8, 9]
    assert block["record"]["n_messages"] == report.consensus.n_messages
    assert report.ledger_hash == block["hash"]


def test_validator_krum_returns_a_single_client_geomedian_returns_a_new_point() -> None:
    mixed, ref, _ = federated_scene(0, None)
    krum_report = make_validator(K, N_ADV, TAU, "krum", seed=0)(mixed, ref)
    gm_report = make_validator(K, N_ADV, TAU, "geomedian", seed=0)(mixed, ref)
    assert len(krum_report.accepted) == 1
    assert any(np.allclose(flatten(krum_report.aggregate), flatten(u.params)) for u in mixed)
    assert len(gm_report.accepted) == K
    assert not any(np.allclose(flatten(gm_report.aggregate), flatten(u.params)) for u in mixed)


def test_validator_fails_closed_when_consensus_cannot_commit() -> None:
    mixed, ref, _ = federated_scene(0, None)
    report = make_validator(m=4, f=3, threshold=TAU, agg="geomedian", seed=0)(mixed, ref)
    assert report.consensus.committed is False
    assert report.accepted == [] and report.aggregate is None
    assert sorted(report.rejected) == [u.cid for u in mixed]


def test_validator_accumulates_the_ledger_across_rounds() -> None:
    mixed, ref, _ = federated_scene(0, None)
    validator = make_validator(K, N_ADV, TAU, "krum", seed=0)
    heads = [validator(mixed, ref).ledger_hash for _ in range(3)]
    assert len(set(heads)) == 3, "each round is a distinct block"
    assert validator.ledger.verify() and validator.round == 3
    assert [b["record"]["round"] for b in validator.ledger.to_list()] == [0, 1, 2]


def test_validator_is_reproducible_from_its_seed() -> None:
    mixed, ref, _ = federated_scene(1, None)
    a = make_validator(K, N_ADV, TAU, "geomedian", seed=11)(mixed, ref)
    b = make_validator(K, N_ADV, TAU, "geomedian", seed=11)(mixed, ref)
    assert a.ledger_hash == b.ledger_hash
    assert a.consensus == b.consensus


def test_validator_accepts_the_documented_aggregator_aliases() -> None:
    for alias in ("krum", "multi_krum", "multiKrum", "geomedian", "geometric_median", "GM"):
        assert make_validator(K, N_ADV, TAU, alias, seed=0).agg in {
            "krum",
            "multi_krum",
            "geomedian",
        }
    with pytest.raises(ValueError, match="unknown agg"):
        make_validator(K, N_ADV, TAU, "median-of-means")


def test_validator_takes_an_injected_latency_model() -> None:
    net = CountingNet(5.0)
    mixed, ref, _ = federated_scene(0, None)
    report = make_validator(K, N_ADV, TAU, "krum", seed=0, net_ms=net)(mixed, ref)
    assert report.consensus.sim_latency_ms == pytest.approx(15.0)  # three 5 ms hops
    assert net.calls == report.consensus.n_messages


# --------------------------------------------------------------------------- #
# defensive paths: misuse must fail loudly rather than aggregate garbage
# --------------------------------------------------------------------------- #
def test_empty_update_lists_are_refused_by_every_aggregator() -> None:
    assert krum_scores([], f=0).shape == (0,)
    for fn in (lambda: krum([], 0), lambda: multi_krum([], 0, 1), lambda: geometric_median([])):
        with pytest.raises(ValueError):
            fn()


def test_single_update_degenerates_gracefully() -> None:
    only = [make_update(0, [np.array([1.0, 2.0])])]
    assert krum_scores(only, f=0).tolist() == [0.0]
    assert krum(only, f=0) == 0
    assert np.allclose(geometric_median(only)[0], [1.0, 2.0])


def test_geometric_median_of_identical_updates_is_that_point() -> None:
    p = np.array([2.0, -5.0, 1.0])
    gm = geometric_median([make_update(i, [p.copy()]) for i in range(4)])
    assert np.allclose(gm[0], p)


def test_unflatten_refuses_a_mismatched_vector() -> None:
    with pytest.raises(ValueError, match="cannot unflatten"):
        unflatten(np.zeros(3), [np.zeros((2, 2))])


def test_ledger_serialises_numpy_arrays_in_records() -> None:
    led = Ledger()
    led.append({"scores": np.array([0.1, 0.2])})
    assert led.verify()
    assert led.to_list()[0]["record"]["scores"] == [0.1, 0.2]


def test_ledger_verify_survives_a_structurally_broken_block() -> None:
    blocks = filled_ledger(2).to_list()
    del blocks[1]["record"]
    assert Ledger.from_list(blocks).verify() is False


@pytest.mark.parametrize(
    "kwargs",
    [{"target_score": 2.5}, {"target_score": -0.1}, {"safety_margin": 1.0}, {"n_peers": 0}],
)
def test_adaptive_cosine_validates_its_configuration(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        AdaptiveCosine(**kwargs)


def test_adaptive_cosine_gives_up_when_there_is_no_orthogonal_direction() -> None:
    """A one-dimensional parameter space has no cone to hide in."""
    atk = AdaptiveCosine(target_score=0.5)
    upd = make_update(0, [np.array([2.0])])
    atk.poison(upd, AttackContext(round=0, ref=[np.array([1.0])], rng=np.random.default_rng(0)))
    out = atk.poison(
        upd, AttackContext(round=1, ref=[np.array([1.5])], rng=np.random.default_rng(0))
    )
    assert np.allclose(out.params[0], upd.params[0])


def test_label_flip_with_zero_fraction_is_a_no_op() -> None:
    labels = np.arange(10, dtype=np.int64) % 8
    assert np.array_equal(LabelFlip(frac=0.0).flip(labels), labels)
    assert LabelFlip(frac=0.5).flip(np.zeros(0, dtype=np.int64)).size == 0


# --------------------------------------------------------------------------- #
# Hypothesis properties (VALIDATION_STANDARD.md §3.8)
# --------------------------------------------------------------------------- #
@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    mat=hnp.arrays(
        dtype=np.float64,
        shape=st.tuples(st.integers(1, 8), st.integers(1, 12)),
        elements=st.floats(-1e4, 1e4, allow_nan=False, allow_infinity=False, width=64),
    ),
    use_ref=st.booleans(),
)
def test_property_fidelity_scores_stay_in_the_unit_interval_pair(
    mat: np.ndarray, use_ref: bool
) -> None:
    """Fidelity in [0, 2] for every update set -- including degenerate ones."""
    updates = [make_update(i, [row.copy()]) for i, row in enumerate(mat)]
    ref = [np.zeros(mat.shape[1])] if use_ref else None
    scores = fidelity_scores(updates, ref=ref)
    assert scores.shape == (mat.shape[0],)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0) and np.all(scores <= 2.0)


@settings(max_examples=60, deadline=None)
@given(m=st.integers(1, 16), extra=st.integers(0, 6), seed=st.integers(0, 2**16))
def test_property_pbft_never_commits_beyond_its_fault_bound(
    m: int, extra: int, seed: int
) -> None:
    """No sampled topology commits with more than (M-1)//3 Byzantine validators."""
    bound = (m - 1) // 3
    f = min(m, bound + extra)
    sim = PBFTSim(m, f, default_net_ms, seed=seed)
    res = sim.run_round(np.array([0.0, 1.0]), threshold=0.5)
    if f > bound:
        assert res.committed is False
        assert res.accepted == []
    else:
        assert res.committed is True
    senders = m if res.committed else f
    assert res.n_messages == (m - 1) + (m - 1) ** 2 + senders * (m - 1)


@settings(max_examples=60, deadline=None)
@given(
    records=st.lists(
        st.dictionaries(
            keys=st.sampled_from(["round", "cid", "score", "flag", "note"]),
            values=st.one_of(
                st.integers(-1000, 1000),
                st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
                st.booleans(),
                st.text(max_size=8),
            ),
            min_size=1,
            max_size=5,
        ),
        min_size=1,
        max_size=5,
    ),
    block_pick=st.integers(0, 4),
    new_value=st.integers(-1000, 1000),
)
def test_property_ledger_detects_a_tamper_in_any_field_of_any_block(
    records: list[dict], block_pick: int, new_value: int
) -> None:
    """verify() fails for every single-field tamper, whichever field is chosen."""
    led = Ledger()
    for r in records:
        led.append(r)
    assert led.verify()

    blocks = led.to_list()
    i = block_pick % len(blocks)
    for field in blocks[i]["record"]:
        tampered = led.to_list()
        assume(tampered[i]["record"][field] != new_value)
        tampered[i]["record"][field] = new_value
        assert Ledger.from_list(tampered).verify() is False
