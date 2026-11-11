"""Tests for the selective SSM classifier (Mamba/S4D-style, pure PyTorch).

Same correctness bar the other classifiers meet (CONTRACT.md, models/metrics):
forward shapes on random tensors, gradient flow to every parameter (the scan and
the state-dynamics parameters included), an analytic ``macs_per_inference`` matched
to a hand count on a tiny config, CPU determinism under a fixed seed, and a
learnability sanity loop (>70% train accuracy on a separable 8-class toy task,
inside a time budget calibrated to the host). Also pins the parameter budget
(efficiency-regime guard),
the federation-sensitive ``local_parameter_names`` set, and the ready-to-activate
integration path (build factory + opt-in battery arms) so activation is one flag.
"""

from __future__ import annotations

import time

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from fedgrid.models.ssm import (
    SSMBlock,
    SSMClassifier,
    _selective_scan_ref,
    selective_scan,
)

# ----------------------------------------------------- fast-scan equivalence


@pytest.mark.parametrize(
    ("b", "length", "e", "n"),
    [
        (1, 8, 5, 3),  # short, tiny
        (2, 16, 8, 4),  # contract length, small
        (3, 16, 12, 16),  # contract length, full state
        (4, 16, 192, 16),  # the default forward shape (E=192, N=16)
        (2, 7, 6, 4),  # odd, non-power-of-two length
        (2, 32, 10, 8),  # longer than the contract window
    ],
)
def test_fast_selective_scan_matches_sequential_reference(
    b: int, length: int, e: int, n: int
) -> None:
    """The parallel scan is numerically equal to the sequential loop it replaces.

    This is the correctness gate for the speed rewrite: a fast-but-wrong scan is
    worthless. Inputs mirror the model's real ranges -- ``delta`` strictly
    positive (a softplus output) and ``A`` strictly negative (S4D-Real,
    ``-exp(A_log)``) -- so the decay stays in ``(0, 1]`` exactly as in the block.
    """
    gen = torch.Generator().manual_seed(1234 + b + length + e + n)
    u = torch.randn(b, length, e, generator=gen)
    delta = F.softplus(torch.randn(b, length, e, generator=gen))  # > 0
    a = -torch.exp(torch.randn(e, n, generator=gen))  # < 0 (S4D-Real)
    b_sel = torch.randn(b, length, n, generator=gen)
    c_sel = torch.randn(b, length, n, generator=gen)
    d = torch.randn(e, generator=gen)

    fast = selective_scan(u, delta, a, b_sel, c_sel, d)
    ref = _selective_scan_ref(u, delta, a, b_sel, c_sel, d)

    assert fast.shape == ref.shape == (b, length, e)
    assert torch.isfinite(fast).all()
    max_abs = (fast - ref).abs().max().item()
    assert torch.allclose(fast, ref, atol=1e-5, rtol=1e-4), f"max|fast-ref|={max_abs:.2e}"


def _scan_leaf_inputs() -> list[torch.Tensor]:
    """Fresh leaf tensors (``requires_grad``) in the model's real value ranges."""
    gen = torch.Generator().manual_seed(7)
    u = torch.randn(3, 16, 12, generator=gen)
    delta = F.softplus(torch.randn(3, 16, 12, generator=gen))  # > 0
    a = -torch.exp(torch.randn(12, 8, generator=gen))  # < 0 (S4D-Real)
    b_sel = torch.randn(3, 16, 8, generator=gen)
    c_sel = torch.randn(3, 16, 8, generator=gen)
    d = torch.randn(12, generator=gen)
    # detach()+requires_grad_ makes each a leaf we can read .grad from.
    return [t.detach().requires_grad_(True) for t in (u, delta, a, b_sel, c_sel, d)]


def test_fast_selective_scan_gradients_match_reference() -> None:
    """Backward equivalence too: the fast scan must train identically, not just infer.

    Both paths get the same upstream gradient; every input's grad must agree, so
    the rewrite cannot silently perturb the learning signal to A/Delta/B/C/D.
    """
    fast_in = _scan_leaf_inputs()
    ref_in = _scan_leaf_inputs()  # same seed -> identical draws
    grad_out = torch.randn(3, 16, 12, generator=torch.Generator().manual_seed(99))

    selective_scan(*fast_in).backward(grad_out)
    _selective_scan_ref(*ref_in).backward(grad_out)
    for name, tf, tr in zip(
        ("u", "delta", "A", "B", "C", "D"), fast_in, ref_in, strict=True
    ):
        assert tf.grad is not None and tr.grad is not None, name
        assert torch.allclose(tf.grad, tr.grad, atol=1e-5, rtol=1e-4), name


# --------------------------------------------------------------------- forward


def test_forward_shape() -> None:
    torch.manual_seed(0)
    model = SSMClassifier()
    out = model(torch.randn(4, 11, 64))
    assert out.shape == (4, 8)
    assert torch.isfinite(out).all()
    # A non-default (smaller) config must reshape accordingly.
    small = SSMClassifier(in_ch=3, n_classes=5, d_model=16, n_layers=2, patch=8)
    assert small(torch.randn(2, 3, 64)).shape == (2, 5)


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="expected"):
        SSMClassifier()(torch.randn(2, 11, 32))  # wrong T
    with pytest.raises(ValueError, match="expected"):
        SSMClassifier()(torch.randn(2, 5, 64))  # wrong C
    with pytest.raises(ValueError, match="divisible"):
        SSMClassifier(patch=5)  # 64 % 5 != 0
    with pytest.raises(ValueError, match="at least one"):
        SSMClassifier(n_layers=0)


def test_gradients_flow() -> None:
    """Every parameter -- projections, conv, and the A/Delta/D dynamics -- gets a grad."""
    torch.manual_seed(0)
    model = SSMClassifier(d_model=32, n_layers=2)
    loss = F.cross_entropy(model(torch.randn(8, 11, 64)), torch.randint(0, 8, (8,)))
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def test_selective_ssm_dynamics_params_receive_gradient() -> None:
    """The state-space dynamics (A_log, dt_proj.bias, D) must actually train."""
    torch.manual_seed(0)
    model = SSMClassifier(d_model=16, n_layers=1)
    F.cross_entropy(model(torch.randn(8, 11, 64)), torch.randint(0, 8, (8,))).backward()
    block = model.blocks[0]
    for name, p in (("A_log", block.A_log), ("D", block.D), ("dt_bias", block.dt_proj.bias)):
        assert p.grad is not None and p.grad.abs().sum() > 0, name


# ---------------------------------------------------------------- determinism


def test_determinism_under_fixed_seed() -> None:
    # There is no stochasticity in the forward pass: equal weights -> equal logits,
    # across two independently-seeded builds and across repeated calls.
    torch.manual_seed(7)
    a = SSMClassifier()
    torch.manual_seed(7)
    b = SSMClassifier()
    x = torch.randn(3, 11, 64, generator=torch.Generator().manual_seed(3))
    a.eval()
    b.eval()
    out_a = a(x)
    assert torch.equal(out_a, b(x))  # same seed -> identical init -> identical logits
    assert torch.equal(out_a, a(x))  # deterministic across calls (no sampling)
    # Different init draws different weights -> different logits.
    torch.manual_seed(8)
    c = SSMClassifier()
    c.eval()
    assert not torch.equal(out_a, c(x))


# ------------------------------------------------------------------ MAC count


def test_macs_hand_count_small_config() -> None:
    # Hand count. in_ch=1, n_classes=2, d_model=8, d_state=4, expand=2, d_conv=2,
    # n_layers=1, patch=2, window_t=8, dt_rank=2.
    # Derived: E = expand*d_model = 16, N = 4, R = 2, L = window_t/patch = 4, D = 8.
    #   stem     : L*D*(in_ch*patch)  = 4*8*(1*2)   =    64
    #   in_proj  : L*D*(2E)           = 4*8*32      = 1,024
    #   conv1d   : L*E*d_conv         = 4*16*2      =   128
    #   x_proj   : L*E*(R+2N)         = 4*16*10     =   640
    #   dt_proj  : L*R*E              = 4*2*16      =   128
    #   scan     : 2*L*E*N            = 2*4*16*4    =   512
    #   out_proj : L*E*D              = 4*16*8      =   512
    #   block total (x1 layer)                      = 2,944
    #   head     : D*n_classes        = 8*2         =    16
    #   total                                       = 3,024
    model = SSMClassifier(
        in_ch=1, n_classes=2, d_model=8, d_state=4, expand=2, d_conv=2,
        n_layers=1, patch=2, window_t=8, dt_rank=2,
    )
    assert model.macs_per_inference() == 3024


def test_macs_default_config() -> None:
    # E=192, N=16, R=6, L=16, D=96, 4 layers:
    #   stem 16*96*44=67,584; per-block 1,130,496 x4 = 4,521,984; head 96*8=768.
    assert SSMClassifier().macs_per_inference() == 4_590_336


# --------------------------------------------------- parameter-budget guard


def test_parameter_budget_in_efficiency_regime() -> None:
    """< 1M params (contract), same order as CNN-LSTM (112k) / SNN (~250k)."""
    n_params = sum(p.numel() for p in SSMClassifier().parameters())
    assert n_params == 278_216  # exact, so the budget cannot silently drift
    assert n_params < 1_000_000
    assert 100_000 < n_params < 500_000  # the stated efficiency band


# --------------------------------------------- federation-sensitive params


def test_local_parameter_names_are_the_dynamics_params() -> None:
    """A_log, dt_proj.bias and D (per-channel dynamics) are flagged local; nothing else."""
    model = SSMClassifier(d_model=16, n_layers=3)
    names = set(model.local_parameter_names())
    param_names = dict(model.named_parameters())

    expected = {
        f"blocks.{i}.{suffix}"
        for i in range(3)
        for suffix in ("A_log", "D", "dt_proj.bias")
    }
    assert names == expected
    # Every flagged name is a real parameter...
    assert names <= set(param_names)
    # ...and the feature/projection weights (safe to average) are NOT flagged.
    for safe in ("stem.weight", "head.weight", "blocks.0.in_proj.weight",
                 "blocks.0.out_proj.weight", "blocks.0.x_proj.weight",
                 "blocks.0.conv1d.weight", "blocks.0.dt_proj.weight"):
        assert safe in param_names and safe not in names


# -------------------------------------------------------- learnability loop


def _toy_data(n: int = 200, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Separable 8-class toy (same construction as the CNN-LSTM/SNN tests).

    Class k puts a square pulse on channel k (rising edge t=16, falling t=48), so
    the selective SSM has a clear, class-specific temporal feature to route.
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


#: Optimiser steps the loop below performs: 10 epochs x ceil(200 / 64) minibatches.
_TOY_STEPS = 40

#: How many times a single warmed-up step the whole loop is allowed to cost, plus
#: a constant for timer noise and the eval pass.
_STEP_BUDGET = 4.0
_TOY_SLACK_S = 5.0

#: Absolute stop, so a genuine hang cannot wedge the suite however slow the host.
_TOY_HARD_CEILING_S = 600.0


def _one_step_seconds() -> float:
    """Cost of one warmed-up training step of the default model, on this host.

    Uses its own generators for the data so the calibration cannot perturb the
    seeded run that follows (the caller reseeds before training regardless).
    """
    model = SSMClassifier()
    x = torch.randn(64, 11, 64, generator=torch.Generator().manual_seed(11))
    y = torch.randint(0, 8, (64,), generator=torch.Generator().manual_seed(12))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    def step() -> None:
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()

    step()  # warm the allocator and the thread pool; not timed
    t0 = time.perf_counter()
    for _ in range(3):
        step()
    return (time.perf_counter() - t0) / 3.0


def test_toy_task_learns() -> None:
    """200 synthetic separable samples: >70% train accuracy, in bounded time.

    Proves the selective scan actually learns end-to-end, not just that shapes
    line up. The accuracy bar is absolute: the default model reaches 100% here.

    The time bar is a *runaway* guard -- a per-timestep Python loop creeping back
    into the hot path, an accidentally quadratic scan, an allocation that grows
    with the epoch -- and not a benchmark, so it is calibrated to the host rather
    than fixed in seconds. Wall-clock seconds are a property of the machine: the
    loop is 40 Adam steps of a 278k-parameter model, and a 6-thread desktop under
    load spends ~75 s on a loop a quiet reference box runs in ~18 s, which is what
    made the old flat 60 s ceiling a flake rather than a signal. Timing one warmed
    step here and requiring the loop to fit in `_STEP_BUDGET` of them still fails
    loudly if per-step cost grows across the loop or if the loop runs more steps
    than it should, while staying immune to how busy the machine is.
    """
    step = _one_step_seconds()
    budget = min(_STEP_BUDGET * _TOY_STEPS * step + _TOY_SLACK_S,
                 _TOY_HARD_CEILING_S)

    torch.manual_seed(0)
    t0 = time.perf_counter()
    acc = _train_accuracy(SSMClassifier(), lr=3e-3, epochs=10)
    elapsed = time.perf_counter() - t0

    assert acc > 0.70, f"train accuracy {acc:.3f} <= 0.70"
    assert elapsed < budget, (
        f"took {elapsed:.1f}s for {_TOY_STEPS} steps; one warmed step costs "
        f"{step * 1e3:.0f} ms on this host, so the loop should fit in {budget:.1f}s")


# ---------------------------------------------------- block-level sanity


def test_block_is_residual_shape_preserving() -> None:
    torch.manual_seed(0)
    block = SSMBlock(d_model=16, d_state=8, expand=2, d_conv=4, dt_rank=2)
    z = torch.randn(2, 16, 16)  # (B, L, d_model)
    assert block(z).shape == z.shape  # out_proj returns to d_model for the residual add


# ------------------------------------------------ ready-to-activate wiring


def test_build_factory_constructs_ssm() -> None:
    """experiments.build.make_model_fn builds the SSM from a config block."""
    from experiments.build import make_model_fn

    cfg = {"model": {"name": "ssm", "kwargs": {"d_model": 16, "n_layers": 2}}, "seed": 0}
    model = make_model_fn(cfg)()
    assert isinstance(model, SSMClassifier)
    assert model(torch.randn(2, 11, 64)).shape == (2, 8)


def test_default_battery_unchanged_and_ssm_is_opt_in() -> None:
    """The SSM must not perturb the fixed Compact battery; it appends only on request."""
    from experiments.manifest import MODELS, MODELS_EXT, battery_runs, make_config

    assert "ssm" not in MODELS and "ssm" in MODELS_EXT  # excluded by default, buildable

    base = battery_runs()
    assert len(base) == 500  # the fixed Compact set, byte-for-byte unchanged
    assert not any(r.model == "ssm" for r in base)

    with_ssm = battery_runs(include_ssm=True)
    ssm_runs = [r for r in with_ssm if r.model == "ssm"]
    # One clean-reference frame (6 arms) x COMPACT Tier-1 seeds, in its own block.
    assert len(with_ssm) == len(base) + len(ssm_runs)
    assert ssm_runs and all(r.block == "t1a_ssm" for r in ssm_runs)
    ids = [r.run_id for r in with_ssm]
    assert len(ids) == len(set(ids))  # no collisions with the default set

    # make_config accepts the ssm model and emits a clean, JSON-native block.
    cfg = make_config(model="ssm", arm="fedavg", seed=0)
    assert cfg["model"] == {"name": "ssm", "kwargs": {}}
