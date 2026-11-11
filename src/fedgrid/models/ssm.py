"""Selective state-space classifier (Mamba/S4D-style) in pure PyTorch.

A third classifier for the FedGrid efficiency frame, alongside the CNN-LSTM
reference and the spiking (LIF) model. It is a linear-time sequence model built
from a stack of selective state-space (SSM) blocks: each block carries a
diagonal recurrent state whose input-dependent B/C/Delta ("selection") let it
route information along the 64-step window, and whose per-channel state
transition A is a learned decay.

No new dependency (CONTRACT.md house rules forbid deps beyond ``pyproject``):
this is a clean, minimal selective-scan implementation, not ``mamba-ssm`` and
not its fused CUDA kernels. It runs identically on CPU and CUDA. The diagonal
recurrence is evaluated by a parallel associative scan over the (short,
downsampled) time axis (:func:`selective_scan`, a numerically stable log-space
prefix-sum formulation) so there is no per-timestep Python loop -- the classic
un-kernelled-SSM serialization is removed without touching the model. A
line-by-line sequential reference (:func:`_selective_scan_ref`) is kept as the
correctness oracle the fast path is unit-tested against to tight float tolerance,
so the forward stays auditable -- the same standard this repo holds the SNN
SynOps path to.

Lineage (cite in the manuscript, all real references):

* Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
  2023 (arXiv:2312.00752) -- the selective SSM block (input-dependent Delta/B/C,
  the gated ``in_proj``/``out_proj`` wrapper, the short causal depthwise conv).
* Gu, Goel, Re, "Efficiently Modeling Long Sequences with Structured State
  Spaces" (S4), ICLR 2022 (arXiv:2111.00396) -- the state-space sequence layer.
* Gu, Gupta, Goel, Re, "On the Parameterization and Initialization of Diagonal
  State Space Models" (S4D), NeurIPS 2022 (arXiv:2206.11893) -- the diagonal
  real (S4D-Real) parameterization / initialization of A used here.

Federation note (CONTRACT.md house rules + docs/OPEN_QUESTIONS.md, "federated
SNN collapses"): the SNN investigation found that federated averaging corrupts
models whose *per-channel dynamics* parameters (there: learnable per-neuron
beta/threshold) are averaged across heterogeneous clients. An SSM has the exact
analogue -- the state transition ``A_log``, the timescale bias ``dt_proj.bias``
(the discretization step Delta) and the per-channel feedthrough ``D``.
:meth:`SSMClassifier.local_parameter_names` names them so an aggregator can hold
them local (personalized) instead of averaging, making the model robust-by-design
for federation without changing the forward pass. This mirrors the convention the
SNN fix is expected to adopt; if the SNN lands a differently named hook, unify on
that name. Interface identical to the other classifiers:
``SSMClassifier(in_ch=11, n_classes=8, ...)``, ``forward`` maps ``(B, 11, 64) ->
(B, 8)`` logits, and ``macs_per_inference() -> int`` is an analytic, hand-checkable
multiply-accumulate count.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["WINDOW_T", "SSMBlock", "SSMClassifier"]

# Data-contract window length (CONTRACT.md: T = 64). The analytic MAC count is
# defined for one contract window; the layers themselves are length-agnostic.
WINDOW_T: int = 64


class _RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean-centering), Mamba's normalization."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight


def _selective_scan_ref(
    u: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor, D: Tensor
) -> Tensor:
    """Sequential reference scan (Mamba's ``selective_scan_ref``, pure PyTorch).

    The readable ground truth: it discretizes ``h' = A h + B u`` by zero-order
    hold and unrolls the diagonal recurrence one timestep at a time::

        dA = exp(delta * A)            (per channel e, per state n)
        h_t = dA_t * h_{t-1} + delta_t * B_t * u_t
        y_t = sum_n C_t[n] * h_t[.,n] + D * u_t

    Kept only as the correctness oracle for :func:`selective_scan` (the fast path
    is asserted ``allclose`` to this in ``tests/test_ssm.py``). The Python loop
    over time makes it slow on GPU -- do not use it in the forward pass.

    Shapes: ``u, delta`` ``(batch, L, E)``; ``A`` ``(E, N)`` (negative real, the
    S4D-Real transition); ``B, C`` ``(batch, L, N)`` (selective -- input
    dependent, shared across channels); ``D`` ``(E,)``. Returns ``(batch, L, E)``.
    """
    b, length, e = u.shape
    n = A.shape[1]
    # Zero-order-hold discretization; A < 0 keeps dA in (0, 1) -> a stable decay.
    d_a = torch.exp(delta.unsqueeze(-1) * A)  # (b, L, E, N)
    # Mamba's simplified input term: dB*u with dB approximated by delta*B (its own
    # reference implementation folds the ZOH B into delta*B for the diagonal case).
    d_bu = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)  # (b, L, E, N)

    h = u.new_zeros(b, e, n)
    ys: list[Tensor] = []
    for t in range(length):
        h = d_a[:, t] * h + d_bu[:, t]  # (b, E, N)
        ys.append(torch.einsum("ben,bn->be", h, C[:, t]))  # readout over states
    y = torch.stack(ys, dim=1)  # (b, L, E)
    return y + u * D  # per-channel skip (feedthrough)


def selective_scan(
    u: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor, D: Tensor, chunk: int = 4
) -> Tensor:
    """Diagonal selective SSM scan -- a chunked parallel scan (no per-step loop).

    Numerically equivalent to :func:`_selective_scan_ref` (asserted ``allclose``
    to tight tolerance in the tests) but with the per-timestep Python loop
    replaced by a handful of vectorized chunks, so the GPU is not fed L tiny,
    launch-bound kernels. Same discretization::

        dA_t = exp(delta_t * A)   (diagonal, per channel e / state n; dA in (0, 1])
        h_t  = dA_t * h_{t-1} + delta_t * B_t * u_t          (h_{-1} = 0)
        y_t  = sum_n C_t[n] * h_t[.,n] + D * u_t

    Unrolling the diagonal recurrence, the product of transitions between two
    steps is an inclusive cumulative sum in log space::

        logA_t = delta_t * A                    (<= 0, since delta > 0 and A < 0)
        prod_{j=k+1..t} dA_j = exp(cumA_t - cumA_k)   for t >= k, and lies in (0, 1].

    Stability comes from keeping the subtraction *inside* the exponential: every
    exponent is <= 0, so nothing overflows -- unlike the
    ``exp(cumA_t) * cumsum(exp(-cumA_k) * dBu_k)`` factorization whose
    ``exp(-cumA_k)`` blows up under strong decay.

    The time axis is processed in blocks of ``chunk`` steps. Within a block the
    ``w x w`` segment-decay matrix ``exp(cumA_i - cumA_k)`` (``cumA`` reset to the
    block start) resolves the intra-block dependency in one shot; the state
    carried from the previous block enters via ``exp(cumA_i) * h_prev`` (both
    factors in ``(0, 1]``). Only ``ceil(L / chunk)`` blocks are visited, and the
    scan touches ``O(chunk)`` times the minimal work instead of the ``O(L)`` a
    full ``L x L`` materialization would -- cheap on CPU, few launches on GPU.
    ``chunk`` only reassociates the identical sum, so it never changes the result.

    Shapes match :func:`_selective_scan_ref`: ``u, delta`` ``(batch, L, E)``;
    ``A`` ``(E, N)``; ``B, C`` ``(batch, L, N)``; ``D`` ``(E,)`` -> ``(batch, L, E)``.
    """
    b, length, e = u.shape
    n = A.shape[1]
    # Per-step log-transition (<= 0) and discretized input, formed once, vectorized.
    log_a = delta.unsqueeze(-1) * A  # (b, L, E, N)
    d_bu = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)  # (b, L, E, N)

    step = max(int(chunk), 1)
    h = u.new_zeros(b, e, n)  # carried state h_{s-1}, zero before the first block
    ys: list[Tensor] = []
    for s in range(0, length, step):
        w = min(step, length - s)
        cum = log_a[:, s : s + w].cumsum(dim=1)  # (b, w, E, N), <= 0, block-local
        # Intra-block segment decay decay[.,i,k,.,.] = exp(cumA_i - cumA_k), i >= k.
        # -inf on the acausal (k > i) triangle -> exp() = 0 with no positive exponent.
        acausal = torch.ones(w, w, dtype=torch.bool, device=u.device).triu_(1)
        mask = torch.zeros(w, w, dtype=cum.dtype, device=u.device)
        mask.masked_fill_(acausal, float("-inf"))
        seg = cum.unsqueeze(2) - cum.unsqueeze(1) + mask[None, :, :, None, None]
        decay = torch.exp(seg)  # (b, i, k, E, N), 0 above the diagonal
        # State over the block: intra-block accumulation + the decayed carry-in.
        h_intra = torch.einsum("biken,bken->bien", decay, d_bu[:, s : s + w])
        h_block = h_intra + torch.exp(cum) * h.unsqueeze(1)  # (b, w, E, N)
        ys.append(torch.einsum("bien,bin->bie", h_block, C[:, s : s + w]))
        h = h_block[:, -1]  # carry the block's final state into the next block
    y = torch.cat(ys, dim=1)  # (b, L, E)
    return y + u * D  # per-channel skip (feedthrough)


class SSMBlock(nn.Module):
    """One Mamba-style selective SSM block: pre-norm, gated, residual.

    ``x -> RMSNorm -> in_proj -> (x, z)``; the ``x`` branch passes through a short
    causal depthwise conv and SiLU, produces its own ``(Delta, B, C)`` via a
    linear projection (this is the *selection*: they depend on the input), runs
    the diagonal :func:`selective_scan`, is gated by ``SiLU(z)`` and projected
    back with ``out_proj``. Bias conventions follow Mamba (only the depthwise conv
    and ``dt_proj`` carry biases).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dt_rank: int = 4,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ) -> None:
        super().__init__()
        d_inner = expand * d_model
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner
        self.d_conv = d_conv
        self.dt_rank = dt_rank

        self.norm = _RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        # Depthwise (groups=d_inner) causal conv: left-pad by d_conv-1, trim the
        # right overhang so position t only ever sees t' <= t.
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv, groups=d_inner, padding=d_conv - 1
        )
        # Produces the selective Delta (low-rank, via dt_proj) plus B and C.
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # A_log parameterizes A = -exp(A_log) (S4D-Real): A initialized to the
        # 1..N ramp so each state starts at a distinct decay timescale.
        a_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a_init))
        self.D = nn.Parameter(torch.ones(d_inner))  # per-channel feedthrough
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        self._init_dt_bias(dt_min, dt_max)

    def _init_dt_bias(self, dt_min: float, dt_max: float) -> None:
        """Init dt_proj.bias so softplus(bias) spans [dt_min, dt_max] (Mamba recipe).

        The Delta timescale prior lives entirely in this bias; seeding it in log
        space and inverting the softplus puts every channel at a sensible,
        distinct initial step size (the property the selective SSM relies on).
        Deterministic under ``torch.manual_seed`` (the engine seeds per client).
        """
        rand = torch.rand(self.d_inner)
        dt = torch.exp(
            rand * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=1e-4)
        # Inverse of softplus: x such that softplus(x) = dt.
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: Tensor) -> Tensor:
        """``(B, L, d_model) -> (B, L, d_model)`` residual output (pre-norm inside)."""
        length = x.shape[1]
        xn = self.norm(x)
        x_inner, z = self.in_proj(xn).chunk(2, dim=-1)  # (B, L, d_inner) each

        # Causal short conv along time, then SiLU (Mamba's local-context gate).
        xc = self.conv1d(x_inner.transpose(1, 2))[..., :length].transpose(1, 2)
        xc = F.silu(xc)

        dbl = self.x_proj(xc)
        dt, b_sel, c_sel = torch.split(
            dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt))  # (B, L, d_inner), strictly positive
        a = -torch.exp(self.A_log)  # (d_inner, d_state), negative real

        y = selective_scan(xc, delta, a, b_sel, c_sel, self.D)
        y = y * F.silu(z)  # gate
        return self.out_proj(y)


class SSMClassifier(nn.Module):
    """Patch-stem + stacked selective SSM blocks + mean-pooled linear head.

    Pipeline: ``(B, C, T)`` window -> a strided depthwise-free Conv1d "patch"
    stem that embeds the 11 channels into ``d_model`` and downsamples time by
    ``patch`` (64 -> 16 by default, mirroring how the CNN-LSTM pools 64 -> 16
    before its recurrent head) -> ``n_layers`` residual :class:`SSMBlock`s ->
    RMSNorm -> mean over time -> linear head to ``n_classes`` logits. The stem is
    the standard SSM front-end (patch embedding, cf. Vision Mamba); it keeps the
    model in the CNN-LSTM/SNN efficiency regime by cutting the sequence length the
    quadratic-free-but-still-per-step scan runs over.

    Defaults give **278,216 parameters / 4,590,336 MACs per inference** -- the
    same order as the CNN-LSTM (112k params, 2.0M MACs) and SNN (~250k params),
    so the efficiency trade-off story stays coherent. There is no stochasticity in
    the forward pass, so it is deterministic given the weights; ``seed`` is
    accepted only for interface parity with the other classifiers (parameter init
    is governed by the ambient ``torch`` seed exactly like the CNN-LSTM).
    """

    def __init__(
        self,
        in_ch: int = 11,
        n_classes: int = 8,
        d_model: int = 96,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        n_layers: int = 4,
        patch: int = 4,
        window_t: int = WINDOW_T,
        dt_rank: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"need at least one SSM block, got {n_layers}")
        if window_t % patch != 0:
            raise ValueError(
                f"patch stem downsamples time: window_t={window_t} not "
                f"divisible by patch={patch}"
            )
        self.in_ch = in_ch
        self.n_classes = n_classes
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_conv = d_conv
        self.n_layers = n_layers
        self.patch = patch
        self.window_t = window_t
        # dt_rank defaults to Mamba's ceil(d_model/16): the low-rank width of the
        # input-dependent Delta projection.
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_model / 16)
        self.seq_len = window_t // patch
        self.seed = seed

        # Non-overlapping patch embedding (kernel == stride == patch): embeds the
        # 11 channels and downsamples time in one cheap, hand-countable op.
        self.stem = nn.Conv1d(in_ch, d_model, kernel_size=patch, stride=patch)
        self.blocks = nn.ModuleList(
            SSMBlock(
                d_model=d_model,
                d_state=d_state,
                expand=expand,
                d_conv=d_conv,
                dt_rank=self.dt_rank,
            )
            for _ in range(n_layers)
        )
        self.norm = _RMSNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """``(B, C, T) -> (B, n_classes)`` logits (mean-pooled over the window)."""
        if x.dim() != 3 or x.shape[1] != self.in_ch or x.shape[2] != self.window_t:
            raise ValueError(
                f"expected (B, {self.in_ch}, {self.window_t}) input, got {tuple(x.shape)}"
            )
        z = self.stem(x).transpose(1, 2)  # (B, seq_len, d_model)
        for block in self.blocks:
            z = z + block(z)  # pre-norm residual (norm lives inside the block)
        z = self.norm(z)
        return self.head(z.mean(dim=1))  # summary after the whole window is seen

    def local_parameter_names(self) -> list[str]:
        """Names of the per-channel SSM *dynamics* parameters -- federation-sensitive.

        These are the SSM analogue of the SNN's learnable per-neuron
        beta/threshold that federated averaging was found to corrupt
        (docs/OPEN_QUESTIONS.md): the state transition ``A_log``, the discretization
        timescale bias ``dt_proj.bias`` (the static part of Delta) and the
        per-channel feedthrough ``D``. Each is intrinsic to a channel's temporal
        dynamics and specializes per client under non-IID data, so averaging them
        across heterogeneous substations mixes incompatible operating points.

        Returned as keys of :meth:`torch.nn.Module.named_parameters`, so an
        aggregator can build an index mask and hold these local (personalized)
        while still averaging every feature/projection weight. It does not change
        the forward pass; it is a hook the FL layer may consult later.
        """
        suffixes = ("A_log", "D", "dt_proj.bias")
        return [
            name
            for name, _ in self.named_parameters()
            if name.startswith("blocks.") and name.split(".", 2)[2] in suffixes
        ]

    def macs_per_inference(self) -> int:
        """Analytic multiply-accumulate count for one contract ``(C, T=64)`` window.

        Convention (matches the CNN-LSTM's): count the multiplies in the linear
        projections, the depthwise conv, the patch stem and the head, plus the two
        genuine multiply-accumulate contractions inside the scan (the state
        recurrence and the readout over states). Excluded -- as sub-percent
        element-wise/activation arithmetic, exactly as the CNN-LSTM excludes its
        LSTM gate element-wise ops: the ``exp``/``softplus``/SiLU activations, the
        ``delta*B*u`` input discretization, the per-channel ``D`` skip, the gate
        multiply and every bias add.

        With ``E = expand*d_model``, ``N = d_state``, ``R = dt_rank``,
        ``L = window_t/patch`` and ``D = d_model``, per block::

            in_proj  : L * D * (2E)
            conv1d   : L * E * d_conv           (depthwise)
            x_proj   : L * E * (R + 2N)
            dt_proj  : L * R * E
            scan     : 2 * L * E * N            (recurrence MAC + readout MAC)
            out_proj : L * E * D

        plus the stem ``L * D * (in_ch * patch)`` and the head ``D * n_classes``.

        Default config, hand check (E=192, N=16, R=6, L=16, D=96, 4 layers):
            stem   : 16*96*(11*4)               = 67,584
            block  : 16*96*384                  = 589,824   (in_proj)
                   + 16*192*4                   = 12,288    (conv1d)
                   + 16*192*(6+32)              = 116,736   (x_proj)
                   + 16*6*192                   = 18,432    (dt_proj)
                   + 2*16*192*16                = 98,304    (scan)
                   + 16*192*96                  = 294,912   (out_proj)
                   = 1,130,496 per block, x4    = 4,521,984
            head   : 96*8                       = 768
            total                               = 4,590,336
        """
        e = self.expand * self.d_model
        n = self.d_state
        r = self.dt_rank
        length = self.seq_len
        d = self.d_model

        macs = length * d * (self.in_ch * self.patch)  # patch stem
        per_block = (
            length * d * (2 * e)  # in_proj
            + length * e * self.d_conv  # depthwise conv
            + length * e * (r + 2 * n)  # x_proj (Delta rank + B + C)
            + length * r * e  # dt_proj
            + 2 * length * e * n  # scan: recurrence MAC + readout MAC
            + length * e * d  # out_proj
        )
        macs += self.n_layers * per_block
        macs += d * self.n_classes  # head (after mean-pool)
        return macs
