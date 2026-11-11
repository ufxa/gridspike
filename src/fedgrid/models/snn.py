"""Spiking classifier: LIF neurons in pure PyTorch (house pattern from paper6).

No snntorch dependency — the surrogate gradient and the SynOps counters must be
auditable line-by-line because the energy claim rests on them. The neuron model
and training recipe are exactly snnTorch's documented default (hard-reset leaky
integrate-and-fire with an arctan surrogate gradient): Eshraghian et al.,
"Training Spiking Neural Networks Using Lessons From Deep Learning",
Proc. IEEE 111(9):1016-1054, 2023 (arXiv:2109.12894).
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn

__all__ = ["LIF", "SpikingClassifier"]


class _ArctanSpike(torch.autograd.Function):
    """Heaviside spike with the arctan surrogate gradient.

    Forward: S = H(v) with v = mem - threshold. Backward uses the derivative of
    the smoothed spike S~ = (1/pi)·arctan(pi·v·alpha/2) + 1/2, i.e.
    dS/dv = (alpha/2) / (1 + (pi·v·alpha/2)^2) — snnTorch's ``surrogate.atan``
    with its default alpha = 2 (Eshraghian et al., Proc. IEEE 2023, §IV).
    """

    @staticmethod
    def forward(ctx, v: Tensor, alpha: float) -> Tensor:
        ctx.save_for_backward(v)
        ctx.alpha = alpha
        return (v > 0).to(v.dtype)

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tuple[Tensor, None]:
        (v,) = ctx.saved_tensors
        alpha = ctx.alpha
        scaled = v * (math.pi * alpha / 2.0)
        return grad_out * (alpha / 2.0) / (1.0 + scaled * scaled), None


def _accumulate(counter: Tensor, value: Tensor) -> Tensor:
    """Out-of-place float64 accumulation, robust under ``inference_mode``.

    Counters are plain (non-buffer) tensors so they never leak into state_dicts
    exchanged by the FL engine; out-of-place adds keep autograd/inference-mode
    entirely out of the accounting path.
    """
    value = value.detach().to(dtype=torch.float64)
    if counter.device != value.device:
        counter = counter.to(value.device)
    return counter + value


class LIF(nn.Module):
    """Hard-reset leaky integrate-and-fire layer of ``n`` neurons.

    Dynamics per timestep: ``mem = beta*mem + I``; a neuron spikes when
    ``mem > threshold`` and its membrane is then hard-reset to zero. Both decay
    and threshold are learnable **per neuron**: beta is sigmoid-parameterised so
    it stays in (0, 1) under unconstrained SGD; the threshold is learned
    directly (snnTorch's ``learn_beta``/``learn_threshold`` pattern, Eshraghian
    et al. 2023). The reset uses the *detached* spike so gradients reach the
    parameters only through the arctan surrogate, matching snnTorch's default.

    Emitted spikes are accumulated in ``spike_count`` (float64, reset via
    :meth:`reset_counters`) — the raw material of SynOps accounting.
    """

    def __init__(
        self,
        n: int,
        beta_init: float = 0.9,
        threshold_init: float = 1.0,
        alpha: float = 2.0,
    ) -> None:
        super().__init__()
        if not 0.0 < beta_init < 1.0:
            raise ValueError(f"beta_init must be in (0, 1), got {beta_init}")
        self.n = n
        self.alpha = alpha
        # Inverse sigmoid so sigmoid(_beta_raw) == beta_init at start.
        raw = math.log(beta_init / (1.0 - beta_init))
        self._beta_raw = nn.Parameter(torch.full((n,), raw))
        self.threshold = nn.Parameter(torch.full((n,), float(threshold_init)))
        self.mem: Tensor | None = None
        self.spike_count: Tensor = torch.zeros((), dtype=torch.float64)

    @property
    def beta(self) -> Tensor:
        """Per-neuron membrane decay in (0, 1)."""
        return torch.sigmoid(self._beta_raw)

    def reset_state(self) -> None:
        """Forget the membrane potential (call at every window boundary)."""
        self.mem = None

    def reset_counters(self) -> None:
        self.spike_count = torch.zeros((), dtype=torch.float64)

    def forward(self, x: Tensor) -> Tensor:
        """One timestep: input current ``(B, n)`` -> spikes ``(B, n)`` in {0, 1}."""
        if self.mem is None or self.mem.shape != x.shape:
            self.mem = torch.zeros_like(x)
        mem = self.beta * self.mem + x
        spk = _ArctanSpike.apply(mem - self.threshold, self.alpha)
        # Hard reset to zero; detached so the reset path carries no gradient.
        self.mem = mem * (1.0 - spk.detach())
        with torch.no_grad():
            self.spike_count = _accumulate(self.spike_count, spk.sum())
        return spk


class SpikingClassifier(nn.Module):
    """Stacked LIF layers over ``T_steps`` internal timesteps, rate-decoded logits.

    Pipeline: ``(B, C, T)`` window -> spike coding -> Linear -> LIF -> [Linear ->
    LIF ...] -> linear readout accumulated over the unroll. Logits are the
    readout averaged over timesteps, i.e. a linear decode of per-neuron firing
    rates: (1/T)·sum_t(W·s_t + b) = W·mean_rate + b (rate decoding).

    Codings (selectable; survey: Auge et al., Neural Processing Letters 2021,
    DOI 10.1007/s11063-021-10562-2):

    * ``"rate"`` — the flattened window is presented at every unroll step as
      Bernoulli spikes with p = sigmoid(x) (inputs are z-scored per channel by
      the upstream ``Normalizer``, so sigmoid is a monotone map onto (0, 1)).
      Sampling uses a private generator seeded with ``seed`` at each forward, so
      coding is a *deterministic dither* of the input — bit-reproducible and
      independent of call order (CPU/CUDA draw sequences differ; tests pin CPU).
    * ``"delta"`` — temporal-contrast coding along the window's native time
      axis: ON/OFF events where the first difference exceeds ±``delta_threshold``
      (in z-score units). The native ``window_t`` axis is folded into ``T_steps``
      chunks of ``window_t / T_steps`` samples, so unroll step t presents the
      events of its own time segment. Fully deterministic. The default
      threshold 0.2 is val-tuned ({0.02, 0.05, 0.1, 0.2, 0.35} x 3 seeds,
      centralized, realistic prior): smaller thresholds fire on z-scored
      CT/PT measurement noise, which *lowers* macro-F1 while *raising*
      SynOps — 0.2 beat the old 0.1 on both axes (0.596 vs 0.535 val
      macro-F1; 61k vs 78k SynOps/inference).

    ``seed``, ``delta_threshold``, ``window_t``, ``beta_init`` and
    ``threshold_init`` extend the contract signature with defaulted keywords
    (contract prefix unchanged); ``window_t`` is the data-contract T = 64 and
    exists so tests can build tiny hand-checkable nets.

    ``threshold_init`` sets the firing threshold every LIF layer starts from and
    is the model's *operating-point* knob. It matters more than a hyper-parameter
    usually would, because an LIF stack initialised too far above its input
    current scale starts **silent**: measured on this task's data at the default
    1.0, the second LIF layer emits exactly zero spikes at every one of the 8
    unroll steps and the logits carry std 0.03, so training must first grow the
    synaptic weights until neurons cross threshold at all -- through an arctan
    surrogate whose mean gradient there is only ~0.10 per layer. Lowering it to
    0.25 puts both layers at a 8-10% firing rate with ~0.55 mean surrogate
    gradient from the first step. See ``docs/OPEN_QUESTIONS.md`` (federated-SNN
    diagnosis) for the measurements and the accuracy/SynOps trade-off this knob
    buys.
    """

    def __init__(
        self,
        in_ch: int = 11,
        n_classes: int = 8,
        hidden: int = 256,
        layers: int = 2,
        T_steps: int = 8,
        coding: Literal["rate", "delta"] = "rate",
        seed: int = 0,
        delta_threshold: float = 0.2,
        window_t: int = 64,
        beta_init: float = 0.9,
        threshold_init: float = 1.0,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError(f"need at least one LIF layer, got {layers}")
        if coding not in ("rate", "delta"):
            raise ValueError(f"coding must be 'rate' or 'delta', got {coding!r}")
        if coding == "delta" and window_t % T_steps != 0:
            raise ValueError(
                f"delta coding folds the window into T_steps chunks: "
                f"window_t={window_t} not divisible by T_steps={T_steps}"
            )
        self.in_ch = in_ch
        self.n_classes = n_classes
        self.hidden = hidden
        self.layers = layers
        self.T_steps = T_steps
        self.coding = coding
        self.seed = seed
        self.delta_threshold = delta_threshold
        self.window_t = window_t
        self.beta_init = beta_init
        self.threshold_init = threshold_init

        if coding == "rate":
            self.in_features = in_ch * window_t
        else:
            self.in_features = 2 * in_ch * (window_t // T_steps)

        self.fc_in = nn.Linear(self.in_features, hidden)
        self.lifs = nn.ModuleList(
            LIF(hidden, beta_init=beta_init, threshold_init=threshold_init)
            for _ in range(layers)
        )
        self.fc_hidden = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers - 1))
        self.readout = nn.Linear(hidden, n_classes)

        # Dense-MAC cost of one fully-active timestep (per sample): every synapse
        # of every Linear fires. SynOps can never exceed T_steps times this.
        self._dense_per_step = (
            self.in_features * hidden
            + (layers - 1) * hidden * hidden
            + hidden * n_classes
        )
        # Spike sources feeding synaptic fan-out (coding layer + each LIF layer);
        # denominator of spike_rate, so spike_rate and synops stay consistent.
        self._sources_per_step = self.in_features + layers * hidden

        self._synops: Tensor = torch.zeros((), dtype=torch.float64)
        self._spikes: Tensor = torch.zeros((), dtype=torch.float64)
        self._dense_macs: float = 0.0
        self._opportunities: float = 0.0

    # ---------------------------------------------------------------- coding
    def _encode(self, x: Tensor) -> Tensor:
        """``(B, C, T)`` -> spike train ``(T_steps, B, in_features)`` in {0, 1}."""
        b = x.shape[0]
        if self.coding == "rate":
            p = torch.sigmoid(x.reshape(b, -1))
            gen = torch.Generator(device=x.device)
            gen.manual_seed(self.seed)
            u = torch.rand(
                (self.T_steps, b, self.in_features),
                generator=gen,
                device=x.device,
                dtype=x.dtype,
            )
            return (u < p.unsqueeze(0)).to(x.dtype)
        # Temporal contrast: ON/OFF events on the first difference; step 0 has no
        # predecessor, hence the left zero-pad (no spurious onset event).
        d = torch.nn.functional.pad(x[:, :, 1:] - x[:, :, :-1], (1, 0))
        events = torch.cat(
            [(d > self.delta_threshold), (d < -self.delta_threshold)], dim=1
        ).to(x.dtype)  # (B, 2C, T)
        chunk = self.window_t // self.T_steps
        events = events.reshape(b, 2 * self.in_ch, self.T_steps, chunk)
        return events.permute(2, 0, 1, 3).reshape(self.T_steps, b, self.in_features)

    # --------------------------------------------------------------- forward
    def forward(self, x: Tensor) -> Tensor:
        """``(B, C, T) -> (B, n_classes)`` logits (rate-decoded)."""
        if x.dim() != 3 or x.shape[1] != self.in_ch or x.shape[2] != self.window_t:
            raise ValueError(
                f"expected (B, {self.in_ch}, {self.window_t}) input, got {tuple(x.shape)}"
            )
        b = x.shape[0]
        for lif in self.lifs:
            lif.reset_state()  # windows are independent — no state bleed
        spikes = self._encode(x)

        logit_acc = x.new_zeros((b, self.n_classes))
        # SynOps: spikes_in × fan_out per synaptic layer per timestep, gathered
        # on-device and folded into the float64 counters once per forward.
        synops_acc = x.new_zeros(())
        spike_acc = x.new_zeros(())
        for t in range(self.T_steps):
            s = spikes[t]
            with torch.no_grad():
                n_in = s.sum()
                synops_acc = synops_acc + n_in * self.hidden
                spike_acc = spike_acc + n_in
            cur = self.fc_in(s)
            for i, lif in enumerate(self.lifs):
                s = lif(cur)
                with torch.no_grad():
                    n_spk = s.detach().sum()
                    fan_out = self.hidden if i < self.layers - 1 else self.n_classes
                    synops_acc = synops_acc + n_spk * fan_out
                    spike_acc = spike_acc + n_spk
                if i < self.layers - 1:
                    cur = self.fc_hidden[i](s)
            logit_acc = logit_acc + self.readout(s)

        with torch.no_grad():
            self._synops = _accumulate(self._synops, synops_acc)
            self._spikes = _accumulate(self._spikes, spike_acc)
            self._dense_macs += float(b * self.T_steps * self._dense_per_step)
            self._opportunities += float(b * self.T_steps * self._sources_per_step)
        return logit_acc / self.T_steps

    # -------------------------------------------------------------- counters
    def synops(self) -> dict[str, float]:
        """``{'synops', 'dense_macs', 'spike_rate', 'ratio'}`` since :meth:`reset_counters`.

        ``synops``: accumulated spikes_in × fan_out over all synaptic layers and
        timesteps. ``dense_macs``: what the same topology would cost fully
        active (every synapse, every timestep) over the same inputs — so
        ``ratio = synops / dense_macs`` is in [0, 1] and equals the fan-out
        weighted activity. ``spike_rate``: emitted spikes / spike opportunities
        across the coding layer and every LIF layer.
        """
        synops = float(self._synops)
        dense = self._dense_macs
        return {
            "synops": synops,
            "dense_macs": dense,
            "spike_rate": float(self._spikes) / self._opportunities if self._opportunities else 0.0,
            "ratio": synops / dense if dense else 0.0,
        }

    def reset_counters(self) -> None:
        """Zero the accumulated spike/SynOps counters (call before each measured pass)."""
        self._synops = torch.zeros((), dtype=torch.float64)
        self._spikes = torch.zeros((), dtype=torch.float64)
        self._dense_macs = 0.0
        self._opportunities = 0.0
        for lif in self.lifs:
            lif.reset_counters()
