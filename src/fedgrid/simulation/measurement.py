"""Merging-unit measurement chain and system-frequency process. The classifier must see what a merging unit sees, not
clean solver output (contract amendment 2026-07-28b). Every parameter has a
source here and a row in ``docs/REALISM.md``:

* **CT/PT accuracy class 0.5** - IEC 61869-2 (current transformers) and
  IEC 61869-3 (inductive voltage transformers) bound the ratio error of a
  class-0.5 instrument to +/-0.5 % at rated signal. Modelled as a fixed
  per-instrument calibration bias ~ U(-0.3 %, +0.3 %) plus zero-mean Gaussian
  noise sigma = 0.05 % truncated at +/-0.2 %, so the *total* applied error is
  bounded by 0.5 % by construction (a conforming instrument stays inside its
  class; the test bar checks this over 1e4 samples).
* **P/Q metering** - active/reactive channels carry the combined CT+VT+angle
  error of a class-0.5S metering chain per IEC 62053-22 (0.5 % limit); modelled
  with the same bounded bias+noise envelope.
* **Frequency estimate** - IEEE C37.118.1-2011 bounds steady-state frequency
  error (FE) to 0.005 Hz for P/M-class PMUs; modelled as N(0, 1 mHz) truncated
  at +/-5 mHz.
* **ADC quantisation** - 16-bit uniform quantisation over fixed full-scale
  ranges, the conservative end of merging-unit front ends (IEC 61869-9 /
  IEC 61850-9-2LE sampled-value practice).
* **Sensor dropout** - communication-loss episodes with hold-last-value
  behaviour (stale-data semantics of RTU/MU buffers). Episode rate 0.5/day of
  mean length 3 samples ~ 0.1 % unavailable samples, inside the 99-99.9 %
  availability range reported for real synchrophasor networks (NASPI Technical
  Report, "Synchrophasor Data Quality", 2017).
* **Frequency wander** - stationary AR(1)/Ornstein-Uhlenbeck wander with
  sigma = 18 mHz and 4-min relaxation, consistent with measured grid-frequency
  statistics (Rydin Gorjao et al., "Open database analysis of scaling and
  spatio-temporal properties of power grid frequencies", Nature Communications
  11:6362, 2020) and with the ENTSO-E Continental Europe +/-50 mHz standard
  frequency range (ENTSO-E Operation Handbook, Policy 1). Event-coupled
  deviations add class-dependent dips/rises of 5-120 mHz (``_EVENT_FDEV``:
  faults -20..-120 mHz, overload -10..-50 mHz, sag -5..-20 mHz, swell
  +10..+50 mHz), the scale of documented deterministic frequency deviations
  (same sources); fitted-synthetic, tabulated. Wander plus event coupling is
  clipped to +/-0.3 Hz on the way out - see :func:`frequency_deviation`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fedgrid.data import CHANNELS, EventClass

__all__ = ["MeasurementChain", "MeasurementSpec", "frequency_deviation"]

# Full-scale ADC ranges per channel, in the channel's own unit (pu; Hz for fdev).
# Voltage channels: 2 pu. Current channels: 40 pu of nominal-load base - bolted
# fault currents reach tens of times load current and merging units digitise
# the protection-core signal, whose accuracy range extends to ~20x CT rated
# current (IEC 61869-2 protective classes, e.g. 5P20, on a CT rated ~2x load);
# clipping fault signatures at metering full-scale would destroy the classes
# we must label. Power follows (+/-16 pu; PV backfeed makes p negative).
# fdev: +/-0.5 Hz. The 16-bit LSB on the wide current range (~6e-4 pu) is of
# the same order as the class-0.5 noise floor at nominal signal - the honest
# resolution cost of a wide-range 16-bit front end.
_ADC_RANGE: dict[str, tuple[float, float]] = {
    "va": (0.0, 2.0), "vb": (0.0, 2.0), "vc": (0.0, 2.0),
    "ia": (0.0, 40.0), "ib": (0.0, 40.0), "ic": (0.0, 40.0),
    "p": (-16.0, 16.0), "q": (-16.0, 16.0),
    "v0": (0.0, 2.0), "i0": (0.0, 40.0),
    "fdev": (-0.5, 0.5),
}
# Channels carrying instrument-class multiplicative error (all but fdev).
_RATIO_CHANNELS = tuple(c for c in CHANNELS if c != "fdev")


@dataclass(frozen=True)
class MeasurementSpec:
    """Parameters of the measurement chain; defaults = IEC class 0.5 envelope."""

    ratio_bias_max: float = 0.003     # fixed calibration bias bound (0.3 %)
    ratio_noise_sigma: float = 0.0005  # per-sample noise sigma (0.05 %)
    ratio_noise_clip: float = 0.002    # truncation (0.2 %); bias+noise <= 0.5 %
    class_limit: float = 0.005         # IEC 61869 class 0.5 ratio-error limit
    freq_noise_sigma: float = 0.001    # 1 mHz (IEEE C37.118.1 FE << 5 mHz)
    freq_noise_clip: float = 0.005     # 5 mHz FE limit
    adc_bits: int = 16                 # IEC 61869-9 / 9-2LE practice (conservative)
    dropout_per_day: float = 0.5       # episodes/day (NASPI availability 99-99.9 %)
    dropout_mean_len: int = 3          # samples; hold-last-value semantics
    dropout_max_len: int = 10


class MeasurementChain:
    """Applies instrument bias, noise, quantisation and dropout to clean channels.

    The per-instrument calibration biases are drawn once from ``bias_rng`` (they
    belong to the substation's installed hardware, so the caller seeds them per
    client, not per day); per-sample noise and dropout use the ``rng`` passed to
    :meth:`apply`.
    """

    def __init__(self, bias_rng: np.random.Generator,
                 spec: MeasurementSpec | None = None) -> None:
        self.spec = spec or MeasurementSpec()
        self.bias = {
            ch: float(bias_rng.uniform(-self.spec.ratio_bias_max, self.spec.ratio_bias_max))
            for ch in _RATIO_CHANNELS
        }

    def apply(self, clean: np.ndarray, rng: np.random.Generator,
              dropout: bool = True) -> np.ndarray:
        """(n_steps, 11) clean channel matrix -> measured matrix, same shape.

        Guarantees |measured/true - 1| <= ``class_limit`` on every ratio channel
        outside dropout episodes (dropout replays stale values instead).
        """
        s = self.spec
        n = clean.shape[0]
        out = np.array(clean, dtype=np.float64, copy=True)
        for j, ch in enumerate(CHANNELS):
            if ch == "fdev":
                err = np.clip(rng.normal(0.0, s.freq_noise_sigma, n),
                              -s.freq_noise_clip, s.freq_noise_clip)
                out[:, j] = out[:, j] + err
            else:
                noise = np.clip(rng.normal(0.0, s.ratio_noise_sigma, n),
                                -s.ratio_noise_clip, s.ratio_noise_clip)
                out[:, j] = out[:, j] * (1.0 + self.bias[ch] + noise)
            lo, hi = _ADC_RANGE[ch]
            lsb = (hi - lo) / float(2 ** s.adc_bits)
            out[:, j] = np.clip(np.round((out[:, j] - lo) / lsb) * lsb + lo, lo, hi)
        if dropout:
            for t0, t1 in self._dropout_episodes(n, rng):
                held = out[max(t0 - 1, 0), :].copy()
                out[t0:t1, :] = held
        return out

    def _dropout_episodes(self, n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
        n_ep = int(rng.poisson(self.spec.dropout_per_day * n / 1440.0))
        eps: list[tuple[int, int]] = []
        for _ in range(n_ep):
            length = int(np.clip(rng.geometric(1.0 / self.spec.dropout_mean_len),
                                 1, self.spec.dropout_max_len))
            t0 = int(rng.integers(1, max(2, n - length)))
            eps.append((t0, min(t0 + length, n)))
        return eps


# Event-coupled frequency deviation scales in Hz (fitted-synthetic; anchored to
# the ENTSO-E +/-50 mHz standard range and the ~100 mHz scale of documented
# deterministic frequency deviations - see module docstring and REALISM.md).
_EVENT_FDEV: dict[int, tuple[float, float]] = {
    int(EventClass.FAULT_LG): (-0.08, -0.02),
    int(EventClass.FAULT_LL): (-0.08, -0.02),
    int(EventClass.FAULT_LLG): (-0.10, -0.03),
    int(EventClass.FAULT_3PH): (-0.12, -0.04),
    int(EventClass.OVERLOAD): (-0.05, -0.01),
    int(EventClass.SAG): (-0.02, -0.005),
    int(EventClass.SWELL): (0.01, 0.05),   # load-rejection over-frequency
}
_AR_TAU_S = 240.0     # relaxation time (minutes-scale autocorrelation; Gorjao 2020)
_AR_SIGMA_HZ = 0.018  # stationary std ~18 mHz (CE distributions; +/-50 mHz range)


def frequency_deviation(
    rng: np.random.Generator,
    labels: np.ndarray,
    step_s: float = 60.0,
    events: Sequence[tuple[int, int, int]] = (),
) -> np.ndarray:
    """System frequency deviation (Hz) per sample: AR(1) wander + event coupling.

    ``labels`` gives the per-sample event class (0 = NORMAL) - only its length
    is used for the wander; ``events`` is [(start_idx, end_idx, cls), ...] and
    drives the event-coupled component. For each event one deviation ``dev`` is
    drawn uniformly from that class's band in ``_EVENT_FDEV`` (classes absent
    from the table, NORMAL included, contribute nothing) and applied as a
    **step**: the full ``dev`` is present at the first in-event sample and held
    flat through ``end - 1``. It does not ramp in. At the 60-s cadence the
    inertial/governor transient that would set a ramp is over well inside one
    sample, so a ramp here would be sub-sample fiction; a step is the honest
    rendering of what 1-min RMS telemetry actually resolves.

    Recovery after clearance is a **fixed 8-sample tail**: samples
    ``end .. end + 7`` receive ``dev * exp(-(t - end + 1) / 2)``, so the tail
    opens at ``exp(-1/2) ~ 0.61 dev`` (clearance is itself a step *down*),
    halves every ~1.4 samples, and is cut to exactly zero after the eighth
    sample rather than continued - by then it is ``exp(-4) ~ 1.8 %`` of ``dev``,
    far below the wander. The tail is also truncated at the array end, so an
    event clearing near the end of the day carries a shorter tail (none at all
    if it runs to ``n``); nothing wraps around. Overlapping events accumulate,
    tails included.

    The sum is finally **clipped to +/-0.3 Hz**. That keeps the channel inside
    the +/-0.5 Hz ``fdev`` ADC range with headroom for the measurement error
    added downstream, so quantisation rather than saturation is the front end's
    limiting effect. With the tabulated bands (``|dev| <= 120 mHz``) on top of
    18 mHz-sigma wander no single event can reach the rail; it binds only on
    stacked/overlapping event windows, whose deviations add, or if a caller
    supplies wider bands.
    """
    n = int(labels.shape[0])
    phi = float(np.exp(-step_s / _AR_TAU_S))
    sig_eps = _AR_SIGMA_HZ * np.sqrt(1.0 - phi * phi)
    wander = np.empty(n)
    wander[0] = rng.normal(0.0, _AR_SIGMA_HZ)
    eps = rng.normal(0.0, sig_eps, n)
    for t in range(1, n):
        wander[t] = phi * wander[t - 1] + eps[t]
    coupled = np.zeros(n)
    for start, end, cls in events:
        band = _EVENT_FDEV.get(int(cls))
        if band is None:
            continue
        dev = rng.uniform(*sorted(band))
        start, end = max(0, int(start)), min(n, int(end))
        if end <= start:
            continue
        coupled[start:end] += dev
        # exponential recovery after clearance
        tail = np.arange(end, min(n, end + 8))
        coupled[tail] += dev * np.exp(-(tail - end + 1) / 2.0)
    return np.clip(wander + coupled, -0.3, 0.3)
