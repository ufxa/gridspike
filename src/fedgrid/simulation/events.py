"""Event plans: what anomaly hits the feeder, when, and with which parameters. Parameter distributions (each with a row in
``docs/REALISM.md``):

* **Fault resistance** - single-line-to-ground faults span bolted to
  high-impedance: log-uniform 0.05-40 ohm, the upper end per Dagenhart,
  "The 40-ohm ground-fault phenomenon", IEEE Trans. Industry Applications
  36(1):30-32, 2000; typical expected values across fault types (phase faults
  well below ground faults) per De Andrade & Sorrentino, "Typical expected
  values of the fault resistance in power systems", IEEE PES T&D-LA 2010.
  Phase-phase / multi-phase faults: log-uniform 0.01-1.5 ohm (metallic to
  low-resistance arcs).
* **Fault location** - uniform along total conductor length: a line is drawn
  with probability proportional to its length (exposure ~ length; Short,
  *Electric Power Distribution Handbook*, CRC 2004), then discretised to the
  nearer terminal bus (OpenDSS Fault objects attach to buses; documented as a
  bus-granular approximation in REALISM.md).
* **Fault episode duration** - 2-15 min sustained episodes. Sub-minute
  breaker/fuse clearing (IEEE Std C37.104 reclosing practice) is below the
  60-s telemetry cadence, so modelled fault episodes represent the conditions
  that *are* observable in 1-min RMS trend data: high-impedance and repeated-
  reclose/lockout conditions, which persist for minutes (Ghaderi, Ginn,
  Mohammadpour, "High impedance fault detection: A review", Electric Power
  Systems Research 143:376-388, 2017). Stated as a cadence-constrained
  approximation in REALISM.md - honesty over convenience.
* **SAG / SWELL** - magnitude bands per IEEE Std 1159-2019 sustained
  undervoltage (0.8-0.9 pu) and overvoltage (1.1-1.2 pu) categories, the
  applicable categories at >1 min duration; origin: remote-fault/source-side
  depressions (Bollen, *Understanding Power Quality Problems*, IEEE Press
  2000). Every sag is realised by the same mechanism - a source-side voltage
  excursion, the source per-unit held at ``depth_pu`` for the episode. The
  ``mech_pv`` flag does not replace that mechanism, it superposes on it: on a
  feeder carrying PV (``pv_penetration > 0``) it additionally collapses
  irradiance to 0.1x for the episode and confines scheduling to the
  09:30-15:30 solar window, so the metering bus also sees local PV infeed
  vanish (imported power *rises* during the dip) - the signature reported on
  high-penetration feeders (Katiraei & Agüero, IEEE Power & Energy Magazine
  9(3):62-71, 2011). No sag here is PV-*caused*; PV only changes how it looks.
* **OVERLOAD** - feeder-wide load multiplier 1.4-2.0x for 5-45 min: the
  *sustained* component of cold-load pickup after restoration (Short 2004;
  IEEE Std C37.104 - the documented 2-5x magnitudes are the seconds-scale
  inrush, decaying to ~1.5-2x at the minutes scale modelled here). Feeders
  whose voltage regime cannot serve the full pickup at peak clamp it
  (sectionalized restoration; per-feeder caps documented in REALISM.md).
* **Event rate** - Poisson(1.4/day), tuned so the window-level class prior
  stays NORMAL >= 90 % (contract realism block; verified by test).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fedgrid.data import EventClass

__all__ = ["EventSpec", "sample_event_plan"]

# Poisson event rate per simulated day; with the T=64/stride-16 windowing this
# yields ~6-8 % non-NORMAL windows -> NORMAL prior 92-94 % (>= 90 % contract bar).
RATE_PER_DAY: float = 1.4
_MIN_GAP_S: float = 180.0  # inter-event guard so labels never overlap
_FAULT_CLASSES = (
    EventClass.FAULT_LG, EventClass.FAULT_LL, EventClass.FAULT_LLG, EventClass.FAULT_3PH,
)


@dataclass
class EventSpec:
    """One scheduled event inside a simulated day."""

    cls: EventClass
    start_s: float
    duration_s: float
    params: dict[str, float]  # fault resistance, load multiplier, sag depth, ...


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _duration_s(rng: np.random.Generator, cls: EventClass) -> float:
    """Class-coupled episode durations (sources in module docstring)."""
    if cls == EventClass.FAULT_LG:
        return float(np.clip(rng.lognormal(np.log(180.0), 0.6), 120.0, 900.0))
    if cls in (EventClass.FAULT_LL, EventClass.FAULT_LLG, EventClass.FAULT_3PH):
        return float(np.clip(rng.lognormal(np.log(150.0), 0.5), 120.0, 600.0))
    if cls in (EventClass.SAG, EventClass.SWELL):
        return float(rng.uniform(120.0, 600.0))  # IEEE 1159 sustained-variation regime
    if cls == EventClass.OVERLOAD:
        return float(np.clip(rng.lognormal(np.log(900.0), 0.5), 300.0, 2700.0))
    raise ValueError(f"no duration model for {cls!r}")


def _params(rng: np.random.Generator, cls: EventClass) -> dict[str, float]:
    """Per-class physical parameters (sources in module docstring / REALISM.md)."""
    if cls in _FAULT_CLASSES:
        if cls == EventClass.FAULT_LG:
            r = _log_uniform(rng, 0.05, 40.0)   # bolted..high-impedance (Dagenhart 2000)
        else:
            r = _log_uniform(rng, 0.01, 1.5)    # metallic phase faults (De Andrade 2010)
        return {
            "r_ohm": r,
            "loc_frac": float(rng.uniform(0.0, 1.0)),  # uniform along conductor length
            "phase": float(rng.integers(1, 4)),
        }
    if cls == EventClass.SAG:
        # Source-side target chosen so that, after the feeders' own voltage
        # regulators partially compensate (controlmode=time tap action), the
        # observed metering-bus depth lands predominantly in the IEEE 1159
        # sustained-undervoltage band (0.8-0.9 pu).
        return {
            "depth_pu": float(rng.uniform(0.75, 0.88)),
            "mech_pv": float(rng.integers(0, 2)),  # 0 = source-side, 1 = PV transient
        }
    if cls == EventClass.SWELL:
        # Source-side target for the IEEE 1159 overvoltage band (1.1-1.2 pu)
        # at the metering bus, accounting for regulator setpoint offsets.
        return {"level_pu": float(rng.uniform(1.08, 1.15))}
    if cls == EventClass.OVERLOAD:
        # Sustained (minutes-scale) cold-load-pickup component: the 2-5x figures
        # in the literature are the seconds-scale inrush; the enduring level
        # relevant at >= 5-min episodes is ~1.5-2x (Short 2004, cold-load
        # pickup; IEEE Std C37.104). Capacity-constrained feeders additionally
        # clamp this in grid.py (sectionalized-restoration practice).
        return {"mult": float(rng.uniform(1.4, 2.0))}
    raise ValueError(f"no parameter model for {cls!r}")


def sample_event_plan(
    rng: np.random.Generator, day_len_s: float, class_weights: np.ndarray
) -> list[EventSpec]:
    """Draw a non-overlapping event schedule whose class mix follows ``class_weights``.

    ``class_weights`` is the client's Dirichlet draw over the 8 classes, so the
    plan realises the federation's label skew at the physics level rather than
    by subsampling: event classes are drawn from ``class_weights[1:]``
    (renormalised); the NORMAL share is governed by the global event rate.
    PV-mechanism sags are scheduled inside the 09:30-15:30 solar window.
    """
    w = np.asarray(class_weights, dtype=np.float64)
    if w.shape[0] != len(EventClass):
        raise ValueError(f"class_weights must have {len(EventClass)} entries")
    we = np.clip(w[1:], 0.0, None)
    if we.sum() <= 0.0:
        return []
    we = we / we.sum()
    n_events = int(rng.poisson(RATE_PER_DAY * day_len_s / 86400.0))
    taken: list[tuple[float, float]] = []
    plan: list[EventSpec] = []
    for _ in range(n_events):
        cls = EventClass(int(rng.choice(np.arange(1, len(EventClass)), p=we)))
        dur = min(_duration_s(rng, cls), day_len_s / 2.0)
        params = _params(rng, cls)
        solar = cls == EventClass.SAG and params.get("mech_pv", 0.0) >= 1.0
        lo = 9.5 * 3600.0 if solar else 0.0
        hi = min(15.5 * 3600.0, day_len_s - dur) if solar else day_len_s - dur
        if hi <= lo:
            continue
        for _attempt in range(200):
            start = float(rng.uniform(lo, hi))
            span = (start - _MIN_GAP_S, start + dur + _MIN_GAP_S)
            if all(span[1] <= a or span[0] >= b for a, b in taken):
                taken.append((start, start + dur))
                plan.append(EventSpec(cls=cls, start_s=start, duration_s=dur, params=params))
                break
    plan.sort(key=lambda e: e.start_s)
    return plan
