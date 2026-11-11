"""Load-shape and irradiance engine backing the feeder co-simulation. Realism basis (each parameter also tabulated in
``docs/REALISM.md``):

* **Residential base shapes** are the 100 measured-model 1-minute daily profiles
  shipped with the IEEE PES European LV Test Feeder (vendored under
  ``feeders/eulv/Daily_1min_100profiles``). The test-feeder documentation states
  they were produced with the CREST high-resolution stochastic demand model
  (Richardson, Thomson, Infield, Clifford, "Domestic electricity use: A
  high-resolution energy demand model", Energy and Buildings 42(10):1878-1887,
  2010) - i.e. realistic occupant-driven appliance-level profiles, not toys.
* **MV spot loads** aggregate ``n`` distinct household profiles (diversity
  smoothing), the standard treatment of coincidence/diversity in distribution
  modelling (Kersting, *Distribution System Modeling and Analysis*, CRC Press,
  ch. "Load models"; diversity factors).
* **Seasonal and day-type structure** is a fitted-synthetic modulation of the
  measured base shapes: an annual sinusoid with winter:summer energy ratio
  ~1.67 and a +5 % weekend uplift for domestic load, both fitted to the ELEXON
  Load Profile Class 1 (Domestic Unrestricted) settlement profiles (ELEXON,
  "Load Profiles and their use in Electricity Settlement"). Marked
  *fitted-synthetic* in REALISM.md.
* **Commercial shapes** are fitted-synthetic business-hours profiles fitted to
  the U.S. DOE/NREL Commercial Reference Building (medium office) simulated
  load data (Deru et al., NREL/TP-5500-46861, 2011): night base ~35 %,
  07-09 h ramp to plateau, 17-20 h decay, weekend plateau ~55 % of weekday.
* **Clear-sky irradiance** uses the Haurwitz model (Haurwitz, "Insolation in
  relation to cloudiness and cloud density", J. Meteorology 2:154-166, 1945),
  one of the best-performing simple GHI clear-sky models in the Sandia
  validation study (Reno, Hansen, Stein, SAND2012-2389).
* **Cloud transients** follow a two-state (clear/cloudy) Markov process whose
  bimodal clear-sky-index distribution mirrors measured 1-min statistics
  (Woyte, Belmans, Nijs, "Fluctuations in instantaneous clearness index",
  Solar Energy 81(2):195-206, 2007); minute-scale ramp depths of 60 %+ of
  clear-sky output are documented in Mills & Wiser, LBNL-3884E (2010) and
  Lave, Kleissl, Stein (IEEE J. Photovoltaics 3(1), 2013).

Day convention: ``day`` counts from 0 = Monday, January 1 (day-of-year and
day-of-week derive from it); weekend = day % 7 in {5, 6}.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = [
    "FEEDER_DIR",
    "clear_sky_ghi",
    "clear_sky_index",
    "commercial_shape",
    "irradiance_profile",
    "residential_library",
    "residential_shape",
    "seasonal_factor",
]

FEEDER_DIR = Path(__file__).resolve().parent / "feeders"
_EULV_PROFILES = FEEDER_DIR / "eulv" / "Daily_1min_100profiles"

# ELEXON Load Profile Class 1: winter:summer average-day energy ratio ~1.6-1.7
# -> sinusoid amplitude 0.25 gives (1+A)/(1-A) = 1.67. Winter peak in mid-January.
_SEASONAL_AMPLITUDE = 0.25
_SEASONAL_PEAK_DOY = 15
# ELEXON PC1: weekend domestic energy slightly above weekday (~+5 %).
_RES_WEEKEND_FACTOR = 1.05
# NREL/TP-5500-46861 medium office: weekend load ~half the weekday plateau.
_COM_WEEKEND_PLATEAU = 0.55
_COM_NIGHT_BASE = 0.35


def _minutes(n_steps: int, step_s: float) -> np.ndarray:
    return np.arange(n_steps, dtype=np.float64) * (step_s / 60.0)


@lru_cache(maxsize=2)
def residential_library(normalised: bool = True) -> np.ndarray:
    """(100, 1440) matrix of measured-model 1-min daily household profiles.

    Source data: IEEE PES EU LV Test Feeder ``Daily_1min_100profiles`` (CREST
    model output; Richardson et al. 2010). With ``normalised=True`` each
    profile is scaled to mean 1 so aggregates of n profiles can scale an MV
    spot load's nominal kW. With ``normalised=False`` the raw kW values are
    returned - the official EU LV case semantics (its loads are kW = 1 with
    the profile as actual demand), which preserves the measured-model peak
    magnitudes (single homes peak at ~10 kW) instead of inflating them by the
    peak-to-mean ratio.
    """
    files = sorted(_EULV_PROFILES.glob("load_profile_*.txt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    if len(files) != 100:
        raise FileNotFoundError(f"expected 100 EU LV profiles under {_EULV_PROFILES}")
    mat = np.stack([np.loadtxt(f, dtype=np.float64) for f in files])
    if mat.shape != (100, 1440):
        raise ValueError(f"unexpected profile matrix shape {mat.shape}")
    if not normalised:
        return mat
    means = mat.mean(axis=1, keepdims=True)
    means[means <= 0] = 1.0
    return mat / means


def seasonal_factor(day: int) -> float:
    """Annual demand modulation, winter-peaking sinusoid (ELEXON PC1 fit).

    Amplitude 0.25 reproduces the ~1.67 winter:summer average-day energy ratio
    of ELEXON Load Profile Class 1 (fitted-synthetic; see REALISM.md).
    """
    doy = day % 365
    return 1.0 + _SEASONAL_AMPLITUDE * np.cos(2.0 * np.pi * (doy - _SEASONAL_PEAK_DOY) / 365.0)


def _is_weekend(day: int) -> bool:
    return day % 7 in (5, 6)


def residential_shape(
    rng: np.random.Generator, day: int, n_aggregate: int = 1, n_steps: int = 1440,
    raw_kw: bool = False,
) -> np.ndarray:
    """Daily multiplier for one residential load (or aggregate of homes).

    Draws ``n_aggregate`` distinct CREST-derived profiles from the measured
    library (diversity smoothing per Kersting's treatment of coincident demand)
    and applies the ELEXON-fitted seasonal and weekend factors. Day-to-day
    variation comes from re-drawing library members - i.e. from measured-model
    diversity, not injected noise. ``raw_kw=True`` keeps the profiles' actual
    kW values (single-home semantics of the official EU LV case, applied to
    kW = 1 loads); the default mean-1 scaling suits MV spot-load aggregates.
    """
    lib = residential_library(normalised=not raw_kw)
    n_aggregate = int(np.clip(n_aggregate, 1, lib.shape[0]))
    idx = rng.choice(lib.shape[0], size=n_aggregate, replace=False)
    shape = lib[idx].mean(axis=0)
    if n_steps != shape.shape[0]:
        pos = np.linspace(0.0, shape.shape[0] - 1.0, n_steps)
        shape = np.interp(pos, np.arange(shape.shape[0]), shape)
    factor = seasonal_factor(day) * (_RES_WEEKEND_FACTOR if _is_weekend(day) else 1.0)
    return shape * factor


def commercial_shape(
    rng: np.random.Generator, day: int, n_steps: int = 1440, step_s: float = 60.0,
) -> np.ndarray:
    """Fitted-synthetic business-hours multiplier (NREL/TP-5500-46861 fit).

    Weekday: base 0.35 overnight, sigmoid ramp 07-09 h to plateau 1.0, decay
    17-20 h; weekend plateau 0.55 of weekday. A small AR(1) multiplicative
    texture (sigma 3 %) represents aggregate short-term variability (modelling
    choice, documented in REALISM.md).
    """
    h = _minutes(n_steps, step_s) / 60.0
    rise = 1.0 / (1.0 + np.exp(-(h - 8.0) / 0.5))
    fall = 1.0 / (1.0 + np.exp(-(h - 18.5) / 0.8))
    plateau = _COM_WEEKEND_PLATEAU if _is_weekend(day) else 1.0
    base = _COM_NIGHT_BASE + (plateau - _COM_NIGHT_BASE) * (rise - fall)
    ar = np.empty(n_steps)
    ar[0] = 0.0
    eps = rng.normal(0.0, 0.03 * np.sqrt(1 - 0.9**2), size=n_steps)
    for t in range(1, n_steps):
        ar[t] = 0.9 * ar[t - 1] + eps[t]
    return base * seasonal_factor(day) * (1.0 + ar)


def clear_sky_ghi(
    day: int, n_steps: int = 1440, step_s: float = 60.0, latitude_deg: float = 35.0,
) -> np.ndarray:
    """Clear-sky GHI in W/m^2, Haurwitz (1945) model.

    GHI_cs = 1098 * cos(Z) * exp(-0.057 / cos(Z)) with solar zenith Z from
    standard declination/hour-angle geometry (solar noon at 12:00 local).
    Validated among simple clear-sky models by Reno et al., SAND2012-2389.
    """
    doy = day % 365 + 1
    decl = np.deg2rad(23.45) * np.sin(2.0 * np.pi * (284 + doy) / 365.0)
    lat = np.deg2rad(latitude_deg)
    hour = _minutes(n_steps, step_s) / 60.0
    hour_angle = np.deg2rad(15.0 * (hour - 12.0))
    cos_z = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    cos_z = np.clip(cos_z, 0.0, 1.0)
    ghi = np.zeros(n_steps)
    up = cos_z > 1e-3
    ghi[up] = 1098.0 * cos_z[up] * np.exp(-0.057 / cos_z[up])
    return ghi


def clear_sky_index(
    rng: np.random.Generator, n_steps: int = 1440, step_s: float = 60.0,
) -> np.ndarray:
    """Two-state Markov clear-sky index k_t, bimodal per Woyte et al. (2007).

    Mean sojourn 40 min clear / 20 min cloudy (cloud fraction ~1/3); clear-state
    k_t ~ N(0.99, 0.02) clipped to [0.90, 1.05] (thin-haze scatter), cloudy-state
    episode depth ~ U(0.15, 0.60) with 8 % within-episode jitter. State edges are
    linearly smoothed over ~2 min, matching documented minute-scale ramp depths
    of 60 %+ of clear-sky output (Mills & Wiser LBNL-3884E; Lave et al. 2013).
    """
    p_leave_clear = step_s / (40.0 * 60.0)
    p_leave_cloud = step_s / (20.0 * 60.0)
    kt = np.empty(n_steps)
    cloudy = bool(rng.random() < 1.0 / 3.0)
    depth = rng.uniform(0.15, 0.60)
    for t in range(n_steps):
        if cloudy:
            kt[t] = np.clip(depth * (1.0 + 0.08 * rng.standard_normal()), 0.05, 0.9)
            if rng.random() < p_leave_cloud:
                cloudy = False
        else:
            kt[t] = np.clip(0.99 + 0.02 * rng.standard_normal(), 0.90, 1.05)
            if rng.random() < p_leave_clear:
                cloudy = True
                depth = rng.uniform(0.15, 0.60)
    smooth_n = max(1, int(round(120.0 / step_s)))
    if smooth_n > 1:
        kernel = np.ones(smooth_n) / smooth_n
        kt = np.convolve(kt, kernel, mode="same")
    return kt


def irradiance_profile(
    rng: np.random.Generator,
    day: int,
    n_steps: int = 1440,
    step_s: float = 60.0,
    latitude_deg: float = 35.0,
) -> np.ndarray:
    """Plane irradiance in per-unit of 1 kW/m^2 (= OpenDSS PVSystem irradiance).

    Haurwitz clear-sky envelope x Markov clear-sky index. The result drives the
    PVSystem daily shape; 1.0 corresponds to STC irradiance.
    """
    ghi = clear_sky_ghi(day, n_steps, step_s, latitude_deg)
    kt = clear_sky_index(rng, n_steps, step_s)
    return np.clip(ghi * kt / 1000.0, 0.0, 1.2)
