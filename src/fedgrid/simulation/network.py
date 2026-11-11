"""Utility-WAN latency model for the PBFT consensus simulation. This is the ``net_ms`` sampler the contract's realism
block requires ("PBFT net_ms drawn from a realistic utility-WAN latency
distribution (documented lognormal fit), not uniform toy delays"). the consensus/attacks module's
``consensus.pbft.PBFTSim`` takes it as its ``net_ms`` callable; it lives here
because the realism table (docs/REALISM.md) is owned by the data-simulation module. Flagged for
a shared helper if a local import is preferred.

Model and sources:

* One-way message latency ~ Lognormal(median 25 ms, sigma_ln 0.5), truncated to
  [5 ms, 200 ms]. Internet/WAN round-trip times are classically well described
  by lognormal-like heavy-tailed distributions (Aikat, Kaur, Smith, Jeffay,
  "Variability in TCP round-trip times", ACM IMC 2003).
* The location/scale is set so p50 ~ 25 ms and p99 < ~100 ms, inside the
  latency budget IEC/TR 61850-90-12 (Wide-area network engineering guidelines)
  assigns to inter-substation wide-area communication classes (tens of ms
  transfer time, 100 ms-class applications).
"""

from __future__ import annotations

import numpy as np

__all__ = ["WAN_LATENCY_MEDIAN_MS", "WAN_LATENCY_SIGMA_LN", "utility_wan_net_ms"]

WAN_LATENCY_MEDIAN_MS: float = 25.0
WAN_LATENCY_SIGMA_LN: float = 0.5
_CLIP_MS = (5.0, 200.0)


def utility_wan_net_ms(rng: np.random.Generator) -> float:
    """One-way utility-WAN message latency in milliseconds (lognormal fit).

    Lognormal(mu=ln 25, sigma=0.5) truncated to [5, 200] ms: p50 = 25 ms,
    p95 ~ 57 ms, p99 ~ 80 ms - consistent with the IEC/TR 61850-90-12 WAN
    class budgets; lognormal form per Aikat et al. (IMC 2003).
    """
    ms = float(rng.lognormal(mean=np.log(WAN_LATENCY_MEDIAN_MS), sigma=WAN_LATENCY_SIGMA_LN))
    return float(np.clip(ms, *_CLIP_MS))
