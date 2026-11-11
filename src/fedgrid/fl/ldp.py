"""Local differential privacy: clip-and-noise on client updates, epsilon accounting.

The paper's point is the tension: LDP noise makes honest clients look Byzantine
to fidelity scoring, so the noise/threshold trade-off curve is a contribution.
LDP here is a *client-side privacy* mechanism applied by honest participants to
their own update before submission — it is never a defense-side bound on
malicious updates (see :mod:`fedgrid.fl.engine` for the pipeline ordering).

Privacy accounting (:func:`epsilon`) follows Renyi differential privacy:

* Mironov, "Renyi Differential Privacy", IEEE CSF 2017,
  DOI 10.1109/CSF.2017.11 (arXiv:1702.07476) — Corollary 3: the Gaussian
  mechanism with noise std ``sigma * Delta`` on a sensitivity-``Delta`` query
  satisfies ``(alpha, alpha / (2 sigma^2))``-RDP; Proposition 1: RDP composes
  additively over adaptive rounds; Proposition 3: ``(alpha, eps_a)``-RDP
  implies ``(eps_a + log(1/delta) / (alpha - 1), delta)``-DP.

Reference validation (asserted in ``tests/test_fl.py``): Dwork & Roth, "The
Algorithmic Foundations of Differential Privacy", 2014,
DOI 10.1561/0400000042 — Theorem A.1: for ``eps < 1`` the Gaussian mechanism
with ``sigma = Delta * sqrt(2 ln(1.25/delta)) / eps`` is ``(eps, delta)``-DP.
At ``delta = 1e-5``, ``eps = 0.5`` this calibration gives
``sigma = 9.689610525210778``; this accountant returns
``epsilon(9.689610525210778, 1, 1e-5) = 0.5005492801617069`` — 0.11 % above
the published 0.5 (both are valid upper bounds on the true privacy loss; the
tests pin these exact digits, check the closed form against an independent
numeric grid minimum over alpha, and check the exact Gaussian composition
identity ``epsilon(sigma, T, delta) == epsilon(sigma / sqrt(T), 1, delta)``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["LDPConfig", "clip_and_noise", "epsilon"]


@dataclass
class LDPConfig:
    """Clip norm and relative noise scale for the Gaussian mechanism."""

    clip: float
    sigma: float
    seed: int = 0


def clip_and_noise(
    params: list[np.ndarray],
    ref: list[np.ndarray],
    cfg: LDPConfig,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """L2-clip the update (``params - ref``) to ``cfg.clip``, add N(0, sigma^2 * clip^2).

    The update is treated as one vector across all arrays: a single global L2
    norm decides the scaling factor ``min(1, clip / ||delta||)`` (Abadi et al.,
    CCS 2016, DOI 10.1145/2976749.2978318, Eq. for per-example clipping —
    here applied per-client to the whole model update). Noise std is
    ``cfg.sigma * cfg.clip`` (relative noise-multiplier convention), drawn
    i.i.d. per coordinate from ``rng`` in list order; no draws are consumed
    when ``sigma == 0`` so a noiseless config stays stream-compatible.
    Arithmetic runs in float64 and is cast back to each input array's dtype.
    Returns FULL parameters (``ref + clipped delta + noise``), mirroring the
    input structure — the engine trades full params, not deltas.
    """
    if cfg.clip <= 0:
        raise ValueError(f"LDPConfig.clip must be > 0, got {cfg.clip}")
    if cfg.sigma < 0:
        raise ValueError(f"LDPConfig.sigma must be >= 0, got {cfg.sigma}")
    if len(params) != len(ref):
        raise ValueError(f"params/ref length mismatch: {len(params)} != {len(ref)}")

    deltas = [
        np.asarray(p, dtype=np.float64) - np.asarray(r, dtype=np.float64)
        for p, r in zip(params, ref, strict=True)
    ]
    norm = math.sqrt(sum(float((d * d).sum()) for d in deltas))
    scale = 1.0 if norm <= cfg.clip else cfg.clip / norm

    out: list[np.ndarray] = []
    for p, r, d in zip(params, ref, deltas, strict=True):
        v = np.asarray(r, dtype=np.float64) + scale * d
        if cfg.sigma > 0:
            v = v + rng.normal(0.0, cfg.sigma * cfg.clip, size=v.shape)
        out.append(v.astype(np.asarray(p).dtype))
    return out


def epsilon(sigma: float, rounds: int, delta: float = 1e-5) -> float:
    """(eps, delta)-DP guarantee of ``rounds`` Gaussian releases at noise multiplier ``sigma``.

    Every round releases the (clipped, sensitivity = ``clip``) client update
    with noise std ``sigma * clip``, i.e. noise multiplier ``sigma`` — so each
    round is ``(alpha, alpha / (2 sigma^2))``-RDP (Mironov 2017, Cor. 3), the
    ``rounds``-fold adaptive composition is ``(alpha, rounds * alpha /
    (2 sigma^2))``-RDP (Prop. 1), and conversion (Prop. 3) gives

        eps(alpha) = rounds * alpha / (2 sigma^2) + log(1/delta) / (alpha - 1).

    Minimising over ``alpha > 1`` in closed form (``c := rounds / (2 sigma^2)``,
    ``L := log(1/delta)``; the minimum sits at ``alpha* = 1 + sqrt(L / c)``):

        eps = c + 2 * sqrt(c * L)

    Full client participation each round — no subsampling amplification is
    (or may be) assumed. ``sigma <= 0`` yields ``inf``: no noise, no guarantee.
    See the module docstring for the published reference value this is
    validated against (Dwork & Roth 2014, Thm. A.1).
    """
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if sigma <= 0:
        return math.inf
    c = rounds / (2.0 * sigma * sigma)
    ell = math.log(1.0 / delta)
    return c + 2.0 * math.sqrt(c * ell)
