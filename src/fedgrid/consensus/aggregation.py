"""Byzantine-robust aggregation over the admitted update set.

Two families, both selected for the paper's ablation because they fail in
opposite ways: Krum *selects* (one honest-looking client wins, so a Byzantine
coalition must beat the neighbourhood statistic to do any damage at all, but a
single client's data drives the round) while the geometric median *averages*
(every admitted client contributes, so a coalition that survives admission moves
the result continuously).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fedgrid.consensus.fidelity import _stack, unflatten

if TYPE_CHECKING:
    from fedgrid.fl.engine import ClientUpdate

__all__ = ["geometric_median", "krum", "krum_scores", "multi_krum"]


def _neighbourhood(n: int, f: int) -> int:
    """Number of nearest neighbours entering a Krum score, clamped to what exists.

    Blanchard's analysis needs ``n >= 2f + 3`` so that ``n - f - 2`` neighbours can
    all be honest. Below that the guarantee is void; rather than raising mid-round
    we clamp to ``[1, n-1]`` and degrade gracefully — the caller (the pipeline) is
    the place where a too-small admitted set is a policy decision, not here.
    Callers guarantee ``n >= 2``: a lone update has no neighbourhood to score.
    """
    return int(np.clip(n - f - 2, 1, n - 1))


def _pairwise_sq_dists(mat: np.ndarray) -> np.ndarray:
    """(n, n) squared Euclidean distances, computed row-wise.

    The Gram-matrix identity is faster but subtracts two large nearly-equal
    numbers: client parameter vectors share the global model, so ``|u_i|^2`` is
    orders of magnitude above ``|u_i - u_j|^2`` and the cancellation eats most of
    the mantissa. Row-wise differencing costs the same asymptotics for the client
    counts we run (K = 10) and is exact.
    """
    n = mat.shape[0]
    out = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        diff = mat - mat[i]
        out[i] = np.einsum("ij,ij->i", diff, diff)
    return out


def krum_scores(updates: list[ClientUpdate], f: int) -> np.ndarray:
    """Krum score per update: sum of squared distances to its ``n-f-2`` nearest peers.

    Blanchard, El Mhamdi, Guerraoui and Stainer, "Machine Learning with
    Adversaries: Byzantine Tolerant Gradient Descent", NeurIPS 2017 (§3). Low
    score = sits inside the dense honest cluster.
    """
    if not updates:
        return np.zeros(0, dtype=np.float64)
    mat = _stack(updates)
    n = mat.shape[0]
    if n == 1:
        return np.zeros(1, dtype=np.float64)
    d2 = _pairwise_sq_dists(mat)
    np.fill_diagonal(d2, np.inf)  # a vector is never its own neighbour
    k = _neighbourhood(n, f)
    nearest = np.sort(d2, axis=1)[:, :k]
    return nearest.sum(axis=1)


def krum(updates: list[ClientUpdate], f: int) -> int:
    """Index of the single update with minimal Krum score under ``f`` Byzantine.

    Ties break to the lowest index, which keeps the function a pure deterministic
    function of the update list (the engine must be bit-reproducible).
    """
    if not updates:
        raise ValueError("krum() needs at least one update")
    return int(np.argmin(krum_scores(updates, f)))


def multi_krum(updates: list[ClientUpdate], f: int, m: int) -> list[int]:
    """Indices of the ``m`` best-scored updates (averaged downstream).

    Multi-Krum as published: score once, take the ``m`` smallest. (The original
    paper also describes an iterative variant that re-scores after each removal;
    the single-pass form is what the reference implementations use and it is what
    the ``m``-argument signature describes.) ``m`` is clamped to ``[1, n]``.
    """
    if not updates:
        raise ValueError("multi_krum() needs at least one update")
    n = len(updates)
    m = int(np.clip(m, 1, n))
    scores = krum_scores(updates, f)
    # Stable sort so equal scores keep client order — determinism again.
    order = np.argsort(scores, kind="stable")
    return [int(i) for i in order[:m]]


def geometric_median(
    updates: list[ClientUpdate], eps: float = 1e-6, max_iter: int = 100
) -> list[np.ndarray]:
    """Weiszfeld iteration on the flattened updates, reshaped back to parameter lists.

    Minimises the sum of Euclidean distances, so up to half the mass can be
    arbitrary before the estimate is dragged out of the honest hull. Two
    departures from the textbook iteration, both necessary:

    * the start point is the coordinate-wise median, not the mean — the mean is
      already broken by one arbitrarily large Byzantine vector, and Weiszfeld
      only converges *locally* for badly placed starts;
    * the Vardi-Zhang modification handles the singularity when the iterate lands
      exactly on a data point (Vardi and Zhang, "The multivariate L1-median and
      associated data depth", PNAS 97(4):1423-1426, 2000,
      DOI 10.1073/pnas.97.4.1423). Without it the update divides by zero, which
      is not a corner case here: with an odd number of identical honest updates
      the coordinate-wise median *is* a data point on iteration 0.

    ``eps`` doubles as the convergence tolerance on the step size and as the
    radius at which the iterate counts as coinciding with a data point.
    """
    if not updates:
        raise ValueError("geometric_median() needs at least one update")
    mat = _stack(updates)
    like = updates[0].params
    if mat.shape[0] == 1:
        return unflatten(mat[0], like)

    y = np.median(mat, axis=0)
    for _ in range(max_iter):
        diff = mat - y
        dist = np.linalg.norm(diff, axis=1)
        at_y = dist <= eps
        far = ~at_y
        if not far.any():
            break  # every point sits on the iterate: it is the median
        w = 1.0 / dist[far]
        weiszfeld = (w[:, None] * mat[far]).sum(axis=0) / w.sum()
        eta = int(at_y.sum())
        if eta == 0:
            y_next = weiszfeld
        else:
            # R is the norm of the sub-gradient contributed by the non-coincident
            # points; gamma blends towards y so the step never overshoots the
            # non-differentiable point it is standing on.
            r = float(np.linalg.norm((diff[far] * w[:, None]).sum(axis=0)))
            gamma = 1.0 if r <= eps else min(1.0, eta / r)
            y_next = (1.0 - gamma) * weiszfeld + gamma * y
        step = float(np.linalg.norm(y_next - y))
        y = y_next
        if step <= eps:
            break
    return unflatten(y, like)
