"""Fidelity scoring: how far each update sits from the coordinate-wise median.

The coordinate-wise median is the cheapest breakdown-point-1/2 location estimate
available at this stage of the pipeline: it needs no pairwise distances (unlike
Krum) and no iteration (unlike the geometric median), so the validators can score
every update before PBFT is entered — the score *is* the ballot each validator
votes on. Cosine distance rather than L2 keeps the score scale-free, so one
global threshold survives changes in learning rate, model size and round index.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fedgrid.fl.engine import ClientUpdate

__all__ = ["fidelity_scores", "flatten", "unflatten"]

# A vector shorter than this carries no usable direction; cosine is undefined and
# the score falls back to the neutral 1.0 (see `fidelity_scores`). Chosen well
# below the norm of any float32 parameter vector that survived training.
_NORM_EPS = 1e-12


def flatten(params: list[np.ndarray]) -> np.ndarray:
    """Concatenate a parameter list into one 1-D vector (fixed layer order).

    Always returns float64: every consumer (median, cosine, Weiszfeld) is a
    reduction over hundreds of thousands of terms, where float32 accumulation
    error is large enough to move a score across a threshold.
    """
    if not params:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in params])


def unflatten(vec: np.ndarray, like: list[np.ndarray]) -> list[np.ndarray]:
    """Inverse of :func:`flatten`: split ``vec`` into arrays shaped like ``like``.

    Floating dtypes are restored from ``like`` (so a float32 model gets float32
    parameters back); integer dtypes are deliberately *not* restored — rounding
    an aggregate into an integer buffer silently destroys it, so those stay
    float64 and the caller must decide what to do.
    """
    total = int(sum(p.size for p in like))
    if vec.size != total:
        raise ValueError(f"cannot unflatten {vec.size} values into {total} slots")
    out: list[np.ndarray] = []
    i = 0
    for p in like:
        chunk = vec[i : i + p.size].reshape(p.shape)
        restore = np.issubdtype(p.dtype, np.floating)
        out.append(chunk.astype(p.dtype, copy=True) if restore else chunk.copy())
        i += p.size
    return out


def _stack(updates: list[ClientUpdate]) -> np.ndarray:
    """(n, d) matrix of flattened updates; raises on a client with a foreign shape."""
    rows = [flatten(u.params) for u in updates]
    d = rows[0].size
    for u, r in zip(updates, rows, strict=True):
        if r.size != d:
            raise ValueError(
                f"client {u.cid} submitted {r.size} parameters, expected {d} — "
                "the federation is not running one model"
            )
    return np.stack(rows)


def fidelity_scores(
    updates: list[ClientUpdate], ref: list[np.ndarray] | None = None
) -> np.ndarray:
    """``1 - cosine(u_i, coordinate-wise median of all flattened updates)``; in [0, 2].

    ``ref`` is a contract-compatible extension (the default ``None`` is exactly the
    contract's signature and semantics). When the caller has the round's starting
    global parameters — the ``ValidatorFn`` does, as its second argument — pass
    them: since ``ClientUpdate.params`` carries *full parameters* rather than
    deltas, every client vector is dominated by the shared global component, and
    cosine distances between full parameter vectors are compressed into a narrow
    band near zero where no threshold separates anything. Scoring ``params - ref``
    measures the quantity the defense actually cares about: the direction each
    client is pushing the model in. :func:`fedgrid.consensus.pipeline.make_validator`
    always passes it; the raw form is kept for callers that only hold updates.

    Numerics: cosine is clipped to [-1, 1] before the subtraction, so the result
    is in [0, 2] by construction. A zero-length update — or a zero median, which
    happens when an even number of clients cancels coordinate-wise — has no
    direction at all; it scores the neutral 1.0 (neither aligned nor opposed)
    rather than producing a NaN that would silently poison the threshold test.
    """
    if not updates:
        return np.zeros(0, dtype=np.float64)

    mat = _stack(updates)
    if ref is not None:
        mat = mat - flatten(ref)[None, :]

    median = np.median(mat, axis=0)
    median_norm = float(np.linalg.norm(median))
    n = mat.shape[0]
    if median_norm <= _NORM_EPS:
        return np.ones(n, dtype=np.float64)

    norms = np.linalg.norm(mat, axis=1)
    usable = norms > _NORM_EPS
    cos = np.zeros(n, dtype=np.float64)
    cos[usable] = (mat[usable] @ median) / (norms[usable] * median_norm)
    scores = 1.0 - np.clip(cos, -1.0, 1.0)
    scores[~usable] = 1.0
    return scores
