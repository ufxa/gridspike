"""Aggregation strategies: the plain baselines robust aggregation is compared against.

* :class:`FedAvg` — n_samples-weighted parameter mean (McMahan et al.,
  "Communication-Efficient Learning of Deep Networks from Decentralized
  Data", AISTATS 2017, arXiv:1602.05629).
* :class:`FedProx` — FedAvg server side plus a proximal term
  ``mu/2 * ||w - w_global||^2`` in the *client* loss (Li et al., "Federated
  Optimization in Heterogeneous Networks", MLSys 2020, arXiv:1812.06127).

Client-side cooperation mechanism (kept within the federated-learning module's files by design):
the engine probes the strategy for an optional ``client_loss_hook(model,
ref)`` method and, when present, adds its return value to every local
minibatch loss; returning ``None`` disables the term for that step. FedAvg
defines no hook; FedProx returns ``None`` at ``mu == 0`` so that
``FedProx(0)`` is bit-identical to FedAvg (no extra autograd nodes).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor, nn

    from fedgrid.fl.engine import ClientUpdate

__all__ = ["FedAvg", "FedProx", "Strategy"]


class Strategy(Protocol):
    """Structural interface the engine calls once per round."""

    def aggregate(self, updates: list[ClientUpdate], current: list[np.ndarray]) -> list[np.ndarray]:
        """Combine client updates into the next global parameter list."""
        ...


class FedAvg(Strategy):
    """``n_samples``-weighted mean of client parameters (McMahan et al.).

    Accumulates in float64 so the result is independent of parameter dtype
    quirks, then casts back to each current array's dtype. Edge cases are
    engine-facing, not errors: an empty update list returns ``current``
    unchanged (an all-rejected consensus round must not crash the run) and
    all-zero ``n_samples`` falls back to uniform weights.
    """

    def aggregate(self, updates: list[ClientUpdate], current: list[np.ndarray]) -> list[np.ndarray]:
        if not updates:
            return [np.asarray(a).copy() for a in current]
        for u in updates:
            if len(u.params) != len(current):
                raise ValueError(
                    f"client {u.cid} sent {len(u.params)} arrays, global has {len(current)}"
                )
        weights = np.asarray([float(u.n_samples) for u in updates], dtype=np.float64)
        total = float(weights.sum())
        if total > 0:
            weights = weights / total
        else:
            weights = np.full(len(updates), 1.0 / len(updates), dtype=np.float64)

        out: list[np.ndarray] = []
        for i, cur in enumerate(current):
            acc = np.zeros(np.shape(cur), dtype=np.float64)
            for w, u in zip(weights, updates, strict=True):
                acc += w * np.asarray(u.params[i], dtype=np.float64)
            out.append(acc.astype(np.asarray(cur).dtype))
        return out


class FedProx(FedAvg):
    """FedAvg aggregation plus a proximal term (mu) in the client loss (Li et al.)."""

    def __init__(self, mu: float = 0.01) -> None:
        if mu < 0:
            raise ValueError(f"mu must be >= 0, got {mu}")
        self.mu = float(mu)

    def client_loss_hook(self, model: nn.Module, ref: Sequence[Tensor]) -> Tensor | None:
        """``mu/2 * sum_p ||p - p_ref||^2`` over ``model.parameters()``.

        ``ref`` is aligned with ``model.parameters()`` order (the engine builds
        it from the round-start global params). Returns ``None`` at ``mu == 0``
        so the engine skips the term entirely and FedProx(0) reproduces the
        FedAvg trajectory bit-for-bit.
        """
        if self.mu == 0.0:
            return None
        terms = [
            ((p - r.detach()) ** 2).sum()
            for p, r in zip(model.parameters(), ref, strict=True)
        ]
        return 0.5 * self.mu * sum(terms)
