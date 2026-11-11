"""Attack interface: a poisoning transform applied to a client's update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np

    from fedgrid.fl.engine import ClientUpdate

__all__ = ["Attack", "AttackContext"]


@dataclass
class AttackContext:
    """Round-local information an attacker may legitimately observe."""

    round: int
    ref: list[np.ndarray]  # the global parameters the round started from
    rng: np.random.Generator


class Attack(Protocol):
    """Structural interface the engine applies to a Byzantine client's update."""

    def poison(self, update: ClientUpdate, ctx: AttackContext) -> ClientUpdate:
        """Return the poisoned update (may reuse or replace ``update``)."""
        ...
