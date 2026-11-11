"""Wire schemas for the live console stream.

``RoundEvent`` is versioned (``v: 1``) and mirrored field-for-field by
``frontend/src/types.ts``; changing one side without the other is a contract
violation. — implemented in phase 1 because both sides of
the cross-language contract must exist before the panels do.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ClientRound", "ConsensusStats", "EnergyStats", "GlobalMetrics", "RoundEvent"]


class ClientRound(BaseModel):
    """One client's outcome in one round, as the console's Topology/Attacks panels see it."""

    cid: int
    n_samples: int
    loss: float
    fidelity: float
    accepted: bool
    attack: str | None = None  # ground-truth tag for evaluation display only


class GlobalMetrics(BaseModel):
    """Global-model quality after the round's aggregation."""

    loss: float
    acc: float
    macro_f1: float


class ConsensusStats(BaseModel):
    """Measured PBFT overhead and the rejected client ids for the round."""

    n_messages: int
    latency_ms: float
    rejected: list[int]


class EnergyStats(BaseModel):
    """SynOps count and SNN/ANN energy ratio accumulated over the round."""

    synops: float
    ratio: float


class RoundEvent(BaseModel):
    """One federated round, as streamed over ``/ws/live`` and stored in RunResult.rounds."""

    # populate_by_name lets Python code use `global_` while the wire keeps "global",
    # which is a Python keyword and therefore needs the alias.
    model_config = ConfigDict(populate_by_name=True)

    v: Literal[1] = 1
    run_id: str
    round: int
    clients: list[ClientRound]
    global_: GlobalMetrics = Field(alias="global")
    consensus: ConsensusStats
    energy: EnergyStats
    ts: datetime  # UTC; serialised as ISO-8601 with "Z"
