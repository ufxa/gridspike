"""Glue: package fidelity + PBFT + robust aggregation as the engine's ValidatorFn.

One call is one validation round:

    fidelity_scores(updates, ref=current)      # public, cheap, scale-free ballot
        -> PBFTSim.run_round(scores, tau)      # M validators agree on the partition
            -> krum | multi_krum | geomedian   # robust aggregate over the admitted set
                -> Ledger.append(record)       # audit trail (interior tamper-evidence
                                               #   only -- see ledger.py on truncation)

Fail-closed at every step: a round that does not reach consensus admits nobody
and returns no aggregate, and the engine is expected to keep the previous global
model for that round rather than aggregate an unvalidated set.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from fedgrid.consensus.aggregation import geometric_median, krum, multi_krum
from fedgrid.consensus.fidelity import fidelity_scores
from fedgrid.consensus.ledger import Ledger
from fedgrid.consensus.pbft import PBFTSim, default_net_ms
from fedgrid.fl.engine import ValidationReport

if TYPE_CHECKING:
    from fedgrid.fl.engine import ClientUpdate, ValidatorFn

__all__ = ["AGGREGATORS", "ConsensusValidationReport", "ConsensusValidator", "make_validator"]

# Accepted values of ``agg``; the aliases exist because "geomedian" is what the
# experiment configs say and "geometric_median" is what the function is called.
AGGREGATORS: dict[str, str] = {
    "krum": "krum",
    "multi_krum": "multi_krum",
    "multikrum": "multi_krum",
    "geomedian": "geomedian",
    "geometric_median": "geomedian",
    "gm": "geomedian",
}


@dataclass
class ConsensusValidationReport(ValidationReport):
    """``ValidationReport`` plus what a *robust-aggregating* validator produces.

    Extension, not a redefinition (contract amendment 2026-07-28c: owners may
    extend the shared types, never shrink them). The base type carries no place
    for parameters, but Krum and the geometric median differ in kind: Krum's
    result is a *selection* the engine could re-derive from ``accepted``, while
    the geometric median is a new point that exists nowhere in the update list.

    Handshake with ``fl/engine.py`` (the federated-learning module): use ``getattr(report, "aggregate",
    None)``; when it is not ``None`` it is the round's global parameters and the
    engine's ``Strategy`` must be bypassed -- the robust aggregator has already
    done the aggregation, and re-averaging it with anything would undo the
    robustness. When it is ``None`` (no commit) the engine keeps ``current``.
    ``accepted`` lists the client ids that actually entered the aggregate, which
    for classic Krum is a single client; ``rejected`` is its complement over the
    round's clients, i.e. the union of the consensus rejections and the ones the
    robust aggregator did not select. The consensus-level partition (what the
    RoundEvent's ``consensus.rejected`` should carry) stays in ``consensus``.

    If ``ValidationReport`` ever gains a field without a default, this subclass
    stops importing -- dataclass inheritance forbids a defaulted field ahead of a
    required one. That is deliberate: the break is loud and immediate rather than
    a silent divergence between the two owners' views of the type.
    """

    aggregate: list[np.ndarray] | None = None
    agg: str = "krum"
    ledger_hash: str | None = None
    rejected: list[int] = field(default_factory=list)


class ConsensusValidator:
    """Callable implementing ``ValidatorFn``; keeps the ledger and the PBFT stream.

    Stateful on purpose: the PBFT generator advances across rounds (so a run is
    reproducible as a whole), the ledger accumulates, and the round counter is
    the validator's own -- the ``ValidatorFn`` signature passes no round number.
    """

    def __init__(
        self,
        m: int,
        f: int,
        threshold: float,
        agg: str = "krum",
        seed: int = 0,
        net_ms: Callable[[np.random.Generator], float] | None = None,
    ) -> None:
        key = AGGREGATORS.get(str(agg).lower())
        if key is None:
            raise ValueError(f"unknown agg {agg!r}; expected one of {sorted(set(AGGREGATORS))}")
        self.agg = key
        self.m = int(m)
        self.f = int(f)
        self.threshold = float(threshold)
        self.seed = int(seed)
        self.pbft = PBFTSim(m=m, f=f, net_ms=net_ms or default_net_ms, seed=seed)
        self.ledger = Ledger()
        self.round = 0

    def _aggregate(self, admitted: list[ClientUpdate]) -> tuple[list[np.ndarray], list[int]]:
        """Robust aggregate over the admitted updates + the indices that fed it.

        The Byzantine budget handed to Krum is the validator ``f``: the contract's
        ``make_validator(m, f, ...)`` carries one ``f``, and in the battery the
        validator and client Byzantine counts are set together. Averaging over a
        multi-Krum selection is unweighted -- ``n_samples`` is a number the client
        reports about itself, so weighting by it hands the adversary a free
        multiplier on its own influence.
        """
        if self.agg == "geomedian":
            return geometric_median(admitted), list(range(len(admitted)))
        if self.agg == "krum":
            i = krum(admitted, self.f)
            return [p.copy() for p in admitted[i].params], [i]
        picks = multi_krum(admitted, self.f, max(1, len(admitted) - self.f))
        stacked = [
            np.mean([admitted[i].params[layer] for i in picks], axis=0)
            for layer in range(len(admitted[0].params))
        ]
        return stacked, sorted(picks)

    def __call__(
        self, updates: list[ClientUpdate], current: list[np.ndarray]
    ) -> ConsensusValidationReport:
        scores = fidelity_scores(updates, ref=current)
        cres = self.pbft.run_round(scores, self.threshold)

        # PBFT votes on positions in the update list; everything downstream of
        # here speaks client ids, which is what the console and the ledger show.
        cids = [u.cid for u in updates]
        consensus = replace(
            cres,
            accepted=[cids[i] for i in cres.accepted],
            rejected=[cids[i] for i in cres.rejected],
        )

        aggregate: list[np.ndarray] | None = None
        accepted_cids: list[int] = []
        if cres.committed and cres.accepted:
            admitted = [updates[i] for i in cres.accepted]
            aggregate, picks = self._aggregate(admitted)
            accepted_cids = [admitted[i].cid for i in picks]
        rejected_cids = [c for c in cids if c not in set(accepted_cids)]

        head = self.ledger.append(
            {
                "round": self.round,
                "agg": self.agg,
                "threshold": self.threshold,
                "validators": {"m": self.m, "f": self.f},
                "scores": {str(c): float(s) for c, s in zip(cids, scores, strict=True)},
                "committed": bool(cres.committed),
                "consensus_accepted": consensus.accepted,
                "consensus_rejected": consensus.rejected,
                "aggregated": accepted_cids,
                "n_messages": int(cres.n_messages),
                "sim_latency_ms": float(cres.sim_latency_ms),
            }
        )
        self.round += 1
        return ConsensusValidationReport(
            accepted=accepted_cids,
            scores=scores,
            consensus=consensus,
            aggregate=aggregate,
            agg=self.agg,
            ledger_hash=head,
            rejected=rejected_cids,
        )


def make_validator(
    m: int,
    f: int,
    threshold: float,
    agg: str = "krum",
    seed: int = 0,
    net_ms: Callable[[np.random.Generator], float] | None = None,
) -> ValidatorFn:
    """Build the validator the engine plugs in: fidelity -> PBFT(m, f) -> ``agg``.

    ``agg`` is one of ``krum`` (classic single selection), ``multi_krum``
    (mean of the ``n - f`` best) or ``geomedian``/``geometric_median``.
    ``net_ms`` is an extension over the contract signature: the inter-validator
    latency distribution belongs to the realism block (the data-simulation module), and leaving it
    injectable is what keeps ``pbft.default_net_ms`` -- which is explicitly a
    provisional placeholder, not a sourced fit -- out of the published numbers.

    The returned object is a :class:`ConsensusValidator`, so callers that want the
    audit trail can reach ``validator.ledger`` and ``validator.pbft``.
    """
    return ConsensusValidator(m=m, f=f, threshold=threshold, agg=agg, seed=seed, net_ms=net_ms)
