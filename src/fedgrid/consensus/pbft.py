"""Discrete-event PBFT simulation: consensus overhead is measured, not assumed.

Protocol modelled: Castro and Liskov, "Practical Byzantine Fault Tolerance",
OSDI 1999 -- normal-case operation only (pre-prepare / prepare / commit). What is
deliberately *not* modelled, because every omission makes our reported overhead a
lower bound rather than a flattering one:

* **view changes.** Replica 0 is the primary and is always honest here, so the
  round never times out and re-elects. A faulty primary costs at least one extra
  timeout plus an O(M^2) view-change exchange; reporting the steady-state cost is
  the conservative direction only in the sense that we never *overstate* the
  system's speed advantage -- it is stated openly in the paper.
* **batching, checkpoints, garbage collection, authentication cost.** One
  validation round is one PBFT instance; MAC/signature verification time is not
  added to the latency, only network time is.
* **crash faults.** The ``f`` faulty validators here equivocate (they vote against
  the honest partition) but stay online and keep sending, which is the harder
  case for agreement and the honest case for message counting.

Message complexity of one round (derivation, ``M`` validators):

* **pre-prepare** -- the primary multicasts to the other replicas: ``M - 1``.
* **prepare** -- every *backup* multicasts to all other replicas. There are
  ``M - 1`` backups and each sends to the ``M - 1`` replicas that are not itself:
  ``(M - 1)^2``. The primary sends no PREPARE: in PBFT its PRE-PREPARE *is* its
  prepare vote (this is the detail that makes the term ``(M-1)^2`` and not
  ``M(M-1)``; the looser textbook count double-counts the primary's vote).
* **commit** -- every replica that reached the *prepared* predicate multicasts
  COMMIT to the other ``M - 1``. In the normal case that is all ``M`` replicas
  (the ``f`` faulty ones send too, they just send something that does not match),
  giving ``M(M - 1)``.

  Total (committing round): ``(M-1) + (M-1)^2 + M(M-1) = 2M(M-1)``.

  When ``M < 3f + 1`` no honest replica ever becomes prepared, so only the ``f``
  faulty replicas emit COMMIT and the round burns
  ``(M-1) + (M-1)^2 + f(M-1) = (M-1)(M + f)`` messages before stalling. Both
  branches are the single exact formula

      ``n_messages = (M-1) + (M-1)^2 + S(M-1)``,  ``S = |{replicas that COMMIT}|``

  with ``S = M`` on commit and ``S = f`` on stall. ``PBFTSim`` draws exactly one
  ``net_ms`` sample per message, so the RNG draw count equals ``n_messages`` --
  which is how the test pins the formula rather than trusting the arithmetic.

Quorum arithmetic (why the safety property is exact, not statistical): a replica
is *prepared* once it holds the pre-prepare and ``2f`` matching PREPAREs, counting
its own if it is a backup. Matching PREPAREs only ever come from honest backups,
of which there are ``M - 1 - f``; both a backup (own vote + honest peers) and the
primary (honest peers only) therefore see the same ``M - 1 - f`` matching votes
and need ``M - 1 - f >= 2f``. Commit needs ``2f + 1`` matching COMMITs out of the
``M - f`` honest replicas, i.e. ``M - f >= 2f + 1``. Both reduce to ``M >= 3f + 1``,
so with ``f > (M-1)/3`` *no* honest replica commits -- the round is decided all-or-
nothing, and ``committed`` is a deterministic predicate of ``(M, f)`` rather than
of the latency draws.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = [
    "DEFAULT_NET_MEDIAN_MS",
    "DEFAULT_NET_SIGMA_LOG",
    "ConsensusResult",
    "PBFTSim",
    "default_net_ms",
]

# PROVISIONAL, NOT A SOURCED FIT. The contract (Realism block, amendment
# 2026-07-28b) binds the PBFT inter-validator latency to a documented lognormal
# fit of a utility WAN, and assigns that parameter row to the data-simulation module / REALISM.md.
# These two numbers exist only so this module is runnable and testable standalone;
# they are a plausible regional-WAN shape (median 12 ms, heavy right tail), not a
# measurement, and any figure produced with them must say so. Experiments inject
# the real callable through ``make_validator(net_ms=...)``.
DEFAULT_NET_MEDIAN_MS = 12.0
DEFAULT_NET_SIGMA_LOG = 0.6


def default_net_ms(rng: np.random.Generator) -> float:
    """One inter-validator message latency in ms (provisional lognormal, see above)."""
    return float(rng.lognormal(mean=math.log(DEFAULT_NET_MEDIAN_MS), sigma=DEFAULT_NET_SIGMA_LOG))


@dataclass
class ConsensusResult:
    """Outcome of one PBFT round over the proposed accept/reject partition."""

    committed: bool
    accepted: list[int]
    n_messages: int
    sim_latency_ms: float
    rejected: list[int]


class PBFTSim:
    """Discrete-event PBFT (pre-prepare/prepare/commit) over M validators, f faulty.

    Message latency sampled from ``net_ms(rng)``; faulty validators vote
    adversarially.

    Validators are ``0..m-1``; the faulty set is the top ``f`` ids, so the primary
    (replica 0) is honest -- see the module docstring for why the faulty-primary
    path is out of scope. The generator is owned by the instance and advances
    across calls, so a sequence of rounds is reproducible from ``seed`` but a
    single round is not independent of the rounds before it (that is what makes a
    whole run reproducible rather than each round separately).
    """

    def __init__(
        self,
        m: int,
        f: int,
        net_ms: Callable[[np.random.Generator], float],
        seed: int = 0,
    ) -> None:
        if m < 1:
            raise ValueError(f"need at least one validator, got m={m}")
        if not 0 <= f <= m:
            raise ValueError(f"f={f} outside [0, m={m}]")
        self.m = int(m)
        self.f = int(f)
        self.net_ms = net_ms
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)
        # Byzantine ids are the tail so that replica 0 (the primary) stays honest.
        self.faulty: frozenset[int] = frozenset(range(self.m - self.f, self.m))
        self.honest: tuple[int, ...] = tuple(i for i in range(self.m) if i not in self.faulty)

    @property
    def tolerates(self) -> bool:
        """Whether ``f`` is within the ``M >= 3f + 1`` bound this round can survive."""
        return self.m >= 3 * self.f + 1

    def _hop(self) -> float:
        """One message's flight time; exactly one RNG draw, so draws == n_messages."""
        dt = float(self.net_ms(self._rng))
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError(f"net_ms returned {dt}; latency must be finite and non-negative")
        return dt

    def run_round(self, scores: np.ndarray, threshold: float) -> ConsensusResult:
        """Propose accepting every update scoring below ``threshold``; run the 3 phases.

        The primary's proposal is the honest partition ``score <= threshold``.
        Honest validators recompute it from the same public scores and therefore
        always match; faulty validators send a *different* partition, so their
        votes never count towards a quorum. Positions in ``scores`` are indices
        into the update list, not client ids -- the pipeline maps them.

        Fails closed: a round that does not commit accepts nothing and rejects
        everything, so a stalled consensus can never leak an unvalidated update
        into the aggregate.
        """
        scores = np.asarray(scores, dtype=np.float64)
        n = int(scores.size)
        proposal = [i for i in range(n) if scores[i] <= threshold]

        # --- phase 1: pre-prepare (primary -> backups), M-1 messages -------------
        arrive_pp = np.zeros(self.m, dtype=np.float64)
        for j in range(1, self.m):
            arrive_pp[j] = self._hop()
        n_messages = max(self.m - 1, 0)

        # --- phase 2: prepare (each backup -> everyone else), (M-1)^2 messages ---
        # A replica's prepare leaves as soon as it has the pre-prepare; faulty
        # backups send at the same time but their content does not match.
        prepare_arrival: dict[int, list[float]] = {k: [] for k in range(self.m)}
        for j in range(1, self.m):
            for k in range(self.m):
                if k == j:
                    continue
                t = arrive_pp[j] + self._hop()
                n_messages += 1
                if j not in self.faulty:
                    prepare_arrival[k].append(t)

        need_prepare = 2 * self.f
        prepared: dict[int, float] = {}
        for k in self.honest:
            votes = list(prepare_arrival[k])
            if k != 0:  # a backup's own PREPARE is in its own log, at zero delay
                votes.append(float(arrive_pp[k]))
            if need_prepare == 0:
                prepared[k] = float(arrive_pp[k])
            elif len(votes) >= need_prepare:
                votes.sort()
                prepared[k] = votes[need_prepare - 1]

        # --- phase 3: commit (each prepared replica -> everyone else) ------------
        # Faulty replicas always emit a (mismatching) COMMIT; honest ones only
        # once prepared. Senders are visited in id order so the RNG stream is a
        # function of (m, f, seed) alone.
        send_commit: dict[int, float] = {k: float(arrive_pp[k]) for k in self.faulty}
        send_commit.update(prepared)
        commit_arrival: dict[int, list[float]] = {k: [] for k in range(self.m)}
        for k in sorted(send_commit):
            for j in range(self.m):
                if j == k:
                    continue
                t = send_commit[k] + self._hop()
                n_messages += 1
                if k not in self.faulty:
                    commit_arrival[j].append(t)

        need_commit = 2 * self.f + 1
        committed_at: list[float] = []
        for k in self.honest:
            if k not in prepared:
                continue
            votes = list(commit_arrival[k])
            votes.append(prepared[k])  # own COMMIT, no network hop
            if len(votes) >= need_commit:
                votes.sort()
                committed_at.append(votes[need_commit - 1])

        # The round is complete only when *every* honest validator holds the
        # decision: each of them writes the block to its ledger replica, so the
        # audit trail -- not the first replica to learn the outcome -- sets the
        # critical path. |honest| >= 2f+1 whenever the round commits at all.
        committed = len(committed_at) == len(self.honest) and self.tolerates
        if committed:
            latency = max(committed_at)
            accepted = proposal
            rejected = [i for i in range(n) if i not in set(proposal)]
        else:
            # Nothing decided: the clock still ran for as long as the protocol
            # kept delivering messages (a real deployment then pays a view-change
            # timeout on top, which we do not model).
            delivered = [t for ts in prepare_arrival.values() for t in ts]
            delivered += [t for ts in commit_arrival.values() for t in ts]
            delivered += list(arrive_pp)
            latency = max(delivered) if delivered else 0.0
            accepted = []
            rejected = list(range(n))

        return ConsensusResult(
            committed=committed,
            accepted=accepted,
            n_messages=n_messages,
            sim_latency_ms=float(latency),
            rejected=rejected,
        )
