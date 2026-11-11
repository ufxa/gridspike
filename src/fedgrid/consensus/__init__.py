"""Robust validation layer: fidelity scoring -> PBFT consensus -> robust aggregation.

Updates are validated *before* aggregation: cosine-distance fidelity scores feed a
discrete-event PBFT simulation among validators; the admitted set is aggregated
with Krum or the geometric median; every decision lands in a hash-chained ledger.
"""

from __future__ import annotations
