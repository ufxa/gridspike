"""Federated learning layer: native in-process engine, strategies, local DP.

Deterministic and seedable by design — no Ray/Flower runtime dependency, so the
whole federation runs in one process and results are bit-reproducible on CPU.
"""

from __future__ import annotations
