"""Measurement layer: energy accounting, inference latency, run serialisation.

Every quoted paper number flows through here — energy from SynOps/MAC counts,
latency percentiles against the 4 ms GOOSE budget, and :class:`RunResult` as the
single on-disk format every experiment writes.
"""

from __future__ import annotations

from fedgrid.metrics.energy import (
    E_AC_INT8_PJ,
    E_AC_PJ,
    E_MAC_INT8_PJ,
    E_MAC_PJ,
    energy_uj,
)
from fedgrid.metrics.latency import GOOSE_BUDGET_MS, LatencyStats, measure_inference_latency
from fedgrid.metrics.runresult import (
    RunResult,
    current_env_fingerprint,
    current_git_commit,
    load_run,
    save_run,
)

__all__ = [
    "E_AC_INT8_PJ",
    "E_AC_PJ",
    "E_MAC_INT8_PJ",
    "E_MAC_PJ",
    "GOOSE_BUDGET_MS",
    "LatencyStats",
    "RunResult",
    "current_env_fingerprint",
    "current_git_commit",
    "energy_uj",
    "load_run",
    "measure_inference_latency",
    "save_run",
]
