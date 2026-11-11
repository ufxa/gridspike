"""FedGrid Vast.ai fleet controller + monitoring harness (phase 3).

This package rents a fleet of GPUs on Vast.ai, runs the phase 4 training battery
(~540 runs) across them with **zero data loss and full observability**, and tears
the fleet down at the end. It is designed against three hard user requirements:

* **Nothing is lost.** Every run is idempotent (``experiments.run_one`` SKIPs when
  ``runs/<run_id>.json`` already exists), results are *pulled* to the controller
  continuously, and any manifest line without a collected result is re-dispatched
  until the collected set equals the manifest (:func:`fleet.Fleet.gap_repair`).
* **Auto-resurrect.** A fallen instance is detected, destroyed and replaced; its
  unfinished shard lines simply reappear as gaps and are re-run from scratch.
* **Map / monitor everything and catch deceiving hosts.** :mod:`.monitor` records
  a full machine-readable map every cycle, verifies each host reports the GPU it
  advertised, and flags hosts whose per-run wall-time exceeds the fleet median.

Security posture (VAST_BUDGET §6, community hosts): the Vast API key lives only in
the operator's environment (``VAST_API_KEY``). It is passed to the ``vastai``
CLI per invocation and is **never** written to a file or copied onto an instance;
results flow home by controller *pull*. The ``vastai`` dependency is optional
(``pip install 'fedgrid[fleet]'``) and is invoked as a subprocess, so importing
this package never requires it.
"""

from __future__ import annotations

__all__ = ["fleet", "monitor"]
