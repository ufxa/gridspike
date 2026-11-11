"""Grid physics layer: quasi-static OpenDSS simulation, event injection, non-IID partitioning.

Not a co-simulation: there is one solver (OpenDSS via ``OpenDSSDirect.py``) run in
daily quasi-static mode, with the event plans and the measurement chain applied as
synthetic overlays on its solved state -- no second coupled simulator is involved.

* :mod:`fedgrid.simulation.grid` -- feeder specs and the quasi-static day simulator.
* :mod:`fedgrid.simulation.events` -- event plans (faults, overload, sag, swell).
* :mod:`fedgrid.simulation.partition` -- K-client federation builder (Dirichlet skew).
* :mod:`fedgrid.simulation.profiles` -- measured-model load shapes + PV irradiance.
* :mod:`fedgrid.simulation.measurement` -- merging-unit measurement chain, frequency.
* :mod:`fedgrid.simulation.network` -- utility-WAN latency sampler (for consensus).
* :mod:`fedgrid.simulation.generate` -- dataset generation CLI (``python -m ...``).
* ``feeders/`` -- vendored IEEE PES test feeders (BSD-3, EPRI/DSS-Extensions).

Every realism parameter is sourced in the module docstrings and tabulated in
``docs/REALISM.md``.
"""

from __future__ import annotations
