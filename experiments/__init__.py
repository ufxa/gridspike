"""FedGrid experiment runner (phase 3).

The experiment-execution package that turns the phase 2 modules into a reproducible
battery: :mod:`experiments.manifest` is the single source of truth for the arm /
config grid (``docs/VALIDATION_STANDARD.md`` sec 3.9 sized to the Compact battery of
CONTRACT amendments e/h), :mod:`experiments.build` maps one config dict to the
concrete objects a run needs, :mod:`experiments.run_one` executes exactly one
manifest line into a ``RunResult`` JSON, :mod:`experiments.run_all` enumerates /
drives the whole battery, and :mod:`experiments.analyze` computes the
non-parametric statistical battery over a directory of results.

Nothing here is imported at package load beyond the standard library so that
``python -m experiments.run_all --dry-run`` stays torch-free and instant.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
