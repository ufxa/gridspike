"""FedGrid: Byzantine-robust federated learning for IEC 61850 substations.

FedGrid is a reproducible co-simulation framework that couples an OpenDSS-based
three-phase distribution grid simulator with a federated deep-learning pipeline
(a CNN-LSTM reference and a spiking LIF classifier whose per-inference energy is
measured via SynOps accounting) and a Byzantine-robust validation layer built on
fidelity scoring, PBFT consensus and robust aggregation.

The package is organised in four layers, mirrored by the sub-packages:

* :mod:`fedgrid.simulation` -- grid physics, anomaly injection, non-IID partitioning.
* :mod:`fedgrid.models` / :mod:`fedgrid.fl` -- CNN-LSTM and spiking (SNN) classifiers,
  federated training.
* :mod:`fedgrid.consensus` / :mod:`fedgrid.attacks` -- robust validation and threat models.
* :mod:`fedgrid.api` / :mod:`fedgrid.metrics` -- observability, benchmarking, live console.

See ``docs/ARCHITECTURE.md`` for the full design.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
