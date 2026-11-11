"""phase 1 smoke tests: the scaffold imports, matches the contract, and /health is live.

No phase 2 logic is exercised here — only that every module loads, the declared
signatures/constants are contract-exact, and the two implemented pieces
(``/health`` and ``RoundEvent``) actually work.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import fields
from typing import Any

import pytest

ALL_MODULES = [
    "fedgrid",
    "fedgrid.data",
    "fedgrid.simulation",
    "fedgrid.simulation.grid",
    "fedgrid.simulation.events",
    "fedgrid.simulation.partition",
    "fedgrid.models",
    "fedgrid.models.cnn_lstm",
    "fedgrid.models.snn",
    "fedgrid.metrics",
    "fedgrid.metrics.energy",
    "fedgrid.metrics.latency",
    "fedgrid.metrics.runresult",
    "fedgrid.fl",
    "fedgrid.fl.engine",
    "fedgrid.fl.strategies",
    "fedgrid.fl.ldp",
    "fedgrid.consensus",
    "fedgrid.consensus.fidelity",
    "fedgrid.consensus.aggregation",
    "fedgrid.consensus.pbft",
    "fedgrid.consensus.ledger",
    "fedgrid.consensus.pipeline",
    "fedgrid.attacks",
    "fedgrid.attacks.base",
    "fedgrid.attacks.suite",
    "fedgrid.api",
    "fedgrid.api.app",
    "fedgrid.api.schemas",
]


@pytest.mark.parametrize("module", ALL_MODULES)
def test_module_imports(module: str) -> None:
    importlib.import_module(module)


def _params(obj: Any) -> dict[str, Any]:
    """Parameter name -> default (inspect.Parameter.empty when required)."""
    return {name: p.default for name, p in inspect.signature(obj).parameters.items()}


def test_simulation_signatures() -> None:
    from fedgrid.simulation.grid import FeederSim
    from fedgrid.simulation.partition import make_federation

    run_day = _params(FeederSim.run_day)
    assert list(run_day) == ["self", "day", "events", "step_s"]
    assert run_day["step_s"] == 60.0

    assert _params(make_federation) == {"k": 10, "alpha": 0.5, "days": 30, "seed": 0}


def test_model_signatures() -> None:
    from fedgrid.models.cnn_lstm import CNNLSTM
    from fedgrid.models.snn import SpikingClassifier

    cnn = _params(CNNLSTM.__init__)
    assert cnn["in_ch"] == 11
    assert cnn["conv_channels"] == (32, 64)
    assert cnn["lstm_hidden"] == 128
    assert cnn["dropout"] == 0.3

    snn = _params(SpikingClassifier.__init__)
    assert snn["hidden"] == 256
    assert snn["T_steps"] == 8
    assert snn["coding"] == "rate"
    assert callable(SpikingClassifier.synops)
    assert callable(SpikingClassifier.reset_counters)


def test_metrics_declarations() -> None:
    from fedgrid.metrics.energy import E_AC_PJ, E_MAC_PJ
    from fedgrid.metrics.latency import GOOSE_BUDGET_MS, LatencyStats, measure_inference_latency
    from fedgrid.metrics.runresult import RunResult

    assert (E_MAC_PJ, E_AC_PJ, GOOSE_BUDGET_MS) == (4.6, 0.9, 4.0)
    assert [f.name for f in fields(LatencyStats)] == ["mean_ms", "p50_ms", "p95_ms", "p99_ms"]
    assert [f.name for f in fields(RunResult)] == [
        "method",
        "seed",
        "config",
        "rounds",
        "final",
        "energy",
        "latency",
        # Mandatory provenance per contract amendment 2026-07-28d §4
        # (landed by the models/metrics module in phase 2; expected list updated 2026-07-28).
        "git_commit",
        "env_fingerprint",
    ]

    lat = _params(measure_inference_latency)
    assert list(lat) == ["model", "x", "n", "device", "warmup"]
    assert (lat["n"], lat["device"], lat["warmup"]) == (1000, "cpu", 50)


def test_fl_signatures() -> None:
    from fedgrid.fl.engine import ClientUpdate, FederatedEngine
    from fedgrid.fl.ldp import LDPConfig, clip_and_noise, epsilon
    from fedgrid.fl.strategies import FedProx

    assert [f.name for f in fields(ClientUpdate)] == [
        "cid",
        "n_samples",
        "params",
        "train_loss",
        "attack",
    ]
    assert ClientUpdate.__dataclass_fields__["attack"].default is None

    eng = _params(FederatedEngine.__init__)
    assert eng["rounds"] is inspect.Parameter.empty
    assert (eng["local_epochs"], eng["batch_size"], eng["lr"]) == (1, 64, 1e-3)
    assert (eng["device"], eng["seed"], eng["on_round"]) == ("cuda", 0, None)

    assert _params(FedProx.__init__)["mu"] == 0.01
    assert list(_params(clip_and_noise)) == ["params", "ref", "cfg", "rng"]
    assert _params(epsilon)["delta"] == 1e-5
    assert _params(LDPConfig)["seed"] == 0


def test_consensus_signatures() -> None:
    from fedgrid.consensus.aggregation import geometric_median, krum
    from fedgrid.consensus.pbft import ConsensusResult, PBFTSim
    from fedgrid.consensus.pipeline import make_validator

    assert list(_params(krum)) == ["updates", "f"]
    gm = _params(geometric_median)
    assert (gm["eps"], gm["max_iter"]) == (1e-6, 100)

    assert [f.name for f in fields(ConsensusResult)] == [
        "committed",
        "accepted",
        "n_messages",
        "sim_latency_ms",
        "rejected",
    ]
    assert list(_params(PBFTSim.__init__)) == ["self", "m", "f", "net_ms", "seed"]
    assert list(_params(PBFTSim.run_round)) == ["self", "scores", "threshold"]

    mv = _params(make_validator)
    assert (mv["agg"], mv["seed"]) == ("krum", 0)


def test_event_class_is_the_contract_taxonomy() -> None:
    from fedgrid.data import EventClass

    expected = {
        "NORMAL": 0,
        "FAULT_LG": 1,
        "FAULT_LL": 2,
        "FAULT_LLG": 3,
        "FAULT_3PH": 4,
        "OVERLOAD": 5,
        "SAG": 6,
        "SWELL": 7,
    }
    assert {e.name: int(e) for e in EventClass} == expected


# (The phase 1 "stubs raise NotImplementedError" test was retired
# when phase 2 implementation started — real behaviour is covered by each owner's
# module test file.)


def test_health_endpoint() -> None:
    from fastapi.testclient import TestClient

    import fedgrid
    from fedgrid.api.app import RunStore, create_app

    with TestClient(create_app(RunStore())) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok", "version": fedgrid.__version__}
        # 2026-07-28, the API module (phase 2): runs endpoints are live now — the old
        # "everything answers 501" checks are superseded (full coverage in
        # tests/test_api.py). Only the phase 3 launcher may still answer 501.
        assert client.get("/api/runs").status_code == 200
        assert client.get("/api/runs/abc").status_code == 404
        assert client.post("/api/experiments").status_code == 501


# The contract's RoundEvent example, verbatim except for a concrete run_id.
ROUND_EVENT_EXAMPLE = {
    "v": 1,
    "run_id": "demo-000",
    "round": 3,
    "clients": [
        {
            "cid": 0,
            "n_samples": 812,
            "loss": 0.41,
            "fidelity": 0.07,
            "accepted": True,
            "attack": None,
        },
    ],
    "global": {"loss": 0.38, "acc": 0.91, "macro_f1": 0.89},
    "consensus": {"n_messages": 264, "latency_ms": 118.2, "rejected": [7]},
    "energy": {"synops": 1.9e6, "ratio": 0.11},
    "ts": "2026-07-28T00:00:00Z",
}


def test_round_event_round_trips_contract_example() -> None:
    import json

    from pydantic import ValidationError

    from fedgrid.api.schemas import RoundEvent

    ev = RoundEvent.model_validate(ROUND_EVENT_EXAMPLE)
    assert ev.v == 1
    assert ev.global_.acc == 0.91
    assert ev.clients[0].attack is None
    # Byte-level wire compatibility: dumping by alias reproduces the example dict.
    assert ev.model_dump(mode="json", by_alias=True) == ROUND_EVENT_EXAMPLE

    # And the JSON-string path (what actually crosses the WebSocket) is stable too.
    again = RoundEvent.model_validate_json(ev.model_dump_json(by_alias=True))
    assert again == ev
    assert json.loads(ev.model_dump_json(by_alias=True)) == ROUND_EVENT_EXAMPLE

    # The version tag is a hard literal: v=2 must be rejected, not coerced.
    with pytest.raises(ValidationError):
        RoundEvent.model_validate({**ROUND_EVENT_EXAMPLE, "v": 2})
