"""phase 2 API tests (the API module): run store, REST endpoints, live WebSocket hub.

The store is exercised against a tmp_path runs directory holding minimal valid
RunResult JSONs built inline; the WS tests drive the real hub through
``app.state.publish_threadsafe`` from the test thread, exactly the way the
phase 3 training thread will.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fedgrid.api.app import RunStore, create_app
from fedgrid.api.schemas import RoundEvent

REPO_ROOT = Path(__file__).resolve().parents[1]

# The contract's RoundEvent example, verbatim except for a concrete run_id.
CONTRACT_EVENT = {
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


def _round(run_id: str, rnd: int, minute: int) -> dict:
    """One RoundEvent-shaped rounds[] entry (what the engine's on_round emits)."""
    return {
        "v": 1,
        "run_id": run_id,
        "round": rnd,
        "clients": [
            {
                "cid": 0,
                "n_samples": 812,
                "loss": 0.5 / rnd,
                "fidelity": 0.07,
                "accepted": True,
                "attack": None,
            },
        ],
        "global": {"loss": 0.6 / rnd, "acc": 0.7 + 0.02 * rnd, "macro_f1": 0.6 + 0.02 * rnd},
        "consensus": {"n_messages": 264, "latency_ms": 110.0, "rejected": []},
        "energy": {"synops": 1.9e6, "ratio": 0.11},
        "ts": f"2026-07-28T10:{minute:02d}:00Z",
    }


def _run_result(run_id: str, method: str, seed: int, n_rounds: int, acc: float) -> dict:
    """Minimal valid RunResult JSON (incl. amendment 2026-07-28d provenance fields)."""
    return {
        "method": method,
        "seed": seed,
        "config": {"rounds": n_rounds, "k": 10},
        "rounds": [_round(run_id, r + 1, r) for r in range(n_rounds)],
        "final": {
            "acc": acc,
            "macro_f1": acc - 0.02,
            "per_class_f1": [acc] * 8,
            "confusion": [[0] * 8 for _ in range(8)],
        },
        "energy": {"snn_uJ": 1.7, "ann_uJ": 15.4, "ratio": 0.11},
        "latency": None,
        "git_commit": "0000000000000000000000000000000000000000",
        "env_fingerprint": "torch-2.11.0+cu128/cpu-test",
    }


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    (d / "krum-s0.json").write_text(
        json.dumps(_run_result("krum-s0", "krum", 0, 3, 0.91)), encoding="utf-8"
    )
    (d / "fedavg-s1.json").write_text(
        json.dumps(_run_result("fedavg-s1", "fedavg", 1, 2, 0.88)), encoding="utf-8"
    )
    # Distinct mtimes make the newest-first ordering deterministic.
    now = time.time()
    os.utime(d / "krum-s0.json", (now - 100, now - 100))
    os.utime(d / "fedavg-s1.json", (now, now))
    # A corrupt file must be skipped by listings, never crash them.
    (d / "broken.json").write_text("{not json", encoding="utf-8")
    return d


@pytest.fixture
def client(runs_dir: Path):
    # Tiny heartbeat so WS tests can observe pings without waiting 20 s.
    app = create_app(RunStore(runs_dir), heartbeat_s=0.05)
    with TestClient(app) as c:
        yield c


# -- REST -----------------------------------------------------------------


def test_health(client: TestClient) -> None:
    import fedgrid

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "version": fedgrid.__version__}


def test_list_runs_summaries_newest_first(client: TestClient) -> None:
    r = client.get("/api/runs")
    assert r.status_code == 200
    runs = r.json()
    assert [s["run_id"] for s in runs] == ["fedavg-s1", "krum-s0"]  # mtime order
    top = runs[0]
    assert top["method"] == "fedavg"
    assert top["seed"] == 1
    assert top["n_rounds"] == 2
    assert top["final_acc"] == 0.88
    assert top["final_macro_f1"] == pytest.approx(0.86)
    assert top["started_at"] == "2026-07-28T10:00:00Z"
    assert top["finished_at"] == "2026-07-28T10:01:00Z"
    assert top["created_at"].endswith("Z")


def test_get_run_returns_full_json(client: TestClient) -> None:
    r = client.get("/api/runs/krum-s0")
    assert r.status_code == 200
    assert r.json() == _run_result("krum-s0", "krum", 0, 3, 0.91)


def test_get_run_404(client: TestClient) -> None:
    assert client.get("/api/runs/no-such-run").status_code == 404


def test_run_id_traversal_is_a_404_not_a_read(runs_dir: Path) -> None:
    store = RunStore(runs_dir)
    for bad in ("../pyproject", "..", "a/b", "a\\b", ".hidden", ""):
        with pytest.raises(KeyError):
            store.get_run(bad)


def test_missing_runs_dir_lists_empty(tmp_path: Path) -> None:
    assert RunStore(tmp_path / "nowhere").list_runs() == []


def test_experiments_still_501(client: TestClient) -> None:
    assert client.post("/api/experiments").status_code == 501


# -- live buffer ----------------------------------------------------------


def test_live_buffer_is_a_ring(tmp_path: Path) -> None:
    store = RunStore(tmp_path, live_buffer=2)
    for r in (1, 2, 3):
        store.push_live({"round": r})
    assert store.latest_live() == [{"round": 2}, {"round": 3}]
    assert store.latest_live(1) == [{"round": 3}]
    assert store.latest_live(0) == []


def test_live_latest_endpoint(client: TestClient) -> None:
    publish = client.app.state.publish_threadsafe
    for rnd in (1, 2, 3):
        publish(_round("live-run", rnd, rnd)).result(timeout=5)
    events = client.get("/api/live/latest").json()
    assert [e["round"] for e in events] == [1, 2, 3]
    assert events[0]["global"]["acc"] == pytest.approx(0.72)  # wire keeps the alias
    assert client.get("/api/live/latest", params={"n": 2}).json()[0]["round"] == 2
    assert client.get("/api/live/latest", params={"n": 0}).status_code == 422


# -- WebSocket hub --------------------------------------------------------


def test_ws_two_events_in_order_heartbeats_dont_corrupt(client: TestClient) -> None:
    publish = client.app.state.publish_threadsafe
    ev1, ev2 = _round("live-run", 1, 0), _round("live-run", 2, 1)

    with client.websocket_connect("/ws/live") as ws:
        # Idle past several heartbeat periods so pings interleave with events.
        time.sleep(0.15)
        publish(ev1).result(timeout=5)
        publish(ev2).result(timeout=5)

        got_events: list[dict] = []
        got_pings = 0
        for _ in range(50):
            frame = json.loads(ws.receive_text())  # every frame must be standalone JSON
            if frame.get("type") == "ping":
                assert set(frame) == {"type", "ts"}
                got_pings += 1
                continue
            got_events.append(frame)
            if len(got_events) == 2:
                break

        assert got_pings >= 1
        assert [e["round"] for e in got_events] == [1, 2]
        # Broadcast frames are the exact wire form of the published events.
        assert got_events[0] == RoundEvent.model_validate(ev1).model_dump(
            mode="json", by_alias=True
        )
        assert got_events[1]["global"]["macro_f1"] == pytest.approx(0.64)


def test_ws_broadcasts_to_all_clients_and_cleans_up(client: TestClient) -> None:
    hub = client.app.state.hub
    publish = client.app.state.publish_threadsafe
    ev = _round("live-run", 5, 5)

    with client.websocket_connect("/ws/live") as a, client.websocket_connect("/ws/live") as b:
        assert hub.subscriber_count == 2
        publish(ev).result(timeout=5)

        def next_event(ws) -> dict:
            for _ in range(50):
                frame = json.loads(ws.receive_text())
                if frame.get("type") != "ping":
                    return frame
            raise AssertionError("no event frame received")

        assert next_event(a)["round"] == 5
        assert next_event(b)["round"] == 5

    # Disconnect handling is asynchronous; the hub must drop both subscribers.
    deadline = time.time() + 2
    while hub.subscriber_count and time.time() < deadline:
        time.sleep(0.01)
    assert hub.subscriber_count == 0


def test_publish_rejects_malformed_events(client: TestClient) -> None:
    from pydantic import ValidationError

    bad = dict(CONTRACT_EVENT, v=2)  # version literal must not coerce
    with pytest.raises(ValidationError):
        client.app.state.publish_threadsafe(bad).result(timeout=5)


# -- schema stability ------------------------------------------------------


def test_round_event_round_trip_stays_byte_stable() -> None:
    ev = RoundEvent.model_validate(CONTRACT_EVENT)
    assert ev.model_dump(mode="json", by_alias=True) == CONTRACT_EVENT
    assert json.loads(ev.model_dump_json(by_alias=True)) == CONTRACT_EVENT
    assert RoundEvent.model_validate_json(ev.model_dump_json(by_alias=True)) == ev


def test_frontend_mock_fixture_is_wire_valid() -> None:
    """The console's mock fixture must stay in lockstep with the pydantic schema.

    This is the cross-language tripwire: if either side of the RoundEvent
    contract drifts, the committed frontend fixture stops round-tripping here.
    """
    fixture = REPO_ROOT / "frontend" / "src" / "fixtures" / "mock-run.json"
    events = json.loads(fixture.read_text(encoding="utf-8"))
    assert len(events) == 10
    assert [e["round"] for e in events] == list(range(1, 11))
    for raw in events:
        ev = RoundEvent.model_validate(raw)
        assert ev.model_dump(mode="json", by_alias=True) == raw
    # The demo story: the adaptive attacker evades the threshold, sign-flip never does.
    adaptive = [c for e in events for c in e["clients"] if c["attack"] == "adaptive_cosine"]
    signflip = [c for e in events for c in e["clients"] if c["attack"] == "sign_flip"]
    assert adaptive and all(c["accepted"] for c in adaptive)
    assert signflip and not any(c["accepted"] for c in signflip)
