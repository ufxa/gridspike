"""FastAPI application factory for the FedGrid backend.

phase 2 status (the API module): run-store endpoints and the ``/ws/live`` broadcast hub
are real; ``POST /api/experiments`` stays 501 until the phase 3 runner exists.

The phase 3 runner publishes rounds through the app, not the socket layer:
``app.state.publish`` (awaitable, for callers already on the server loop) and
``app.state.publish_threadsafe`` (for the training thread — returns a
``concurrent.futures.Future``). Both fan out to every connected console and
append to the store's live ring buffer that backs ``GET /api/live/latest``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
from collections import deque
from concurrent.futures import Future
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, WebSocket
from starlette.websockets import WebSocketDisconnect

from fedgrid import __version__
from fedgrid.api.schemas import RoundEvent

__all__ = ["RunStore", "create_app"]

log = logging.getLogger(__name__)

# Run ids come from URL segments; constraining them to filename-safe characters
# closes path traversal without having to reason about OS-specific separators.
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _iso_z(dt: datetime) -> str:
    """Wire timestamp convention: ISO-8601 UTC with a literal ``Z`` suffix."""
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


class RunStore:
    """Repository of persisted RunResults backing ``/api/runs``.

    Filesystem-backed: one ``{run_id}.json`` per run under ``runs_dir`` (written
    by :func:`fedgrid.metrics.runresult.save_run` from phase 3 on — the store
    only ever reads). Also owns the in-memory ring buffer of live RoundEvents so
    a console that (re)connects mid-run can catch up via ``/api/live/latest``.
    """

    def __init__(self, runs_dir: str | Path = "runs", live_buffer: int = 256) -> None:
        self.runs_dir = Path(runs_dir)
        self._live: deque[dict] = deque(maxlen=live_buffer)

    # -- persisted runs ---------------------------------------------------

    def list_runs(self) -> list[dict]:
        """Summaries (method, seed, final acc/macro-F1, timestamps), newest first.

        Unreadable files are skipped with a warning rather than failing the
        listing: one corrupt run must not take the console down.
        """
        if not self.runs_dir.is_dir():
            return []
        summaries: list[dict] = []
        for path in sorted(self.runs_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                summaries.append(self._summarize(path, data))
            except (OSError, ValueError) as exc:
                log.warning("skipping unreadable run file %s: %s", path, exc)
        summaries.sort(key=lambda s: (s["created_at"], s["run_id"]), reverse=True)
        return summaries

    def get_run(self, run_id: str) -> dict:
        """Full RunResult JSON for one run id.

        Raises ``KeyError`` for unknown or unsafe ids (mapped to 404 by the API).
        """
        if not _RUN_ID_RE.fullmatch(run_id):
            raise KeyError(run_id)
        path = self.runs_dir / f"{run_id}.json"
        if not path.is_file():
            raise KeyError(run_id)
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _summarize(path: Path, data: dict) -> dict:
        rounds = data.get("rounds") or []
        final = data.get("final") or {}
        # rounds[] dicts carry RoundEvent fields, so "ts" bounds the run in time;
        # file mtime is the fallback ordering key when rounds are empty.
        first_ts = rounds[0].get("ts") if rounds else None
        last_ts = rounds[-1].get("ts") if rounds else None
        return {
            "run_id": path.stem,
            "method": data.get("method"),
            "seed": data.get("seed"),
            "n_rounds": len(rounds),
            "final_acc": final.get("acc"),
            "final_macro_f1": final.get("macro_f1"),
            "started_at": first_ts,
            "finished_at": last_ts,
            "created_at": _iso_z(datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)),
        }

    # -- live-run buffer ---------------------------------------------------

    def push_live(self, event: dict) -> None:
        """Append one wire-shaped RoundEvent dict; oldest entries roll off."""
        self._live.append(event)

    def latest_live(self, n: int = 50) -> list[dict]:
        """The last ``n`` buffered events, oldest first (console catch-up order)."""
        if n <= 0:
            return []
        return list(self._live)[-n:]


class LiveHub:
    """Fan-out of published RoundEvents to every connected ``/ws/live`` client.

    Each subscriber owns an unbounded queue so one slow consumer never blocks
    the publisher or its peers; late joiners recover history from the store's
    ring buffer instead of from the socket.
    """

    def __init__(self, store: RunStore) -> None:
        self._store = store
        self._subscribers: set[asyncio.Queue[str]] = set()

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def subscribe(self) -> asyncio.Queue[str]:
        queue: asyncio.Queue[str] = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[str]) -> None:
        self._subscribers.discard(queue)

    async def publish(self, event: RoundEvent | dict) -> None:
        """Validate, buffer, and broadcast one round event.

        Dicts are accepted so the phase 3 runner can hand over raw round records;
        validation here means a malformed event fails loudly at the publisher,
        never as a half-written frame on a client.
        """
        ev = event if isinstance(event, RoundEvent) else RoundEvent.model_validate(event)
        wire = ev.model_dump_json(by_alias=True)
        # The buffer keeps wire-shaped dicts so /api/live/latest replays exactly
        # what the sockets carried ("global" alias included).
        self._store.push_live(json.loads(wire))
        for queue in tuple(self._subscribers):
            queue.put_nowait(wire)


def _ping_frame() -> str:
    """In-band heartbeat; consoles drop frames whose ``type`` is ``ping``."""
    return json.dumps({"type": "ping", "ts": _iso_z(datetime.now(UTC))})


def create_app(store: RunStore, *, heartbeat_s: float = 20.0) -> FastAPI:
    """Build the API around a run store; the store is injected so tests can fake it.

    ``heartbeat_s`` is keyword-only and test-facing: production keeps the 20 s
    contract default, tests shrink it to observe pings without waiting.
    """
    hub = LiveHub(store)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Captured so publish_threadsafe can hop from the training thread onto
        # the server loop; only valid while the app is running.
        app.state.loop = asyncio.get_running_loop()
        yield
        app.state.loop = None

    app = FastAPI(title="FedGrid", version=__version__, lifespan=lifespan)
    app.state.store = store
    app.state.hub = hub
    app.state.publish = hub.publish

    def publish_threadsafe(event: RoundEvent | dict) -> Future[None]:
        loop = getattr(app.state, "loop", None)
        if loop is None:
            raise RuntimeError("publish_threadsafe requires a running app (lifespan not started)")
        return asyncio.run_coroutine_threadsafe(hub.publish(event), loop)

    app.state.publish_threadsafe = publish_threadsafe

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/runs")
    def list_runs() -> list[dict]:
        return store.list_runs()

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        try:
            return store.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}") from exc

    @app.post("/api/experiments")
    def launch_experiment() -> dict:
        raise HTTPException(status_code=501, detail="Not implemented until phase 3 (the API module)")

    @app.get("/api/live/latest")
    def live_latest(n: Annotated[int, Query(ge=1, le=1000)] = 50) -> list[dict]:
        return store.latest_live(n)

    @app.websocket("/ws/live")
    async def ws_live(websocket: WebSocket) -> None:
        await websocket.accept()
        queue = hub.subscribe()

        async def sender() -> None:
            while True:
                try:
                    frame = await asyncio.wait_for(queue.get(), timeout=heartbeat_s)
                except TimeoutError:
                    frame = _ping_frame()
                await websocket.send_text(frame)

        async def receiver() -> None:
            # Consoles never send application data; this exists purely to see
            # the close frame promptly instead of at the next failed send.
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return

        tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in pending:
                with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect):
                    await task
            for task in done:
                # Retrieve results so a send-after-close never surfaces as an
                # "exception was never retrieved" warning; disconnects are the
                # expected way for either task to end.
                with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                    task.result()
        finally:
            hub.unsubscribe(queue)

    return app


def main() -> None:
    """Dev entrypoint: ``python -m fedgrid.api.app`` serves ./runs on :8000."""
    import uvicorn

    uvicorn.run(create_app(RunStore()), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
