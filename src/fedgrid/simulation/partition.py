"""Federation builder: K clients, non-IID by feeder identity + Dirichlet label skew. Protocol: Dirichlet(alpha) label skew per Hsu, Qi,
Brown (arXiv:1909.06335) - the community-standard non-IID protocol - combined
with per-client feeder/load-shape identity (natural feature skew). Windows of
T = 64 one-minute samples are cut with stride 16, giving 87 windows per
simulated day and ~2.6k windows per client for the 30-day default - inside the
contract's 1.5-3k per-client budget - with the event rate tuned so the
window-level prior stays NORMAL >= 90 % (realism block).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fedgrid.data import (
    N_CLASSES,
    WINDOW_T,
    ClientDataset,
    ClientMeta,
    Federation,
)

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "STRIDE",
    "client_feeder_spec",
    "make_federation",
    "sim_calendar_day",
    "windows_from_frame",
]

# Window stride in samples: (1440 - 64) // 16 + 1 = 87 windows/day.
STRIDE: int = 16
# A window takes an event's label when >= 2 of its samples are event samples
# (all event durations are >= 2 samples, so no event can fall through).
_MIN_EVENT_SAMPLES: int = 2
# Client-day d simulates calendar day 3 + 12*d: 30 days then span the full year
# (seasonal structure appears in the *data*, not just the model) and 12 mod 7
# cycles every weekday/weekend residue. Deterministic, documented in REALISM.md.
_CAL_OFFSET, _CAL_SPACING = 3, 12
# PV penetration levels alternate across clients: moderate / high (Hoke et al.
# 2013; see grid.FeederSpec docstring and REALISM.md).
_PV_LEVELS: tuple[float, float] = (0.15, 0.40)
_FED_TAG = 0x0FED0001
# Per-client load-shape seed spread (large prime keeps clients' load draws and
# split permutations well separated across federation seeds).
_LOAD_SEED_STRIDE = 100_003


def _window_label(lbl: np.ndarray) -> int:
    """Modal non-NORMAL class if it covers >= _MIN_EVENT_SAMPLES, else NORMAL."""
    counts = np.bincount(lbl, minlength=N_CLASSES)
    counts[0] = 0
    top = int(counts.argmax())
    return top if counts[top] >= _MIN_EVENT_SAMPLES else 0


def windows_from_frame(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Cut one day-frame into (X (n, 11, 64) float32, y (n,) int64) windows."""
    from fedgrid.data import CHANNELS

    arr = np.ascontiguousarray(df[list(CHANNELS)].to_numpy(dtype=np.float32).T)
    lbl = df["label"].to_numpy(dtype=np.int64)
    n = arr.shape[1]
    starts = range(0, n - WINDOW_T + 1, STRIDE)
    xs = np.stack([arr[:, s:s + WINDOW_T] for s in starts])
    ys = np.array([_window_label(lbl[s:s + WINDOW_T]) for s in starts], dtype=np.int64)
    return xs, ys


def sim_calendar_day(day: int) -> int:
    """Calendar day simulated for the ``day``-th client-day.

    Client-day d maps to 3 + 12d so a 30-day run spans the full year (seasonal
    structure present in the data) and cycles every weekday/weekend residue.
    """
    return _CAL_OFFSET + _CAL_SPACING * day


def client_feeder_spec(seed: int, cid: int):
    """Deterministic feeder identity for client ``cid`` of federation ``seed``.

    Exposed (and used internally) so the exact battery cases can be
    reconstructed - e.g. by regression tests - without re-deriving the seeding.
    """
    from fedgrid.simulation.grid import FeederSpec, available_feeders

    feeders = available_feeders()
    return FeederSpec(
        name=feeders[cid % len(feeders)],
        load_shape_seed=seed * _LOAD_SEED_STRIDE + cid,
        pv_penetration=_PV_LEVELS[cid % len(_PV_LEVELS)],
    )


def make_federation(k: int = 10, alpha: float = 0.5, days: int = 30, seed: int = 0) -> Federation:
    """Builds K :class:`~fedgrid.data.ClientDataset` objects.

    Dirichlet(alpha) drives per-client class_weights; feeder specs cycle through
    :func:`~fedgrid.simulation.grid.available_feeders`. Bit-reproducible for a
    fixed ``seed`` (per-client/day RNG streams are derived from it and OpenDSS
    is deterministic for identical command sequences on one platform). Solver
    non-convergence is tolerated per grid.py's documented policy and the total
    count is recorded on each :class:`~fedgrid.data.ClientMeta`.
    """
    from fedgrid.simulation.events import sample_event_plan
    from fedgrid.simulation.grid import FeederSim

    if k < 1 or days < 1:
        raise ValueError("k and days must be >= 1")
    rng = np.random.default_rng(np.random.SeedSequence([_FED_TAG, seed]))
    class_weights = rng.dirichlet(alpha * np.ones(N_CLASSES), size=k)

    clients: list[ClientDataset] = []
    for cid in range(k):
        spec = client_feeder_spec(seed, cid)
        sim = FeederSim(spec)
        ev_rng = np.random.default_rng(np.random.SeedSequence([_FED_TAG, seed, cid, 1]))
        xs_all: list[np.ndarray] = []
        ys_all: list[np.ndarray] = []
        n_nonconv = n_steps = 0
        for day in range(days):
            plan = sample_event_plan(ev_rng, 86400.0, class_weights[cid])
            df = sim.run_day(sim_calendar_day(day), plan)
            n_nonconv += int(df.attrs.get("n_nonconverged", 0))
            n_steps += int(df.attrs.get("n_steps", 0))
            xs, ys = windows_from_frame(df)
            xs_all.append(xs)
            ys_all.append(ys)
        meta = ClientMeta(
            cid=cid,
            feeder=spec,
            class_weights=class_weights[cid].copy(),
            days=days,
            split_seed=seed * _LOAD_SEED_STRIDE + 7919 * (cid + 1),
            n_nonconverged=n_nonconv,
            n_steps=n_steps,
        )
        clients.append(ClientDataset(np.concatenate(xs_all), np.concatenate(ys_all), meta))
    return Federation(clients, alpha=alpha, seed=seed, days=days)
