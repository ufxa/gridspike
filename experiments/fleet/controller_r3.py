"""Round-3 Vast.ai fleet controller -- 534-run temporal + magnitude battery.

This is the *operator* for round 3: it rents boxes, deploys work, collects results
and destroys everything when the work (or the money) runs out. It is written
against :mod:`experiments.fleet.fleet` (search/create/destroy/ssh primitives) and
:file:`experiments/fleet/run_shard.sh` (the worker loop, unchanged).

SAFETY REQUIREMENT #1 (non-negotiable)
--------------------------------------
The Vast account also carries the owner's **production** instances (labels
instances owned by other tenants). They are NOT ours. This
controller therefore treats the instance label as a hard capability:

* every instance it creates is labelled **exactly** ``fedgrid-r3``;
* every listing, adoption, probe, deploy and destroy goes through the single pure
  filter :func:`select_ours` (exact, case-sensitive, type-strict string equality);
* anything else is *invisible*: never listed for action, never probed, never
  destroyed -- not even by the spend-cap emergency teardown.

:func:`select_ours` is a pure function over the raw API rows and is unit tested
(``tests/test_fleet_r3.py``); :meth:`ControllerR3._destroy` re-checks the label on
the raw row immediately before issuing the destroy, so a bug upstream still cannot
reach a production box.

Operational lessons baked in (each one cost money or a night in an earlier round)
--------------------------------------------------------------------------------
1.  Binary ``.venv/Scripts/vastai.exe``; dedicated key ``~/.ssh/vast_fedgrid``
    (generated + registered on the account when missing).
2.  Offers: {RTX 3090, 3090 Ti, 4080, 4090}, verified, rentable, <= $0.60/hr,
    disk >= 32 GB, inet_down >= 200 Mbps, image ``vastai/pytorch``.
3.  A booting box gets **13 minutes of patience** before any strike is counted;
    readiness is a *light* ``ssh true`` probe -- never ``import torch``.
4.  Deploy: the ~700 KB code bundle is **always** re-pushed; data is **sharded**
    (only the federations this box's runs need, ~9 MB each) instead of the 120 MB
    battery; at most :data:`NET_CONCURRENCY` heavy scp/ssh run at once; the
    federation directory is *verified present on the box* before it counts as
    deployed; the wheel is installed ``--no-deps``; the previous shard is killed
    with a bracketed pattern (``[r]un_shard``) issued in its **own** ssh call so it
    can never match -- and kill -- its own shell; the start is confirmed with
    ``pgrep``, never with a captured ``echo`` (ssh drops stdout often enough).
5.  Box states from one cheap probe: BUSY / IDLE / DEAD, plus deploy-fail strikes
    so a permanently broken box is destroyed instead of billing forever.
6.  A separate collector thread pulls new ``<run_id>.json`` home every 30 s, so a
    box dying loses at most the run it was mid-way through.
7.  Money is measured from the **real account balance** (snapshot persisted, so it
    survives a controller restart), minus a configurable franchise for the owner's
    production burn (``--external-burn-per-h``, default $0.40/hr).
8.  Work-stealing is by *federation*: the shard planner keeps whole federations on
    one box so a re-deploy rarely re-uploads data. Progressive pruning destroys
    idle boxes once the remaining work is smaller than the fleet.

Usage::

    python -m experiments.fleet.controller_r3 \
        --manifest experiments/manifest_temporal.jsonl \
        --manifest experiments/manifest_magnitude.jsonl \
        --target 14 --cap 15 --out-dir runs_r3 --data-dir data_battery

    python -m experiments.fleet.controller_r3 --manifest ... --dry-run   # plan only
    python -m experiments.fleet.controller_r3 --status                   # what is up
    python -m experiments.fleet.controller_r3 --teardown                 # kill switch

Windows notes: every path is a :class:`~pathlib.Path`, every ``open`` and every
subprocess pipe is explicitly UTF-8, no subprocess uses ``shell=True``, and the
console output is pure ASCII (the Windows console is cp1252).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ``experiments`` lives at the repo root; make the package importable however this
# module was invoked (``-m``, direct path, or from a test).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.fleet.fleet import (  # noqa: E402
    REMOTE_BUNDLE_DIR,
    REMOTE_DATA_DIR,
    REMOTE_RESULTS_DIR,
    REMOTE_WORKSPACE,
    CliVastBackend,
    Instance,
    Offer,
    SshUnavailableError,
    parse_instance,
    parse_offer,
    validate_bundle,
)
from experiments.stage_data import federation_id  # noqa: E402

__all__ = [
    "LABEL",
    "BoxProbe",
    "BoxState",
    "ControllerConfig",
    "ControllerR3",
    "Shard",
    "assign_shards_to_boxes",
    "build_query_r3",
    "classify_box",
    "desired_box_count",
    "entry_fed_id",
    "federation_id",
    "filter_offers_r3",
    "find_wheel",
    "group_by_federation",
    "is_ours",
    "load_manifest_entries",
    "main",
    "our_instance_ids",
    "plan_shards",
    "rank_offers_r3",
    "select_ours",
    "spend_from_balance",
]

# --------------------------------------------------------------------------- #
# THE label. Hard-coded on purpose: it is a safety boundary, not a preference.
# --------------------------------------------------------------------------- #

#: The ONLY label this controller will ever create, list, probe or destroy.
LABEL = "fedgrid-r3"

# -- provisioning contract ----------------------------------------------------
#: Query token form (Vast wants spaces as underscores inside ``in [...]``).
GPU_QUERY_NAMES = ("RTX_3090", "RTX_3090_Ti", "RTX_4080", "RTX_4090")
#: Normalised names accepted client-side (lowercased, underscores -> spaces).
GPU_ALLOWED = ("rtx 3090", "rtx 3090 ti", "rtx 4080", "rtx 4090")
PRICE_CAP_HR = 0.60
DISK_GB = 32
MIN_INET_DOWN = 200.0
MIN_DISK_SPACE = 32.0
#: ``vastai/pytorch`` ships torch + CUDA and (unlike ``pytorch/pytorch``) an sshd,
#: which Vast's SSH launch mode needs. Unpinned: round 3 only needs py>=3.10.
IMAGE = "vastai/pytorch"

# -- timing / patience --------------------------------------------------------
#: A box that is still "loading" gets this long before ANY strike is counted.
BOOT_GRACE_S = 13 * 60.0
#: ... and this long before we give up on it entirely and destroy it.
BOOT_HARD_TIMEOUT_S = 28 * 60.0
#: Shard log untouched for this long => the box is not really working.
LOG_STALE_S = 180.0
#: Consecutive ssh probe failures that make a box DEAD.
SSH_STRIKES_MAX = 6
#: Consecutive deploy failures that make a box DEAD (a broken box bills forever).
DEPLOY_STRIKES_MAX = 6
#: Heavy transfers (scp / tar-extract) allowed at once. Saturating the operator's
#: uplink took the whole fleet down once.
NET_CONCURRENCY = 4
CYCLE_S = 45.0
COLLECT_S = 30.0
#: Grace between launching a shard and asking pgrep whether it is really there.
START_VERIFY_DELAY_S = 8.0
#: sshd probe budget per attempt. Deliberately short: a dead box must not stall a
#: cycle for the 15 minutes the fleet default would allow.
SSH_READY_BUDGET_S = 30.0

# -- remote layout (shared with run_shard.sh) ---------------------------------
REMOTE_STAGE_DIR = f"{REMOTE_WORKSPACE}/stage"
REMOTE_SHARD_LOG = f"{REMOTE_WORKSPACE}/shard.log"
REMOTE_STATUS_PY = f"{REMOTE_WORKSPACE}/_status.py"
REMOTE_PULL_TAR = f"{REMOTE_WORKSPACE}/_new_results.tgz"

DEFAULT_SSH_KEY = Path("~/.ssh/vast_fedgrid").expanduser()
DEFAULT_VASTAI_BIN = REPO_ROOT / ".venv" / "Scripts" / "vastai.exe"
VASTAI_KEY_FILE = Path("~/.config/vastai/vast_api_key").expanduser()

#: The dependency-free status merger ``run_shard.sh`` calls after every run. It is
#: normally written by ``onstart.sh``; shipping it in the bundle too means the
#: heartbeat works even when onstart was replaced or failed early.
STATUS_PY_SOURCE = '''\
import json, os, sys, tempfile, time
path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
except (OSError, ValueError):
    data = {}
def coerce(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v in ("true", "false"):
        return v == "true"
    if v in ("null", ""):
        return None
    return v
for pair in sys.argv[2:]:
    k, _, v = pair.partition("=")
    data[k] = coerce(v)
data.setdefault("instance_id", int(os.environ.get("CONTAINER_ID", 0) or 0))
if "heartbeat_ts" not in {p.split("=", 1)[0] for p in sys.argv[2:]}:
    data["heartbeat_ts"] = time.time()
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".")
with os.fdopen(fd, "w", encoding="utf-8") as fh:
    json.dump(data, fh, sort_keys=True)
os.replace(tmp, path)
'''


# =========================================================================== #
# PURE FUNCTIONS -- no network, no filesystem, unit tested in tests/test_fleet_r3
# =========================================================================== #


def is_ours(row: Mapping[str, Any]) -> bool:
    """``True`` iff this raw Vast instance row is one WE created.

    Exact, case-sensitive, type-strict equality against :data:`LABEL`. No
    stripping, no prefix match, no case folding: a near-miss label means *not
    ours*, which fails in the safe direction (we may leak a box we created, we can
    never touch a box we did not).
    """
    if not isinstance(row, Mapping):
        return False
    label = row.get("label")
    return isinstance(label, str) and label == LABEL


def select_ours(rows: Iterable[Mapping[str, Any]]) -> list[dict]:
    """THE safety filter: the fedgrid-r3 subset of an ``show instances`` payload.

    Everything the controller does -- listing, adopting, probing, deploying,
    destroying, the emergency spend-cap teardown -- flows through this function.
    Instances the owner runs in production (``other-project-a``, ``other-project-b``,
    non-project instances) are dropped here and are invisible to the rest of the module.
    """
    return [dict(row) for row in rows or [] if is_ours(row)]


def our_instance_ids(rows: Iterable[Mapping[str, Any]]) -> list[int]:
    """Instance ids of the fedgrid-r3 subset (convenience over :func:`select_ours`)."""
    out: list[int] = []
    for row in select_ours(rows):
        try:
            out.append(int(row.get("id") or 0))
        except (TypeError, ValueError):
            continue
    return [i for i in out if i > 0]


# -- manifest / federation ----------------------------------------------------


def load_manifest_entries(paths: Sequence[str | Path]) -> list[dict]:
    """Concatenate one or more ``*.jsonl`` manifests, de-duplicated by ``run_id``.

    Order is first-seen; a ``run_id`` appearing in two manifests is kept once (the
    first occurrence wins) so ``--manifest a --manifest b`` is safe even if the two
    files overlap.
    """
    seen: dict[str, dict] = {}
    for path in paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"manifest not found: {p}")
        with p.open(encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{p}:{lineno}: not valid JSON ({exc})") from exc
                rid = str(entry["run_id"])
                seen.setdefault(rid, entry)
    return list(seen.values())


def entry_fed_id(entry: Mapping[str, Any]) -> str:
    """The on-disk federation id for a manifest entry (``fed_<sha256[:10]>``).

    Delegates to :func:`experiments.stage_data.federation_id` -- the module that
    *created* ``data_battery/<fed_id>/`` -- so the controller can never disagree
    with the staging layout about which directory a run needs.
    """
    return federation_id(dict(entry["config"]["federation"]))


def group_by_federation(entries: Iterable[Mapping[str, Any]]) -> dict[str, list[dict]]:
    """Manifest entries bucketed by federation id, in first-seen order."""
    groups: dict[str, list[dict]] = {}
    for entry in entries:
        groups.setdefault(entry_fed_id(entry), []).append(dict(entry))
    return groups


@dataclass(frozen=True)
class Shard:
    """One box's worth of work: whole runs plus the federations they need."""

    fed_ids: tuple[str, ...]
    run_ids: tuple[str, ...]
    entries: tuple[Mapping[str, Any], ...] = field(default=(), repr=False)

    def __len__(self) -> int:
        return len(self.run_ids)


def _chunk(seq: Sequence[Any], n_chunks: int) -> list[list[Any]]:
    """Split ``seq`` into ``n_chunks`` contiguous, near-equal, non-empty chunks."""
    n_chunks = max(1, min(n_chunks, len(seq)))
    size, extra = divmod(len(seq), n_chunks)
    out: list[list[Any]] = []
    start = 0
    for i in range(n_chunks):
        stop = start + size + (1 if i < extra else 0)
        out.append(list(seq[start:stop]))
        start = stop
    return out


def plan_shards(entries: Sequence[Mapping[str, Any]], n_boxes: int) -> list[Shard]:
    """Pack ``entries`` into at most ``n_boxes`` shards, grouped by federation.

    Two forces pull against each other:

    * **Data transfer** -- a federation is ~9 MB and must be on the box before its
      runs can start, so splitting a federation across boxes costs an extra upload.
    * **Makespan** -- we pay for the whole fleet until the *last* box finishes, and
      a box's queue is claimed work that idle boxes cannot steal. An 8-run box next
      to a 108-run box wastes most of the fleet.

    Makespan wins, because it is the thing we rent by the hour. The planner
    therefore *allocates boxes to federations* rather than cutting fixed-size
    pieces:

    1. each federation gets a share of the ``n_boxes`` proportional to its run
       count (largest-remainder rounding), so a federation with three times the
       work gets three times the boxes;
    2. a federation with at least one box is split into that many near-equal
       contiguous chunks -- those shards each carry exactly one federation;
    3. federations too small to earn a box are appended whole to the least-loaded
       shard (preferring one that already holds that federation), so a small
       dataset is never split and never strands a box on 8 runs.

    Cutting fixed ``ceil(total / n)`` pieces instead (the obvious approach) leaves
    a few boxes holding two pieces and doubles the makespan: with 534 runs over 14
    boxes it produced 62-run boxes next to 31-run ones. Proportional allocation
    keeps the spread inside ~10%.

    Total, disjoint and order-preserving: the union of the shards' run ids is
    exactly the input's, with no duplicates.
    """
    entries = [dict(e) for e in entries]
    if not entries or n_boxes <= 0:
        return []
    groups = group_by_federation(entries)
    total = len(entries)
    n_shards = min(n_boxes, total)

    # -- 1. largest-remainder proportional allocation of boxes to federations --
    exact = {fed: len(runs) * n_shards / total for fed, runs in groups.items()}
    quota = {fed: min(int(v), len(groups[fed])) for fed, v in exact.items()}
    left = n_shards - sum(quota.values())
    if left > 0:
        order = sorted(groups, key=lambda f: (-(exact[f] % 1.0), -len(groups[f]), f))
        idx = guard = 0
        while left > 0 and guard < 10 * n_shards + 10:
            guard += 1
            fed = order[idx % len(order)]
            idx += 1
            if quota[fed] < len(groups[fed]):
                quota[fed] += 1
                left -= 1

    # -- 2. dedicated, single-federation shards --------------------------------
    shard_runs: list[list[dict]] = []
    leftovers: list[list[dict]] = []
    for fed, runs in groups.items():
        if quota[fed] >= 1:
            shard_runs.extend(c for c in _chunk(runs, quota[fed]) if c)
        else:
            leftovers.append(list(runs))
    if not shard_runs:  # pragma: no cover -- sum(quota) == n_shards >= 1
        shard_runs = [[]]

    # -- 3. small federations ride along on the least-loaded shard -------------
    shard_feds = [{entry_fed_id(e) for e in runs} for runs in shard_runs]
    for piece in sorted(leftovers, key=lambda p: (-len(p), entry_fed_id(p[0]))):
        fed = entry_fed_id(piece[0])
        target = min(
            range(len(shard_runs)),
            key=lambda i: (len(shard_runs[i]), 0 if fed in shard_feds[i] else 1, i),
        )
        shard_runs[target].extend(piece)
        shard_feds[target].add(fed)
    return [_shard_from(runs) for runs in shard_runs if runs]


def _shard_from(entries: Sequence[Mapping[str, Any]]) -> Shard:
    feds: list[str] = []
    for e in entries:
        fed = entry_fed_id(e)
        if fed not in feds:
            feds.append(fed)
    return Shard(
        fed_ids=tuple(feds),
        run_ids=tuple(str(e["run_id"]) for e in entries),
        entries=tuple(dict(e) for e in entries),
    )


def assign_shards_to_boxes(
    shards: Sequence[Shard], boxes: Sequence[int], staged: Mapping[int, Iterable[str]]
) -> dict[int, Shard]:
    """Match shards to free boxes, preferring boxes that already hold the data.

    ``staged`` maps instance id -> federation ids verified present on that box. A
    box that already has the shard's federation skips a ~9 MB upload, which is the
    whole point of federation-grouped shards.
    """
    free = list(boxes)
    plan: dict[int, Shard] = {}
    for shard in shards:
        if not free:
            break
        want = set(shard.fed_ids)
        best = max(
            free,
            key=lambda bid: (len(want & set(staged.get(bid, ()))), -bid),
        )
        free.remove(best)
        plan[best] = shard
    return plan


# -- offers -------------------------------------------------------------------


def build_query_r3(
    *,
    price_cap: float = PRICE_CAP_HR,
    disk_gb: int = DISK_GB,
    min_inet_down: float = MIN_INET_DOWN,
) -> str:
    """The ``vastai search offers`` query for the round-3 GPU set."""
    names = ",".join(GPU_QUERY_NAMES)
    return (
        f"gpu_name in [{names}] num_gpus=1 rentable=true verified=true "
        f"disk_space>={disk_gb} inet_down>={int(min_inet_down)} "
        f"dph_total<={price_cap}"
    )


def filter_offers_r3(
    offers: Iterable[Offer],
    *,
    price_cap: float = PRICE_CAP_HR,
    min_inet_down: float = MIN_INET_DOWN,
    min_disk_gb: float = MIN_DISK_SPACE,
    exclude_machine_ids: Iterable[int] = (),
) -> list[Offer]:
    """Client-side re-check of every hard filter (the query is not trusted alone)."""
    excluded = set(exclude_machine_ids)
    keep: list[Offer] = []
    for o in offers:
        if o.machine_id in excluded:
            continue
        if o.num_gpus != 1 or not o.verified:
            continue
        name = o.gpu_name.replace("_", " ").strip().lower()
        if name not in GPU_ALLOWED:
            continue
        if o.dph_total <= 0 or o.dph_total > price_cap:
            continue
        if o.inet_down_mbps < min_inet_down:
            continue
        if float(o.raw.get("disk_space") or 0.0) < min_disk_gb:
            continue
        keep.append(o)
    return keep


def rank_offers_r3(offers: Iterable[Offer]) -> list[Offer]:
    """Cheapest first; ties to higher reliability, then faster downlink."""
    return sorted(offers, key=lambda o: (o.dph_total, -o.reliability, -o.inet_down_mbps))


def desired_box_count(target: int, pending_runs: int) -> int:
    """Never rent more boxes than there is work: ``min(target, pending)``."""
    return max(0, min(int(target), int(pending_runs)))


# -- money --------------------------------------------------------------------


def spend_from_balance(
    balance_start: float,
    balance_now: float,
    elapsed_h: float,
    external_burn_per_h: float,
) -> float:
    """Our spend = balance drop MINUS the owner's production franchise.

    The account balance also falls because of the owner's production instances
    (~$0.36/hr for ``other-project-a`` + ``other-project-b``). Attributing that to us
    would trip the cap early and destroy a healthy fleet, so a flat franchise of
    ``external_burn_per_h`` is subtracted. Clamped at 0: a top-up between snapshots
    must never read as negative spend.

    Balance-based (not rate x uptime) on purpose: it survives a controller restart,
    a missed cycle, and Vast's storage/bandwidth line items.
    """
    drop = float(balance_start) - float(balance_now)
    franchise = float(external_burn_per_h) * max(0.0, float(elapsed_h))
    return max(0.0, drop - franchise)


# -- box state ----------------------------------------------------------------


@dataclass(frozen=True)
class BoxProbe:
    """One cheap SSH probe's worth of facts about a box."""

    ok: bool = False
    run_one_alive: bool = False
    shard_alive: bool = False
    feds_present: tuple[str, ...] = ()
    results: int = 0
    log_age_s: float | None = None
    shard_done: bool = False
    last_log_line: str = ""
    error: str = ""

    def data_ok(self, required: Iterable[str]) -> bool:
        return set(required) <= set(self.feds_present)


def classify_box(
    probe: BoxProbe,
    *,
    required_feds: Iterable[str] = (),
    ssh_fails: int = 0,
    deploy_fails: int = 0,
    log_stale_s: float = LOG_STALE_S,
) -> str:
    """``"BUSY"`` | ``"IDLE"`` | ``"DEAD"`` | ``"UNKNOWN"`` from one probe.

    * **DEAD** -- ``SSH_STRIKES_MAX`` consecutive ssh failures, or
      ``DEPLOY_STRIKES_MAX`` consecutive deploy failures (a box that can never be
      deployed to would otherwise bill forever).
    * **BUSY** -- ``run_one`` alive AND its data present AND the shard log touched
      within ``log_stale_s``.
    * **IDLE** -- shard finished, or no ``run_one``, or the log stalled, or the box
      is starved of data. IDLE boxes get re-deployed with the next work.
    * **UNKNOWN** -- this probe failed but the strike budget is not spent; the box
      is left alone this cycle (it is probably mid-reboot).
    """
    if ssh_fails >= SSH_STRIKES_MAX or deploy_fails >= DEPLOY_STRIKES_MAX:
        return "DEAD"
    if not probe.ok:
        return "UNKNOWN"
    if probe.shard_done:
        return "IDLE"
    if not probe.data_ok(required_feds):
        return "IDLE"  # starved: data never landed
    if not probe.run_one_alive:
        return "IDLE"
    if probe.log_age_s is None or probe.log_age_s >= log_stale_s:
        return "IDLE"
    return "BUSY"


# =========================================================================== #
# Controller
# =========================================================================== #


@dataclass
class ControllerConfig:
    """Every knob, with the round-3 defaults."""

    manifests: tuple[Path, ...] = ()
    out_dir: Path = Path("runs_r3")
    data_dir: Path = Path("data_battery")
    target: int = 14
    cap_usd: float = 15.0
    external_burn_per_h: float = 0.40
    price_cap: float = PRICE_CAP_HR
    disk_gb: int = DISK_GB
    image: str = IMAGE
    cycle_s: float = CYCLE_S
    collect_s: float = COLLECT_S
    eta_min_per_run: float = 2.0
    ssh_key: Path = DEFAULT_SSH_KEY
    vastai_bin: Path = DEFAULT_VASTAI_BIN
    max_cycles: int = 100_000
    dry_run: bool = False


@dataclass
class BoxState:
    """The controller's bookkeeping for one rented box."""

    instance_id: int
    first_seen: float
    ssh_fails: int = 0
    deploy_fails: int = 0
    staged_feds: set[str] = field(default_factory=set)
    assigned: tuple[str, ...] = ()
    required_feds: tuple[str, ...] = ()
    deployed_ts: float | None = None
    state: str = "NEW"
    last_probe: BoxProbe | None = None

    def to_json(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "first_seen": self.first_seen,
            "ssh_fails": self.ssh_fails,
            "deploy_fails": self.deploy_fails,
            "staged_feds": sorted(self.staged_feds),
            "assigned": list(self.assigned),
            "required_feds": list(self.required_feds),
            "deployed_ts": self.deployed_ts,
            "state": self.state,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> BoxState:
        box = cls(int(data["instance_id"]), float(data.get("first_seen") or time.time()))
        box.ssh_fails = int(data.get("ssh_fails") or 0)
        box.deploy_fails = int(data.get("deploy_fails") or 0)
        box.staged_feds = set(data.get("staged_feds") or ())
        box.assigned = tuple(data.get("assigned") or ())
        box.required_feds = tuple(data.get("required_feds") or ())
        box.deployed_ts = data.get("deployed_ts")
        box.state = str(data.get("state") or "NEW")
        return box


class ControllerR3:
    """The round-3 operator loop over a Vast backend.

    Everything that reaches the network goes through ``self.backend`` (a
    :class:`~experiments.fleet.fleet.CliVastBackend` in production, a mock in
    tests); everything that decides is one of the pure functions above.
    """

    def __init__(
        self,
        backend: Any,
        entries: Sequence[Mapping[str, Any]],
        cfg: ControllerConfig,
    ) -> None:
        self.backend = backend
        self.entries = [dict(e) for e in entries]
        self.by_id = {str(e["run_id"]): e for e in self.entries}
        self.cfg = cfg
        self.out_dir = Path(cfg.out_dir)
        self.data_dir = Path(cfg.data_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir = self.out_dir / "_r3"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.out_dir / "controller_r3_state.json"
        self.boxes: dict[int, BoxState] = {}
        self.balance_start: float | None = None
        self.balance_start_ts: float | None = None
        self.exclude_machines: set[int] = set()
        #: Rate x uptime accrual, used ONLY when the balance API is unavailable --
        #: otherwise a broken 'show user' would silently disable the spend cap.
        self.rate_accrued: float = 0.0
        self._last_rate: float = 0.0
        self._last_rate_ts: float = time.time()
        self._net_sem = threading.BoundedSemaphore(NET_CONCURRENCY)
        self._lock = threading.RLock()
        self._pull_locks: dict[int, threading.Lock] = {}
        self._fed_locks: dict[str, threading.Lock] = {}
        self._stop = threading.Event()
        self._collector: threading.Thread | None = None
        self._cycle = 0
        self._started = time.time()
        self._load_state()

    # -- state persistence ---------------------------------------------------
    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            say(f"WARN: unreadable state at {self.state_path}; starting fresh")
            return
        if data.get("label") != LABEL:
            say(f"WARN: state at {self.state_path} is for label {data.get('label')!r}; ignoring")
            return
        self.balance_start = data.get("balance_start")
        self.balance_start_ts = data.get("balance_start_ts")
        self.rate_accrued = float(data.get("rate_accrued") or 0.0)
        self.exclude_machines = {int(m) for m in data.get("exclude_machines") or ()}
        for row in data.get("boxes") or []:
            try:
                box = BoxState.from_json(row)
            except (KeyError, TypeError, ValueError):
                continue
            self.boxes[box.instance_id] = box

    def _save_state(self) -> None:
        payload = {
            "label": LABEL,
            "balance_start": self.balance_start,
            "balance_start_ts": self.balance_start_ts,
            "rate_accrued": round(self.rate_accrued, 4),
            "exclude_machines": sorted(self.exclude_machines),
            "boxes": [b.to_json() for b in self.boxes.values()],
            "updated_ts": time.time(),
        }
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.state_path)

    # -- OUR instances -------------------------------------------------------
    def our_rows(self) -> list[dict]:
        """Raw rows for the fedgrid-r3 instances. The ONLY entry point to the API list."""
        return select_ours(self.backend.show_instances())

    def our_instances(self) -> list[Instance]:
        return [parse_instance(r) for r in self.our_rows()]

    def _destroy(self, inst: Instance, why: str) -> bool:
        """Destroy one instance -- after re-checking the label on its own raw row.

        The last gate before an irreversible action: even if a caller somehow hands
        us a foreign instance, this refuses. Belt and braces on top of
        :func:`select_ours`.
        """
        if not is_ours(inst.raw):
            say(f"REFUSING to destroy instance {inst.id}: label {inst.label!r} != {LABEL!r}")
            return False
        say(f"destroy {inst.id} ({why}); rate ${inst.dph_total:.3f}/hr")
        try:
            self.backend.destroy_instance(inst.id)
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            say(f"WARN: destroy {inst.id} failed: {exc}")
            return False
        with self._lock:
            self.boxes.pop(inst.id, None)
        return True

    def destroy_all_ours(self, why: str) -> int:
        """Destroy every fedgrid-r3 instance -- and nothing else. The kill switch."""
        killed = 0
        for inst in self.our_instances():
            if self._destroy(inst, why):
                killed += 1
        remaining = self.our_instances()
        if remaining:
            say(f"WARN: {len(remaining)} fedgrid-r3 instances still present: "
                f"{[i.id for i in remaining]}")
        else:
            say("teardown complete: no fedgrid-r3 instances remain.")
        self._save_state()
        return killed

    # -- work accounting -----------------------------------------------------
    def collected_ids(self) -> set[str]:
        """Run ids with a result JSON already on the operator's disk (the truth)."""
        return {
            p.stem
            for p in self.out_dir.glob("*.json")
            if not p.name.startswith("controller_") and not p.name.startswith("fleet_")
        }

    def pending_entries(self) -> list[dict]:
        have = self.collected_ids()
        return [e for e in self.entries if str(e["run_id"]) not in have]

    # -- money ---------------------------------------------------------------
    def account_balance(self) -> float | None:
        try:
            return self.backend.account_credit()
        except (RuntimeError, OSError, subprocess.SubprocessError, ValueError) as exc:
            say(f"WARN: could not read account balance: {exc}")
            return None

    def snapshot_balance(self) -> None:
        """Record the starting balance once; a restart reuses the persisted one."""
        if self.balance_start is not None and self.balance_start_ts is not None:
            say(f"money: resuming from persisted snapshot ${self.balance_start:.2f} "
                f"({(time.time() - self.balance_start_ts) / 3600.0:.2f} h ago)")
            return
        bal = self.account_balance()
        if bal is None:
            say("WARN: no starting balance available; spend tracking will be rate-based only")
            return
        self.balance_start = bal
        self.balance_start_ts = time.time()
        say(f"money: starting balance ${bal:.2f} (cap ${self.cfg.cap_usd:.2f}, "
            f"external franchise ${self.cfg.external_burn_per_h:.2f}/hr)")
        self._save_state()

    def spend(self) -> tuple[float, float | None]:
        """``(our_spend_usd, balance_now)`` using the real balance minus the franchise.

        Falls back to rate x uptime when the balance is unreadable: an outage of
        ``show user`` must never be the thing that turns the spend cap off.
        """
        if self.balance_start is None or self.balance_start_ts is None:
            return self.rate_accrued, None
        bal = self.account_balance()
        if bal is None:
            return self.rate_accrued, None
        elapsed_h = max(0.0, (time.time() - self.balance_start_ts) / 3600.0)
        return (
            spend_from_balance(
                self.balance_start, bal, elapsed_h, self.cfg.external_burn_per_h
            ),
            bal,
        )

    # -- provisioning --------------------------------------------------------
    def search_offers(self, want: int) -> list[Offer]:
        query = build_query_r3(
            price_cap=self.cfg.price_cap, disk_gb=self.cfg.disk_gb,
            min_inet_down=MIN_INET_DOWN,
        )
        if hasattr(self.backend, "storage_gb"):
            self.backend.storage_gb = float(self.cfg.disk_gb)
        rows = self.backend.search_offers(query, order="dph_total")
        offers = rank_offers_r3(
            filter_offers_r3(
                (parse_offer(r) for r in rows),
                price_cap=self.cfg.price_cap,
                exclude_machine_ids=self.exclude_machines,
            )
        )
        if len(offers) < want:
            say(f"WARN: only {len(offers)} credible offers for {want} wanted boxes")
        return offers[:want] if want > 0 else offers

    def provision(self, n_new: int) -> list[int]:
        """Create ``n_new`` boxes with the fedgrid-r3 label (one machine each)."""
        if n_new <= 0:
            return []
        offers = self.search_offers(n_new + 3)
        created: list[int] = []
        used_machines: set[int] = set()
        for offer in offers:
            if len(created) >= n_new:
                break
            if offer.machine_id in used_machines:
                continue  # spread over hosts: one bad machine must not take two slots
            try:
                iid = self.backend.create_instance(
                    offer.id,
                    image=self.cfg.image,
                    disk_gb=self.cfg.disk_gb,
                    label=LABEL,  # EXACT -- the safety boundary
                    onstart_cmd=self._onstart_cmd(),
                    env="",
                )
            except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
                say(f"WARN: create from offer {offer.id} failed: {exc}")
                self.exclude_machines.add(offer.machine_id)
                continue
            used_machines.add(offer.machine_id)
            created.append(int(iid))
            with self._lock:
                self.boxes[int(iid)] = BoxState(int(iid), time.time())
            say(f"created instance {iid} from offer {offer.id} "
                f"({offer.gpu_name}, ${offer.dph_total:.3f}/hr, {offer.geolocation})")
        self._save_state()
        return created

    def _onstart_cmd(self) -> str:
        onstart = Path(__file__).with_name("onstart.sh")
        if onstart.exists():
            return onstart.read_text(encoding="utf-8")
        return "mkdir -p /workspace/control /workspace/status /workspace/results"

    # -- bundle / data staging ----------------------------------------------
    def build_base_bundle(self) -> Path:
        """Assemble the ~400 KB code bundle once; per-box copies add the manifest."""
        base = self.work_dir / "bundle_base"
        if base.exists():
            shutil.rmtree(base, ignore_errors=True)
        base.mkdir(parents=True, exist_ok=True)

        wheel = find_wheel()
        if wheel is None:
            raise RuntimeError(
                "no readable fedgrid wheel in dist/ or bundle_light/; build one first "
                "(python -m build --wheel)"
            )
        shutil.copy2(wheel, base / wheel.name)

        here = Path(__file__).parent
        for script in ("run_shard.sh", "onstart.sh"):
            shutil.copy2(here / script, base / script)
        (base / "_status.py").write_text(STATUS_PY_SOURCE, encoding="utf-8")

        pkg = base / "experiments"
        pkg.mkdir(exist_ok=True)
        for src in sorted((REPO_ROOT / "experiments").glob("*.py")):
            shutil.copy2(src, pkg / src.name)
        # Placeholder so validate_bundle() passes on the base; each box overwrites
        # it with its own shard manifest.
        (base / "manifest.jsonl").write_text("", encoding="utf-8")

        missing = validate_bundle(base)
        if missing:
            raise RuntimeError(f"assembled bundle is incomplete: {missing}")
        size = sum(p.stat().st_size for p in base.rglob("*") if p.is_file())
        say(f"bundle ready: {base} ({size / 1e3:.0f} KB, wheel {wheel.name} from {wheel.parent})")
        return base

    def box_bundle(self, instance_id: int, shard: Shard) -> Path:
        """A per-box copy of the base bundle carrying only that box's manifest."""
        base = self.work_dir / "bundle_base"
        with self._lock:
            # Rebuild on demand: a long battery is long enough for the work dir to
            # be cleaned or synced away underneath us, and one missing bundle would
            # otherwise strike out every box in the fleet.
            if not (base / "manifest.jsonl").exists():
                base = self.build_base_bundle()
        dest_parent = self.work_dir / "boxes" / str(instance_id)
        dest = dest_parent / "bundle"
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        dest_parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(base, dest)
        lines = [
            json.dumps(self.by_id[rid], sort_keys=True, separators=(",", ":"))
            for rid in shard.run_ids
        ]
        (dest / "manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return dest

    def fed_tar(self, fed_id: str) -> Path:
        """Local, cached ``<fed_id>.tar.gz`` of one staged federation (~9 MB).

        Built at most once even though several boxes share a federation and deploy
        concurrently: without the per-federation lock the threads race on the temp
        file and every loser fails its deploy (and burns a strike).
        """
        cache = self.work_dir / "fedtars"
        cache.mkdir(parents=True, exist_ok=True)
        tar_path = cache / f"{fed_id}.tar.gz"
        src = self.data_dir / fed_id
        if not (src / "manifest.json").exists():
            raise RuntimeError(f"federation data missing locally: {src}")
        if tar_path.exists() and tar_path.stat().st_size > 0:
            return tar_path
        with self._lock:
            lock = self._fed_locks.setdefault(fed_id, threading.Lock())
        with lock:
            if tar_path.exists() and tar_path.stat().st_size > 0:
                return tar_path  # another thread packed it while we waited
            tmp = cache / f"{fed_id}.{os.getpid()}.{threading.get_ident()}.tmp"
            with tarfile.open(tmp, "w:gz", compresslevel=1) as tf:
                tf.add(src, arcname=fed_id)
            tmp.replace(tar_path)
        say(f"packed {fed_id}: {tar_path.stat().st_size / 1e6:.1f} MB")
        return tar_path

    # -- ssh helpers ---------------------------------------------------------
    def _exec(self, instance_id: int, command: str, *, timeout: float = 120.0) -> str:
        """Run one command over ssh. Falls back for backends without ``timeout=``."""
        try:
            return self.backend.execute(instance_id, command, timeout=timeout)
        except TypeError:
            return self.backend.execute(instance_id, command)

    def _ssh_alive(self, instance_id: int) -> bool:
        """The LIGHT readiness probe: does sshd answer? Never imports torch."""
        probe = getattr(self.backend, "ssh_ready", None)
        if probe is None:
            return True
        try:
            return bool(probe(instance_id))
        except (RuntimeError, OSError, subprocess.SubprocessError):
            return False

    # -- probe ---------------------------------------------------------------
    #: NOTE: this script must never contain the literal strings "run_shard" or
    #: "run_one" outside a bracket class -- pgrep -f would otherwise match the very
    #: shell that is running the probe.
    #:
    #: ``logts`` is the newest mtime among the shard log, the worker heartbeat
    #: (``status/self.json``, touched every 20 s by run_shard.sh) and the results
    #: dir. Using the shard log ALONE would be wrong: the FL engine prints nothing
    #: per round, so a legitimately busy box is silent for a whole run (median 80 s,
    #: tail 7 min) and the 3-minute staleness rule would kill it mid-training.
    PROBE_SCRIPT = (
        "set +e; "
        "echo ts=$(date +%s); "
        "pgrep -f 'experiments[.]run_[o]ne' >/dev/null 2>&1 && echo one=1 || echo one=0; "
        "pgrep -f '[r]un_shard[.]sh' >/dev/null 2>&1 && echo shard=1 || echo shard=0; "
        f"echo feds=$(cd {REMOTE_DATA_DIR} 2>/dev/null && ls -d fed_*/ 2>/dev/null "
        "| tr -d '/' | tr '\\n' ','); "
        f"echo res=$(ls -1 {REMOTE_RESULTS_DIR}/*.json 2>/dev/null | wc -l); "
        f"echo logts=$(stat -c %Y {REMOTE_SHARD_LOG} {REMOTE_WORKSPACE}/status/self.json "
        f"{REMOTE_RESULTS_DIR} 2>/dev/null | sort -n | tail -1 || echo 0); "
        f"grep -q 'shard complete' {REMOTE_SHARD_LOG} 2>/dev/null "
        "&& echo done=1 || echo done=0; "
        f"echo last=$(tail -n 1 {REMOTE_SHARD_LOG} 2>/dev/null | tr -d '\\r' | tail -c 120); "
        "exit 0"
    )

    def probe(self, instance_id: int) -> BoxProbe:
        """One cheap ssh round trip -> :class:`BoxProbe` (never raises)."""
        try:
            out = self._exec(instance_id, self.PROBE_SCRIPT, timeout=90.0)
        except (RuntimeError, OSError, SshUnavailableError, subprocess.SubprocessError) as exc:
            return BoxProbe(ok=False, error=str(exc)[:200])
        facts: dict[str, str] = {}
        for line in out.splitlines():
            key, _, val = line.strip().partition("=")
            if key:
                facts[key] = val
        try:
            remote_ts = float(facts.get("ts") or 0.0)
            log_ts = float(facts.get("logts") or 0.0)
        except ValueError:
            remote_ts = log_ts = 0.0
        log_age = (remote_ts - log_ts) if (remote_ts > 0 and log_ts > 0) else None
        feds = tuple(f for f in (facts.get("feds") or "").split(",") if f.startswith("fed_"))
        return BoxProbe(
            ok=True,
            run_one_alive=facts.get("one") == "1",
            shard_alive=facts.get("shard") == "1",
            feds_present=feds,
            results=int(facts.get("res") or 0),
            log_age_s=log_age,
            shard_done=facts.get("done") == "1",
            last_log_line=(facts.get("last") or "")[:120],
        )

    # -- deploy --------------------------------------------------------------
    def deploy(self, instance_id: int, shard: Shard) -> bool:
        """Push code + this box's data, install, (re)start the shard. Verified.

        Returns ``True`` only when the federation directories were *seen on the
        box* and a shard/run process was *seen running* afterwards.
        """
        box = self.boxes.get(instance_id)
        if box is None:
            return False
        tag = f"[{instance_id}]"
        try:
            # (a) code bundle -- ALWAYS re-pushed, never skipped because it exists.
            local_bundle = self.box_bundle(instance_id, shard)
            with self._net_sem:
                self.backend.copy(str(local_bundle), f"{instance_id}:{REMOTE_BUNDLE_DIR}")
            self._exec(
                instance_id,
                f"mkdir -p {REMOTE_DATA_DIR} {REMOTE_RESULTS_DIR} {REMOTE_STAGE_DIR} "
                f"{REMOTE_WORKSPACE}/status {REMOTE_WORKSPACE}/control && "
                f"cp -f {REMOTE_BUNDLE_DIR}/_status.py {REMOTE_STATUS_PY}",
                timeout=120.0,
            )

            # (b) SHARDED data: only the federations this box's runs need.
            for fed in shard.fed_ids:
                if fed in box.staged_feds:
                    continue
                tar_path = self.fed_tar(fed)
                remote_tar = f"{REMOTE_STAGE_DIR}/{fed}.tar.gz"
                with self._net_sem:
                    self.backend.copy(str(tar_path), f"{instance_id}:{remote_tar}")
                    self._exec(
                        instance_id,
                        f"tar -xzf {shlex.quote(remote_tar)} -C {REMOTE_DATA_DIR} && "
                        f"rm -f {shlex.quote(remote_tar)}",
                        timeout=600.0,
                    )

            # (d) VERIFY the data is really there before calling this deployed.
            checks = " && ".join(
                f"test -f {REMOTE_DATA_DIR}/{fed}/manifest.json" for fed in shard.fed_ids
            )
            self._exec(instance_id, f"{checks} && echo DATA_OK", timeout=120.0)
            box.staged_feds.update(shard.fed_ids)

            # (e) wheel install -- --no-deps: the image already has torch/numpy and
            # the training path never touches the heavy optional deps.
            self._exec(
                instance_id,
                "for V in /venv/main /opt/conda; do [ -f \"$V/bin/activate\" ] && "
                ". \"$V/bin/activate\" && break; done; "
                f"pip install --no-deps {REMOTE_BUNDLE_DIR}/fedgrid-*.whl 2>&1 | tail -2",
                timeout=600.0,
            )

            # (f) stop the previous shard -- IN ITS OWN ssh CALL, bracketed pattern,
            # so the pattern can never match this very shell (that self-match killed
            # a whole deploy round once).
            try:
                self._exec(
                    instance_id,
                    "pkill -f '[r]un_shard[.]sh' >/dev/null 2>&1; "
                    "pkill -f 'experiments[.]run_[o]ne' >/dev/null 2>&1; "
                    f"rm -f {REMOTE_SHARD_LOG}; sleep 1; exit 0",
                    timeout=90.0,
                )
            except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
                say(f"{tag} WARN: pre-kill failed (continuing): {exc}")

            # (g) start. Separate call again; shard 0 of 1 == every line of the
            # per-box manifest.
            # An outer `sh -c '... &'` wrapper is the one start form that provably
            # returns: the bare `& disown; exit 0` variant left the ssh channel
            # open until the 90 s timeout on every box of the first r3 launch,
            # which surfaced as a blanket "deploy FAILED (TimeoutExpired)".
            self._exec(
                instance_id,
                f"sh -c 'cd {REMOTE_WORKSPACE}; setsid nohup bash "
                f"{REMOTE_BUNDLE_DIR}/run_shard.sh 0 1 > {REMOTE_SHARD_LOG} 2>&1 "
                f"< /dev/null &'; exit 0",
                timeout=90.0,
            )
            time.sleep(START_VERIFY_DELAY_S)
            # Confirm by pgrep, NOT by a captured echo: ssh loses stdout often
            # enough that "started" is not evidence of anything.
            started = self._exec(
                instance_id,
                "set +e; pgrep -f '[r]un_shard[.]sh' >/dev/null 2>&1 && echo LIVE=1 || "
                "{ pgrep -f 'experiments[.]run_[o]ne' >/dev/null 2>&1 && echo LIVE=1 || "
                "echo LIVE=0; }; exit 0",
                timeout=90.0,
            )
            if "LIVE=1" not in started:
                raise RuntimeError("shard process not visible after start")
        except (RuntimeError, OSError, SshUnavailableError, subprocess.SubprocessError) as exc:
            box.deploy_fails += 1
            say(f"{tag} deploy FAILED ({box.deploy_fails}/{DEPLOY_STRIKES_MAX}): "
                f"{str(exc)[:220]}")
            return False

        box.deploy_fails = 0
        box.assigned = shard.run_ids
        box.required_feds = shard.fed_ids
        box.deployed_ts = time.time()
        box.state = "BUSY"
        say(f"{tag} deployed {len(shard.run_ids)} runs, feds={','.join(shard.fed_ids)}")
        return True

    # -- collection ----------------------------------------------------------
    def pull_new(self, instance_id: int) -> int:
        """Pull every result JSON this box has that we do not. Returns the count.

        Serialised **per box**: the collector thread and the control cycle both pull,
        and two concurrent pulls of the same box would fight over one remote tar and
        one local temp file (a real "file in use" failure on Windows). A pull already
        in flight is simply left to finish -- it is collecting the same files.
        """
        with self._lock:
            lock = self._pull_locks.setdefault(instance_id, threading.Lock())
        if not lock.acquire(blocking=False):
            return 0
        try:
            return self._pull_new(instance_id)
        finally:
            lock.release()

    def _pull_new(self, instance_id: int) -> int:
        try:
            listing = self._exec(
                instance_id,
                f"ls -1 {REMOTE_RESULTS_DIR} 2>/dev/null || true",
                timeout=90.0,
            )
        except (RuntimeError, OSError, SshUnavailableError, subprocess.SubprocessError):
            return 0
        names = [n.strip() for n in listing.splitlines() if n.strip().endswith(".json")]
        have = self.collected_ids()
        new = [n for n in names if Path(n).stem not in have]
        if not new:
            return 0
        landed = 0
        for batch in (new[i:i + 40] for i in range(0, len(new), 40)):
            joined = " ".join(shlex.quote(n) for n in batch)
            local_tar = self.work_dir / f"pull_{instance_id}.tgz"
            try:
                with self._net_sem:
                    self._exec(
                        instance_id,
                        f"tar -czf {REMOTE_PULL_TAR} -C {REMOTE_RESULTS_DIR} {joined}",
                        timeout=300.0,
                    )
                    if local_tar.exists():
                        local_tar.unlink()
                    self.backend.copy(f"{instance_id}:{REMOTE_PULL_TAR}", str(local_tar))
                    self._exec(instance_id, f"rm -f {REMOTE_PULL_TAR}", timeout=60.0)
                landed += self._ingest_tar(local_tar)
            except (RuntimeError, OSError, SshUnavailableError,
                    subprocess.SubprocessError, tarfile.TarError) as exc:
                say(f"[{instance_id}] WARN: pull failed this cycle: {str(exc)[:160]}")
                break
            finally:
                if local_tar.exists():
                    local_tar.unlink(missing_ok=True)
        if landed:
            say(f"[{instance_id}] collected {landed} new result(s) "
                f"-> {len(self.collected_ids())}/{len(self.entries)}")
        return landed

    def _ingest_tar(self, tar_path: Path) -> int:
        """Extract ``*.json`` members into ``out_dir`` (flat, atomic, never overwrite)."""
        added = 0
        with tarfile.open(tar_path, "r:gz") as tf:
            for member in tf.getmembers():
                if not member.isfile() or not member.name.endswith(".json"):
                    continue
                name = Path(member.name).name  # flatten: no path traversal, ever
                target = self.out_dir / name
                if target.exists():
                    continue
                src = tf.extractfile(member)
                if src is None:
                    continue
                tmp = self.out_dir / (name + ".tmp")
                with tmp.open("wb") as fh:
                    shutil.copyfileobj(src, fh)
                tmp.replace(target)
                added += 1
        return added

    def harvest_all(self) -> int:
        """Pull from every box we ever deployed to (used before any teardown)."""
        with self._lock:
            ids = [b.instance_id for b in self.boxes.values() if b.deployed_ts]
        if not ids:
            return 0
        with ThreadPoolExecutor(max_workers=min(NET_CONCURRENCY, len(ids))) as pool:
            return sum(pool.map(self.pull_new, ids))

    def _collector_loop(self) -> None:
        while not self._stop.is_set():
            try:
                with self._lock:
                    ids = [b.instance_id for b in self.boxes.values() if b.deployed_ts]
                for iid in ids:
                    if self._stop.is_set():
                        break
                    self.pull_new(iid)
            except Exception as exc:  # noqa: BLE001 -- a collector crash must not end the run
                say(f"WARN: collector cycle error: {str(exc)[:200]}")
            self._stop.wait(self.cfg.collect_s)

    # -- the cycle -----------------------------------------------------------
    def cycle(self) -> bool:
        """One control cycle. Returns ``True`` when all work is collected."""
        self._cycle += 1
        now = time.time()
        # Accrue the fallback meter with LAST cycle's rate before anything else, so
        # the cap below is armed even when the balance API is down.
        self.rate_accrued += self._last_rate * max(0.0, (now - self._last_rate_ts) / 3600.0)
        self._last_rate_ts = now

        instances = self.our_instances()
        live = {i.id: i for i in instances}

        with self._lock:
            for iid in list(self.boxes):
                if iid not in live:
                    self.boxes.pop(iid)
            for iid, inst in live.items():
                # Boot patience counts from the instance's own start, so a restarted
                # controller does not hand a 40-minute-old box a fresh 13 minutes.
                self.boxes.setdefault(iid, BoxState(iid, float(inst.start_date or now)))

        # -- money gate, before anything is created ---------------------------
        spent, balance = self.spend()
        if spent >= self.cfg.cap_usd:
            say(f"SPEND CAP HIT: ${spent:.2f} >= ${self.cfg.cap_usd:.2f} -- harvesting, "
                f"then destroying every {LABEL} instance and exiting.")
            # A few extra seconds of billing is nothing against losing finished runs.
            self.harvest_all()
            self.destroy_all_ours("spend cap")
            raise SystemExit(2)

        pending = self.pending_entries()
        if not pending:
            return True

        # -- reap the obviously broken ---------------------------------------
        for inst in instances:
            box = self.boxes[inst.id]
            age = now - box.first_seen
            if inst.is_offline:
                self._destroy(inst, f"status={inst.actual_status}")
                continue
            if not inst.is_running and age > BOOT_HARD_TIMEOUT_S:
                self.exclude_machines.add(inst.machine_id)
                self._destroy(inst, f"never booted in {age / 60:.0f} min")

        instances = [i for i in self.our_instances() if not i.is_offline]
        running = [i for i in instances if i.is_running]

        # -- probe every running box (a box in its boot grace is probed but never
        #    struck: sshd routinely lands minutes after "running") --------------
        states: dict[int, str] = {}
        if running:
            with ThreadPoolExecutor(max_workers=min(8, len(running))) as pool:
                results = list(pool.map(lambda i: (i.id, self.probe(i.id)), running))
            for iid, pr in results:
                box = self.boxes.get(iid)
                if box is None:
                    continue
                box.last_probe = pr
                in_grace = (now - box.first_seen) < BOOT_GRACE_S
                if pr.ok:
                    box.ssh_fails = 0
                    box.staged_feds.update(pr.feds_present)
                elif not in_grace:
                    box.ssh_fails += 1
                state = classify_box(
                    pr,
                    required_feds=box.required_feds,
                    ssh_fails=box.ssh_fails,
                    deploy_fails=box.deploy_fails,
                )
                if state == "UNKNOWN" and in_grace:
                    state = "BOOTING"
                box.state = state
                states[iid] = state

            # Pull from every box that looks finished BEFORE recomputing what is
            # pending -- otherwise its own results still count as missing and it
            # gets re-deployed with work it has already done.
            harvest = [
                iid for iid, st in states.items()
                if st == "IDLE" and (self.boxes[iid].last_probe.results if
                                     self.boxes[iid].last_probe else 0) > 0
            ]
            if harvest:
                with ThreadPoolExecutor(max_workers=min(NET_CONCURRENCY, len(harvest))) as pool:
                    list(pool.map(self.pull_new, harvest))

        # -- DEAD boxes go away (they bill regardless of usefulness) ----------
        for inst in list(instances):
            if states.get(inst.id) != "DEAD":
                continue
            box = self.boxes.get(inst.id)
            strikes = (box.ssh_fails, box.deploy_fails) if box else ("?", "?")
            self.exclude_machines.add(inst.machine_id)
            self._destroy(inst, f"dead (ssh_fails={strikes[0]}, deploy_fails={strikes[1]})")
        instances = [i for i in self.our_instances() if not i.is_offline]
        running = [i for i in instances if i.is_running]

        # -- work stealing: unclaimed = pending minus what BUSY boxes hold ----
        pending = self.pending_entries()
        if not pending:
            return True
        busy_ids = [i.id for i in running if states.get(i.id) == "BUSY"]
        claimed: set[str] = set()
        for iid in busy_ids:
            claimed.update(self.boxes[iid].assigned)
        unclaimed = [e for e in pending if str(e["run_id"]) not in claimed]
        # Rebalance source: when everything pending is already claimed by a busy
        # box, new/idle capacity can still help by taking slices from the DEEP
        # tails of the busy queues (round-robin over reversed queues, so the runs
        # least likely to be reached soon move first). A stolen run also stays in
        # its original queue -- run collection dedups by run_id, so the only cost
        # is bounded duplicate compute at the very end of the tail, which is
        # exactly the phase where extra boxes otherwise sit worthless.
        steal_pool = unclaimed
        if not unclaimed and pending:
            by_id = {str(e["run_id"]): e for e in pending}
            tails = []
            for iid in busy_ids:
                undone = [r for r in self.boxes[iid].assigned if r in by_id]
                if undone:
                    tails.append(list(reversed(undone)))
            stealable: list[dict] = []
            while any(tails):
                for t in tails:
                    if t:
                        rid = t.pop(0)
                        entry = by_id.pop(rid, None)
                        if entry is not None:
                            stealable.append(entry)
            steal_pool = stealable
        # A box is free only when this cycle's probe answered AND classified it IDLE
        # (shard finished / no run_one / log stalled / starved of data). Boxes that
        # are UNKNOWN (mid-reboot) or still booting are left alone.
        free_ids = [i.id for i in running if states.get(i.id) == "IDLE"]

        # -- progressive pruning ----------------------------------------------
        # An idle box is only worth keeping if there is unclaimed work for it.
        # "More boxes than runs left" and "everything left is already claimed by a
        # busy box" are the same waste: it bills and does nothing.
        surplus = max(len(running) - len(pending), len(free_ids) - len(steal_pool))
        if surplus > 0 and free_ids:
            for iid in list(free_ids[:surplus]):
                inst = next((i for i in running if i.id == iid), None)
                if inst is None:
                    continue
                self.pull_new(iid)  # last chance: never destroy uncollected results
                if self._destroy(inst, "surplus (no unclaimed work)"):
                    free_ids = [x for x in free_ids if x != iid]
            instances = [i for i in self.our_instances() if not i.is_offline]
            running = [i for i in instances if i.is_running]

        # -- hand out work ----------------------------------------------------
        deployed_now = 0
        assigned_runs = 0
        if free_ids and steal_pool:
            shards = plan_shards(steal_pool, len(free_ids))
            staged = {i: tuple(self.boxes[i].staged_feds) for i in free_ids}
            plan = assign_shards_to_boxes(shards, free_ids, staged)
            if plan:
                items = list(plan.items())
                with ThreadPoolExecutor(max_workers=min(NET_CONCURRENCY * 2, len(items))) as pool:
                    outcomes = list(pool.map(lambda kv: self.deploy(kv[0], kv[1]), items))
                deployed_now = sum(1 for ok in outcomes if ok)
                assigned_runs = sum(
                    len(sh) for (_bid, sh), ok in zip(items, outcomes, strict=True) if ok
                )

        # -- keep the fleet at the right size --------------------------------
        # Never rent for work that is already on a box: the deficit is capped by
        # the runs no box currently holds, so the tail phase shrinks instead of
        # churning boxes in and out.
        unassigned = max(0, len(steal_pool) - assigned_runs)
        want = desired_box_count(self.cfg.target, len(pending))
        deficit = min(want - len(instances), unassigned)
        if deficit > 0 and spent < self.cfg.cap_usd * 0.9:
            self.provision(deficit)

        # -- one status line --------------------------------------------------
        loading = [i for i in instances if not i.is_running]
        rate = sum(i.dph_total for i in running)
        self._last_rate = rate
        n_busy = sum(1 for s in states.values() if s == "BUSY")
        n_idle = sum(1 for s in states.values() if s == "IDLE")
        n_deployed = sum(1 for b in self.boxes.values() if b.deployed_ts)
        collected = len(self.collected_ids())
        bal_txt = f" bal=${balance:.2f}" if balance is not None else ""
        say(
            f"cycle {self._cycle:>4} t+{(now - self._started) / 60:6.1f}m | "
            f"boxes total={len(self.boxes)} run={len(running)} load={len(loading)} "
            f"busy={n_busy} idle={n_idle} | deployed={n_deployed} (+{deployed_now}) | "
            f"collected {collected}/{len(self.entries)} pending={len(pending)} | "
            f"${rate:.3f}/hr | spent ${spent:.2f}/{self.cfg.cap_usd:.2f}{bal_txt}"
        )
        self._save_state()
        return False

    # -- top level ------------------------------------------------------------
    def run(self) -> int:
        """Provision -> deploy -> collect -> teardown. Returns a process exit code."""
        pending = self.pending_entries()
        say(f"round-3 controller: {len(self.entries)} runs total, {len(pending)} pending, "
            f"out={self.out_dir}, data={self.data_dir}, target={self.cfg.target}, "
            f"cap=${self.cfg.cap_usd:.2f}")
        if not pending:
            say("nothing to do: every run already has a local result.")
            self.destroy_all_ours("no work left")
            return 0

        self.build_base_bundle()
        self.snapshot_balance()

        adopted = self.our_instances()
        if adopted:
            say(f"adopted {len(adopted)} existing {LABEL} instance(s): "
                f"{[i.id for i in adopted]}")

        self._collector = threading.Thread(
            target=self._collector_loop, name="r3-collector", daemon=True
        )
        self._collector.start()

        done = False
        try:
            while self._cycle < self.cfg.max_cycles:
                done = self.cycle()
                if done:
                    break
                self._stop.wait(self.cfg.cycle_s)
        except KeyboardInterrupt:
            self._stop.set()
            say("")
            say("INTERRUPTED. Boxes are LEFT RUNNING so the work is not lost -- either")
            say("  re-run the same command to resume, or destroy the fleet with:")
            say("  python -m experiments.fleet.controller_r3 --teardown")
            return 130
        finally:
            self._stop.set()
            if self._collector is not None:
                self._collector.join(timeout=30.0)

        # Final sweep: pull anything still sitting on a box, then tear down.
        self.harvest_all()
        missing = [str(e["run_id"]) for e in self.pending_entries()]
        spent, balance = self.spend()
        self.destroy_all_ours("work complete" if not missing else "controller exiting")

        say("")
        say("=" * 72)
        say(f"round 3 summary: {len(self.entries) - len(missing)}/{len(self.entries)} "
            f"runs collected into {self.out_dir}")
        say(f"spend (balance-based, external franchise "
            f"${self.cfg.external_burn_per_h:.2f}/hr): ${spent:.2f} of ${self.cfg.cap_usd:.2f}"
            + (f" | balance now ${balance:.2f}" if balance is not None else ""))
        if missing:
            say(f"MISSING {len(missing)} run(s); first 10: {missing[:10]}")
        say("=" * 72)
        return 0 if not missing else 1

    # -- dry run --------------------------------------------------------------
    def dry_run(self) -> int:
        """Print the plan -- offers, shards, data check, cost -- and create nothing."""
        pending = self.pending_entries()
        groups = group_by_federation(pending)
        say("=" * 72)
        say(f"DRY RUN -- nothing will be created. label={LABEL}")
        say("=" * 72)
        say(f"manifests   : {[str(p) for p in self.cfg.manifests]}")
        say(f"runs        : {len(self.entries)} total, {len(pending)} pending "
            f"({len(self.entries) - len(pending)} already in {self.out_dir})")
        say(f"federations : {len(groups)} distinct")

        say("")
        say(f"-- data check (--data-dir {self.data_dir}) --")
        missing_feds: list[str] = []
        fed_mb: dict[str, float] = {}
        for fed, runs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            src = self.data_dir / fed
            ok = (src / "manifest.json").exists()
            size_mb = (
                sum(p.stat().st_size for p in src.rglob("*") if p.is_file()) / 1e6 if ok else 0.0
            )
            fed_mb[fed] = size_mb
            if not ok:
                missing_feds.append(fed)
            say(f"  {fed}  runs={len(runs):>4}  {'present' if ok else 'MISSING'}  "
                f"{size_mb:6.1f} MB")
        say(f"  staged total on disk: {sum(fed_mb.values()):.1f} MB")
        if missing_feds:
            say(f"  ERROR: {len(missing_feds)} federation(s) not staged; run "
                f"'python -m experiments.stage_data --manifest <m> --out {self.data_dir}' first")

        say("")
        say("-- code bundle (read-only preflight; nothing is written) --")
        for line in bundle_report():
            say(f"  {line}")

        say("")
        say("-- ssh key --")
        key_state = ssh_key_report(self.cfg.ssh_key, self.backend)
        for line in key_state:
            say(f"  {line}")

        say("")
        say("-- offers --")
        query = build_query_r3(price_cap=self.cfg.price_cap, disk_gb=self.cfg.disk_gb)
        say(f"  query: {query}")
        offers = self.search_offers(0)
        say(f"  credible offers after client-side filter: {len(offers)}")
        for o in offers[: min(10, len(offers))]:
            disk = float(o.raw.get("disk_space") or 0.0)
            say(f"    id={o.id:<10} {o.gpu_name:<12} ${o.dph_total:.4f}/hr  "
                f"rel={o.reliability:.3f} disk={disk:6.0f}GB down={o.inet_down_mbps:7.0f}Mbps "
                f"{o.geolocation}")
        chosen = offers[: self.cfg.target]

        say("")
        say("-- shard plan --")
        n_boxes = desired_box_count(self.cfg.target, len(pending))
        shards = plan_shards(pending, n_boxes)
        upload_mb = 0.0
        for i, sh in enumerate(shards):
            data_mb = sum(fed_mb.get(f, 0.0) for f in sh.fed_ids)
            upload_mb += data_mb
            say(f"  box {i:>2}: {len(sh.run_ids):>4} runs  feds={','.join(sh.fed_ids)}  "
                f"data~{data_mb:.1f} MB")
        say(f"  {len(shards)} shard(s) over {n_boxes} box(es); one federation per box "
            f"where possible")
        say(f"  data upload total ~{upload_mb:.0f} MB across the fleet (vs "
            f"~{sum(fed_mb.values()) * max(1, n_boxes):.0f} MB if every box carried the whole "
            f"battery); code bundle ~0.4 MB per box per deploy")

        say("")
        say("-- cost estimate --")
        if chosen:
            rate = sum(o.dph_total for o in chosen)
            per_box = len(pending) / max(1, len(chosen))
            hours = per_box * self.cfg.eta_min_per_run / 60.0
            say(f"  {len(chosen)} boxes at ${rate:.3f}/hr total "
                f"(mean ${rate / len(chosen):.3f}/box)")
            say(f"  ~{per_box:.0f} runs/box at {self.cfg.eta_min_per_run:.1f} min/run "
                f"-> ~{hours:.1f} h")
            say(f"  estimated compute cost ~${rate * hours:.2f}  (cap ${self.cfg.cap_usd:.2f})")
            say(f"  note: the owner's production instances burn ~"
                f"${self.cfg.external_burn_per_h:.2f}/hr and are EXCLUDED from our spend")
        else:
            say("  no offers matched -- cannot estimate")

        say("")
        bal = self.account_balance()
        if bal is not None:
            say(f"account balance now: ${bal:.2f}")
        ours = self.our_instances()
        say(f"existing {LABEL} instances: {[i.id for i in ours] if ours else 'none'}")
        foreign = [
            r for r in (self.backend.show_instances() or []) if not is_ours(r)
        ]
        say(f"other instances on this account (INVISIBLE to this controller): "
            f"{[(r.get('id'), r.get('label')) for r in foreign]}")
        say("=" * 72)
        return 1 if missing_feds else 0


# =========================================================================== #
# Console / ssh key / backend helpers
# =========================================================================== #

_SAY_LOCK = threading.Lock()


def say(msg: str) -> None:
    """Print one ASCII line, thread-safely (the Windows console is cp1252)."""
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}" if msg else ""
    with _SAY_LOCK:
        try:
            print(line, flush=True)
        except UnicodeEncodeError:  # pragma: no cover -- defensive
            print(line.encode("ascii", "replace").decode("ascii"), flush=True)


def read_api_key() -> str | None:
    """``VAST_API_KEY`` if set, else the key the ``vastai`` CLI already stores."""
    key = os.environ.get("VAST_API_KEY")
    if key:
        return key.strip()
    if VASTAI_KEY_FILE.exists():
        try:
            return VASTAI_KEY_FILE.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None
    return None


#: Where a built fedgrid wheel may live, best source first.
WHEEL_DIRS = ("dist", "bundle_light", "bundle_new", "bundle")


def find_wheel() -> Path | None:
    """The newest fedgrid wheel that is actually **readable** right now.

    Newest-first, but a candidate that cannot be opened is skipped: a wheel being
    rebuilt by another process is momentarily locked on Windows, and silently
    shipping a half-written wheel to 14 boxes is far worse than using yesterday's.
    """
    candidates: list[Path] = []
    for name in WHEEL_DIRS:
        candidates.extend((REPO_ROOT / name).glob("fedgrid-*.whl"))
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with path.open("rb") as fh:
                fh.read(4)
        except OSError:
            continue
        return path
    return None


def bundle_report() -> list[str]:
    """Read-only check that the ~400 KB worker bundle *could* be assembled."""
    out: list[str] = []
    wheel = find_wheel()
    if wheel is not None:
        out.append(f"wheel: {wheel.name} from {wheel.parent.name}/ "
                   f"({wheel.stat().st_size / 1e3:.0f} KB)")
    else:
        out.append("wheel: MISSING/unreadable -- build one into dist/ before running for real")
    here = Path(__file__).parent
    for script in ("run_shard.sh", "onstart.sh"):
        out.append(f"{script}: {'present' if (here / script).exists() else 'MISSING'}")
    py = sorted((REPO_ROOT / "experiments").glob("*.py"))
    size = sum(p.stat().st_size for p in py) + (wheel.stat().st_size if wheel else 0)
    out.append(f"experiments package: {len(py)} modules; bundle ~{size / 1e3:.0f} KB per box")
    return out


def ensure_ssh_key(path: Path) -> Path:
    """Make sure the dedicated fleet key exists locally; generate it if not.

    The private key never leaves this machine and is referenced by path only.
    """
    path = Path(path).expanduser()
    if path.exists() and Path(str(path) + ".pub").exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    say(f"ssh key {path} missing -- generating a new ed25519 key pair")
    cmd = [
        "ssh-keygen", "-t", "ed25519", "-N", "", "-C", LABEL, "-f", str(path),
    ]
    proc = subprocess.run(  # noqa: S603
        cmd, capture_output=True, text=True, encoding="utf-8", timeout=120,
    )
    if proc.returncode != 0 or not path.exists():
        raise RuntimeError(f"ssh-keygen failed ({proc.returncode}): {proc.stderr.strip()}")
    say(f"generated {path}")
    return path


def register_ssh_key(path: Path, backend: Any) -> bool:
    """Attach the public key to the Vast account if it is not registered yet."""
    pub_path = Path(str(path) + ".pub")
    if not pub_path.exists():
        return False
    pub = pub_path.read_text(encoding="utf-8").strip()
    body = " ".join(pub.split()[:2])  # type + material, comment is not compared
    runner = getattr(backend, "_run", None)
    if runner is None:
        return False
    try:
        existing = runner(["show", "ssh-keys"]) or []
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        say(f"WARN: could not list account ssh keys: {exc}")
        return False
    for row in existing:
        if body and body in str(row.get("public_key") or ""):
            return True
    say("registering the fleet public key on the Vast account")
    try:
        runner(["create", "ssh-key", pub], raw=False)
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        say(f"WARN: create ssh-key failed: {exc}")
        return False
    return True


def ssh_key_report(path: Path, backend: Any) -> list[str]:
    """Human-readable key status for the dry run (never creates anything)."""
    path = Path(path).expanduser()
    pub = Path(str(path) + ".pub")
    out = [f"private key {path}: {'present' if path.exists() else 'MISSING (will be generated)'}"]
    runner = getattr(backend, "_run", None)
    if runner is None:
        return out
    try:
        keys = runner(["show", "ssh-keys"]) or []
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        out.append(f"account keys: unavailable ({exc})")
        return out
    out.append(f"account has {len(keys)} registered ssh key(s)")
    if pub.exists():
        body = " ".join(pub.read_text(encoding="utf-8").split()[:2])
        known = any(body in str(k.get("public_key") or "") for k in keys)
        out.append(f"this key is {'registered' if known else 'NOT registered (will be added)'}")
    return out


def make_backend(cfg: ControllerConfig) -> CliVastBackend:
    """A :class:`CliVastBackend` wired to the repo's vastai binary and fleet key."""
    key = read_api_key()
    if not key:
        raise RuntimeError(
            "no Vast API key: set VAST_API_KEY or run 'vastai set api-key <key>' "
            f"(looked for {VASTAI_KEY_FILE})"
        )
    binary = str(cfg.vastai_bin) if Path(cfg.vastai_bin).exists() else "vastai"
    return CliVastBackend(
        api_key=key,
        binary=binary,
        ssh_key=str(cfg.ssh_key),
        ssh_ready_timeout_s=SSH_READY_BUDGET_S,
    )


# =========================================================================== #
# CLI
# =========================================================================== #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m experiments.fleet.controller_r3",
        description=f"Round-3 Vast.ai fleet controller (label {LABEL} ONLY).",
    )
    p.add_argument("--manifest", action="append", default=[], metavar="PATH",
                   help="manifest jsonl; repeat to concatenate (temporal + magnitude)")
    p.add_argument("--out-dir", default="runs_r3", help="local result store (default runs_r3)")
    p.add_argument("--data-dir", default="data_battery",
                   help="root of staged data_battery/fed_*/ (default data_battery)")
    p.add_argument("--target", type=int, default=14, help="target fleet size (default 14)")
    p.add_argument("--cap", type=float, default=15.0,
                   help="hard spend cap in USD (default 15)")
    p.add_argument("--external-burn-per-h", type=float, default=0.40,
                   help="franchise for the owner's production instances (default 0.40 $/hr)")
    p.add_argument("--price-cap", type=float, default=PRICE_CAP_HR,
                   help=f"max $/hr per box (default {PRICE_CAP_HR})")
    p.add_argument("--cycle-s", type=float, default=CYCLE_S, help="control cycle seconds")
    p.add_argument("--collect-s", type=float, default=COLLECT_S, help="collector period seconds")
    p.add_argument("--eta-min-per-run", type=float, default=2.0,
                   help="dry-run cost estimate: minutes per run (default 2.0)")
    p.add_argument("--ssh-key", default=str(DEFAULT_SSH_KEY), help="fleet ssh private key path")
    p.add_argument("--vastai-bin", default=str(DEFAULT_VASTAI_BIN), help="vastai CLI path")
    p.add_argument("--max-cycles", type=int, default=100_000, help="safety stop")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan (offers, shards, cost) and create NOTHING")
    p.add_argument("--status", action="store_true",
                   help=f"show the live {LABEL} instances and spend, then exit")
    p.add_argument("--teardown", action="store_true",
                   help=f"destroy every {LABEL} instance (and only those), then exit")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    cfg = ControllerConfig(
        manifests=tuple(Path(m) for m in args.manifest),
        out_dir=Path(args.out_dir),
        data_dir=Path(args.data_dir),
        target=int(args.target),
        cap_usd=float(args.cap),
        external_burn_per_h=float(args.external_burn_per_h),
        price_cap=float(args.price_cap),
        cycle_s=float(args.cycle_s),
        collect_s=float(args.collect_s),
        eta_min_per_run=float(args.eta_min_per_run),
        ssh_key=Path(args.ssh_key).expanduser(),
        vastai_bin=Path(args.vastai_bin),
        max_cycles=int(args.max_cycles),
        dry_run=bool(args.dry_run),
    )
    backend = make_backend(cfg)

    if args.teardown or args.status:
        ctl = ControllerR3(backend, [], cfg)
        ours = ctl.our_instances()
        say(f"{LABEL} instances: {[(i.id, i.actual_status, round(i.dph_total, 3)) for i in ours]}")
        bal = ctl.account_balance()
        if bal is not None:
            say(f"account balance: ${bal:.2f}")
        if args.teardown:
            ctl.destroy_all_ours("operator --teardown")
        return 0

    if not cfg.manifests:
        say("ERROR: at least one --manifest is required")
        return 2
    try:
        entries = load_manifest_entries(cfg.manifests)
    except (FileNotFoundError, ValueError) as exc:
        say(f"ERROR: {exc}")
        return 2
    if not entries:
        say("ERROR: manifests are empty")
        return 2

    ctl = ControllerR3(backend, entries, cfg)
    if cfg.dry_run:
        return ctl.dry_run()

    ensure_ssh_key(cfg.ssh_key)
    register_ssh_key(cfg.ssh_key, backend)
    return ctl.run()


if __name__ == "__main__":
    raise SystemExit(main())
