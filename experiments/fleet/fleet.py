"""Vast.ai fleet controller for the FedGrid phase 4 battery.

The controller is a small, resumable state machine over a pluggable
:class:`VastBackend` (the real one shells out to the ``vastai`` CLI; tests inject a
mock). Every network side effect goes through the backend, and every decision that
matters -- which offers to buy, how manifest lines map to workers, which runs are
still missing, whether spending may continue -- is a **pure function** that is unit
tested without touching the network.

Design (VAST_BUDGET §5): *static shard-by-striding + controller pull-collection +
end-of-run gap repair*. No shared queue, no credentials on community hosts.

State machine (see :meth:`Fleet.run_fleet`)::

    SEARCH -> PROVISION -> ASSIGN -> RUN(collect + gap_repair, looped) -> TEARDOWN

**Transport (rehearsal finding, 2026-07-28).** ``vastai execute`` only accepts a
restricted command whitelist (arbitrary shell -> HTTP 400 *"Invalid command
given"*) and ``vastai copy`` is a no-op unless an SSH key is registered on the
account. :class:`CliVastBackend` therefore uses the ``vastai`` CLI **only** for the
control plane (search / create / show / destroy / attach-ssh) and moves every
byte and every command over plain ``ssh``/``scp`` with a dedicated key
(``~/.ssh/vast_fedgrid``, override with ``FEDGRID_VAST_SSH_KEY``). The key lives in
``~/.ssh`` and is referenced by *path* -- it is never written into an instance, a
repo file, or an env var, and the API key stays env-only.

It is resumable: re-invoking :meth:`Fleet.run_fleet` after a crash reconciles the
live instances (found by their label tag) against the collected results in
``runs/`` and continues -- it never restarts completed work and never double-creates
instances. This module imports nothing from ``vastai``; the CLI is
run as a subprocess, so the core install never needs the optional ``fleet`` group.
"""

from __future__ import annotations

import json
import logging
import os
import posixpath
import re
import shlex
import shutil
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger("fedgrid.fleet")

# --------------------------------------------------------------------------- #
# Constants -- the fleet contract (CONTRACT amendment h, VAST_BUDGET §4/§7).
# --------------------------------------------------------------------------- #

TARGET_GPU_NAME = "RTX 3060"
#: Substring used to recognise the target GPU in a host's self-report / Vast's
#: label. Kept loose ("3060") because Vast reports "RTX 3060" while the search
#: query spells it "RTX_3060"; a deceiving host advertising a 3060 but booting a
#: different card fails this check (see :func:`classify_instance`).
GPU_MATCH_TOKEN = "3060"
#: A genuine 12 GB RTX 3060 reports ~12 GiB; anything materially below this at boot
#: is an "underdelivering" host and is destroyed + replaced.
MIN_GPU_RAM_GB = 11.0

DEFAULT_LABEL = "fedgrid-fleet"
DEFAULT_PRICE_CAP_USD_HR = 0.08
DEFAULT_MIN_RELIABILITY = 0.95
DEFAULT_MIN_CPU_CORES = 4.0
DEFAULT_MIN_INET_DOWN_MBPS = 200.0
#: $5/TB expressed as $/GB -- the egress cap from VAST_BUDGET §4 (filters the
#: $16-53/TB outliers). Applied to *both* directions to be safe.
DEFAULT_MAX_EGRESS_USD_PER_GB = 0.005
#: cu12.x image runs on driver 535..595 (VAST_BUDGET §3); cu13 would exclude ~19%
#: of the book. Filter keeps only hosts whose max-good CUDA clears our image.
DEFAULT_CUDA_MIN = 12.4
#: Vast counts the *unpacked image* against the instance disk. The verified image
#: unpacks to ~13 GB, so 16 GB left no room for the bundle + extracted dataset;
#: 32 GB costs ~$0.009/hr/instance at the usual $0.20/GB/month.
DEFAULT_DISK_GB = 32

#: HARD, non-negotiable ceiling (CONTRACT amendment h; NIGHT_LOG guardrail 1).
SPEND_CAP_USD = 30.0
#: Conservative wall-clock horizon used only for the *provisioning* pre-check
#: (worst-case 3060 row in VAST_BUDGET §4 is ~6 h). The live projection during a
#: run is the monitor's job.
DEFAULT_PROVISION_HORIZON_HR = 6.0

#: Vast-maintained PyTorch image. **Must be a ``vastai/*`` image**: SSH launch mode
#: replaces the image ENTRYPOINT with Vast's setup script, which only brings up an
#: sshd if the image ships one -- ``pytorch/pytorch:*`` does not, which is why the
#: first rehearsal got "connection refused" on every port.
#: Pinned to a dated tag so all workers are byte-identical; cu128 wheels keep the
#: driver requirement at >=525, well inside the ``cuda_vers>=12.4`` host filter, and
#: torch 2.11 matches the version the battery was developed against.
DEFAULT_IMAGE = "vastai/pytorch:2.11.0-cu128-cuda-12.9-mini-py312-2026-06-15"

# Remote layout on every worker. These paths are a contract shared with
# ``onstart.sh`` / ``run_shard.sh``; change them in lock-step.
REMOTE_WORKSPACE = "/workspace"
REMOTE_DATA_DIR = "/workspace/data"
REMOTE_BUNDLE_DIR = "/workspace/bundle"
REMOTE_RESULTS_DIR = "/workspace/results"
REMOTE_STATUS_FILE = "/workspace/status/self.json"
REMOTE_CONTROL_DIR = "/workspace/control"

RUN_ONE_MODULE = "experiments.run_one"

# -- SSH transport ------------------------------------------------------------
#: Dedicated fleet key. Only the *path* is ever used; the private key never leaves
#: the operator box and is never copied into an instance.
DEFAULT_SSH_KEY = "~/.ssh/vast_fedgrid"
SSH_KEY_ENV = "FEDGRID_VAST_SSH_KEY"
DEFAULT_SSH_USER = "root"
SSH_CONNECT_TIMEOUT_S = 20
#: A freshly created instance answers on 22 only after the image is pulled and
#: Vast's setup script has started sshd -- minutes, not seconds.
SSH_READY_TIMEOUT_S = 900.0
SSH_EXEC_TIMEOUT_S = 900.0
SSH_COPY_TIMEOUT_S = 3600.0
#: A multi-minute large copy gets "connection reset by peer" often enough that one
#: attempt is not a transport (seen pushing the 120 MB bundle over a home uplink).
SSH_COPY_ATTEMPTS = 4

#: ``<instance_id>:<remote path>``; anchored on digits so a Windows drive letter
#: (``E:\\paper7\\...``) is never mistaken for an instance spec.
_REMOTE_SPEC = re.compile(r"^(\d+):(.*)$")


class SpendCapError(RuntimeError):
    """Raised when a hard-cap invariant is violated (accrued spend >= cap)."""


class SshUnavailableError(RuntimeError):
    """sshd on a worker is not answering (yet). Recoverable: retry / treat pending."""


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Offer:
    """A single rentable Vast offer, normalised from the ``bundles`` API row."""

    id: int
    dph_total: float
    gpu_name: str
    num_gpus: int
    gpu_ram_gb: float
    reliability: float
    cpu_cores_effective: float
    inet_down_mbps: float
    inet_up_cost: float  # $/GB
    inet_down_cost: float  # $/GB
    cuda_max: float
    verified: bool
    geolocation: str
    machine_id: int
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class Instance:
    """A live (or booting) rented instance, normalised from ``show instances``."""

    id: int
    label: str
    actual_status: str  # "running" | "loading" | "created" | "exited" | ...
    gpu_name: str
    num_gpus: int
    gpu_ram_gb: float
    dph_total: float
    start_date: float | None  # unix epoch seconds, or None before it boots
    machine_id: int
    cur_state: str
    status_msg: str
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_running(self) -> bool:
        return self.actual_status == "running"

    @property
    def is_offline(self) -> bool:
        return self.actual_status in {"exited", "offline", "stopped"} or self.cur_state == "stopped"

    def accrued_usd(self, now: float) -> float:
        """Best-effort accrued cost = rate x wall-clock since boot.

        Per-second billing (VAST_BUDGET §2) makes this a faithful estimate of the
        Vast-side charge for compute; storage/bandwidth are the fixed ~$1-2 line.
        """
        if not self.start_date:
            return 0.0
        return self.dph_total * max(0.0, now - self.start_date) / 3600.0


@dataclass(frozen=True)
class ShardAssignment:
    """Which manifest lines a worker owns, recomputable from (index, n_shards)."""

    instance_id: int
    shard_index: int
    n_shards: int
    run_ids: tuple[str, ...]


@dataclass
class FleetConfig:
    """All fleet knobs in one place (defaults = CONTRACT amendment h)."""

    target_size: int = 20
    over_provision: int = 2  # over-create +2; destroy slowest 2 after first timed run
    label: str = DEFAULT_LABEL
    image: str = DEFAULT_IMAGE
    disk_gb: int = DEFAULT_DISK_GB
    price_cap: float = DEFAULT_PRICE_CAP_USD_HR
    min_reliability: float = DEFAULT_MIN_RELIABILITY
    min_cpu_cores: float = DEFAULT_MIN_CPU_CORES
    min_inet_down: float = DEFAULT_MIN_INET_DOWN_MBPS
    max_egress_usd_per_gb: float = DEFAULT_MAX_EGRESS_USD_PER_GB
    cuda_min: float = DEFAULT_CUDA_MIN
    gpu_query_name: str = "RTX_3060"
    gpu_match_token: str = GPU_MATCH_TOKEN
    min_gpu_ram_gb: float = MIN_GPU_RAM_GB
    spend_cap_usd: float = SPEND_CAP_USD
    provision_horizon_hr: float = DEFAULT_PROVISION_HORIZON_HR
    poll_interval_s: float = 10.0
    #: Measured on 2026-07-28: a cold host needs 15-25 min just to pull the ~6 GB
    #: image before Vast's setup script even runs, so 10 min was far too short.
    boot_timeout_s: float = 1800.0
    #: Machines to never rent again this session -- e.g. one that failed to pull the
    #: image inside ``boot_timeout_s``, or a host destroyed for under-delivering.
    #: Without it the "cheapest first" ranking re-picks the same bad box forever.
    exclude_machine_ids: tuple[int, ...] = ()
    data_url: str | None = None
    data_sha256: str | None = None
    dry_run: bool = False


# --------------------------------------------------------------------------- #
# Backend abstraction
# --------------------------------------------------------------------------- #


@runtime_checkable
class VastBackend(Protocol):
    """Everything the controller needs from Vast; the only network surface."""

    def search_offers(self, query: str, *, order: str = "dph_total") -> list[dict]: ...

    def create_instance(
        self,
        offer_id: int,
        *,
        image: str,
        disk_gb: int,
        label: str,
        onstart_cmd: str,
        env: str,
    ) -> int: ...

    def show_instances(self, *, label: str | None = None) -> list[dict]: ...

    def destroy_instance(self, instance_id: int) -> None: ...

    def copy(self, src: str, dst: str) -> None: ...

    def execute(self, instance_id: int, command: str) -> str: ...

    def account_credit(self) -> float | None: ...

    # Optional (duck-typed, deliberately *not* part of the required surface so a
    # test double stays valid): ``ssh_ready(instance_id) -> bool``. When a backend
    # provides it, :meth:`Fleet._poll_until_ready` also waits for sshd.


class CliVastBackend:
    """:class:`VastBackend` backed by the ``vastai`` CLI + plain ``ssh``/``scp``.

    Control plane (``vastai`` CLI): search, create, show, destroy, attach-ssh. The
    API key is read from ``VAST_API_KEY`` and passed with ``--api-key`` on every
    call. We deliberately never run ``vastai set api-key`` (which would persist the
    key to ``~/.config/vastai``) and never place the key in an instance's env.

    Data plane (``ssh``/``scp``): :meth:`execute` and :meth:`copy`. ``vastai
    execute`` rejects anything outside its whitelist ("Invalid command given") and
    ``vastai copy`` silently no-ops without a registered account key, so neither can
    carry the workload. Both methods here return real output and **raise on
    failure** -- a push is size-verified against the source, never assumed.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        binary: str | None = None,
        ssh_key: str | None = None,
        ssh_user: str = DEFAULT_SSH_USER,
        ssh_ready_timeout_s: float = SSH_READY_TIMEOUT_S,
    ) -> None:
        self._api_key = api_key or os.environ.get("VAST_API_KEY")
        self._bin = binary or os.environ.get("VASTAI_BIN", "vastai")
        raw_key = ssh_key or os.environ.get(SSH_KEY_ENV) or DEFAULT_SSH_KEY
        self._ssh_key = str(Path(raw_key).expanduser())
        self._ssh_user = ssh_user
        self._ssh_ready_timeout_s = ssh_ready_timeout_s
        self._rows_cache: tuple[float, list[dict]] | None = None
        self._ssh_ok: set[int] = set()
        self._attached: set[int] = set()
        self._endpoint: dict[int, tuple[str, int]] = {}
        #: GiB of disk priced into ``search offers`` (Vast defaults to 5 GiB, which
        #: understates the real ``dph_total`` of a 32 GB worker by ~$0.008/hr).
        self.storage_gb: float = float(DEFAULT_DISK_GB)

    # -- low level ------------------------------------------------------------
    def _run(self, args: Sequence[str], *, raw: bool = True) -> Any:
        if not self._api_key:
            raise RuntimeError(
                "VAST_API_KEY is not set; the fleet backend refuses to run without a key."
            )
        if shutil.which(self._bin) is None and not Path(self._bin).exists():
            raise RuntimeError(
                f"vastai CLI '{self._bin}' not found; install with pip install 'fedgrid[fleet]'."
            )
        cmd = [self._bin, "--api-key", self._api_key, *args]
        if raw:
            cmd.append("--raw")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # noqa: S603
        if proc.returncode != 0:
            raise RuntimeError(f"vastai {' '.join(args)} failed ({proc.returncode}): {proc.stderr}")
        if not raw:
            return proc.stdout
        out = proc.stdout.strip()
        return json.loads(out) if out else None

    # -- protocol -------------------------------------------------------------
    def search_offers(self, query: str, *, order: str = "dph_total") -> list[dict]:
        out = self._run([
            "search", "offers", query, "-o", order, "--type", "on-demand",
            "--storage", str(self.storage_gb),
        ])
        return list(out or [])

    def create_instance(
        self,
        offer_id: int,
        *,
        image: str,
        disk_gb: int,
        label: str,
        onstart_cmd: str,
        env: str,
    ) -> int:
        args = [
            "create", "instance", str(offer_id),
            "--image", image, "--disk", str(disk_gb), "--label", label,
            "--onstart-cmd", onstart_cmd, "--ssh", "--direct", "--cancel-unavail",
        ]
        if env:
            args += ["--env", env]
        out = self._run(args)
        if not out or not out.get("success", True):
            raise RuntimeError(f"create instance failed for offer {offer_id}: {out}")
        iid = int(out.get("new_contract") or out.get("id"))
        # Belt and braces: the account key is inherited at create time, but an
        # explicit attach guarantees propagation to *this* contract (idempotent).
        self.attach_ssh_key(iid)
        self._rows_cache = None
        return iid

    def show_instances(self, *, label: str | None = None) -> list[dict]:
        args = ["show", "instances"]
        if label is not None:
            args += ["--label", label]
        out = self._run(args)
        return list(out or [])

    def destroy_instance(self, instance_id: int) -> None:
        self._run(["destroy", "instance", str(instance_id), "-y"])
        self._rows_cache = None
        self._ssh_ok.discard(instance_id)
        self._attached.discard(instance_id)

    def account_credit(self) -> float | None:
        out = self._run(["show", "user"])
        if not isinstance(out, Mapping):
            return None
        for key in ("credit", "balance"):
            if out.get(key) is not None:
                return float(out[key])
        return None

    # -- ssh data plane -------------------------------------------------------
    def attach_ssh_key(self, instance_id: int) -> bool:
        """Attach the fleet *public* key to an instance (idempotent, best effort)."""
        if instance_id in self._attached:
            return True
        pub = Path(self._ssh_key + ".pub")
        if not pub.exists():
            log.warning("no public key at %s; cannot attach to instance %s", pub, instance_id)
            return False
        try:
            self._run(["attach", "ssh", str(instance_id), pub.read_text("utf-8").strip()],
                      raw=False)
        except (RuntimeError, subprocess.SubprocessError) as exc:
            log.info("attach ssh on %s: %s", instance_id, exc)
            return False
        self._attached.add(instance_id)
        return True

    def instance_row(self, instance_id: int, *, refresh: bool = False) -> dict:
        """The raw ``show instances`` row for one instance (30 s cache)."""
        now = time.monotonic()
        if refresh or self._rows_cache is None or now - self._rows_cache[0] > 30.0:
            self._rows_cache = (now, self.show_instances())
        for row in self._rows_cache[1]:
            if int(row.get("id") or 0) == int(instance_id):
                return row
        raise RuntimeError(f"instance {instance_id} is not in the account's instance list")

    def ssh_target(self, instance_id: int, *, refresh: bool = False) -> tuple[str, int]:
        """The endpoint we last reached this instance on, else the preferred one."""
        known = self._endpoint.get(instance_id)
        if known is not None and not refresh:
            return known
        return ssh_endpoint(self.instance_row(instance_id, refresh=refresh))

    def _ssh_opts(self, *, connect_timeout: int = SSH_CONNECT_TIMEOUT_S) -> list[str]:
        # UserKnownHostsFile=<null>: the proxy reuses ssh5.vast.ai:<port> across
        # contracts, so a persistent known_hosts would raise a spurious
        # "REMOTE HOST IDENTIFICATION HAS CHANGED" on the next rental.
        return [
            "-i", self._ssh_key,
            "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={os.devnull}",
            "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={connect_timeout}",
            "-o", "ServerAliveInterval=20",
            "-o", "ServerAliveCountMax=6",
            "-o", "LogLevel=ERROR",
        ]

    def _probe(self, host: str, port: int) -> tuple[bool, str]:
        cmd = ["ssh", *self._ssh_opts(connect_timeout=10), "-p", str(port),
               f"{self._ssh_user}@{host}", "true"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=40)  # noqa: S603
        except subprocess.SubprocessError as exc:
            return False, str(exc)
        return proc.returncode == 0, (proc.stderr or "").strip()

    def _try_endpoints(self, instance_id: int) -> tuple[tuple[str, int] | None, str]:
        """Probe every candidate endpoint once; remember the one that answers."""
        try:
            row = self.instance_row(instance_id, refresh=True)
        except RuntimeError as exc:
            return None, str(exc)
        candidates = ssh_endpoints(row)
        if not candidates:
            return None, f"no ssh endpoint published yet (status={row.get('actual_status')!r})"
        last = ""
        for host, port in candidates:
            ok, last = self._probe(host, port)
            if ok:
                log.info("ssh ready on instance %s (%s:%s)", instance_id, host, port)
                self._endpoint[instance_id] = (host, port)
                self._ssh_ok.add(instance_id)
                return (host, port), ""
        return None, f"{candidates}: {last}"

    def ssh_ready(self, instance_id: int) -> bool:
        """One non-blocking probe round: is sshd answering with our key right now?"""
        if instance_id in self._ssh_ok:
            return True
        endpoint, err = self._try_endpoints(instance_id)
        if endpoint is not None:
            return True
        if instance_id not in self._attached:
            self.attach_ssh_key(instance_id)
        log.debug("ssh not ready on %s: %s", instance_id, err)
        return False

    def wait_for_ssh(self, instance_id: int, timeout_s: float | None = None) -> tuple[str, int]:
        """Block (with backoff) until sshd answers; raise :class:`SshUnavailableError`."""
        if instance_id in self._ssh_ok:
            return self._endpoint[instance_id]
        budget = self._ssh_ready_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + budget
        delay, last = 5.0, "no attempt made"
        while True:
            endpoint, last = self._try_endpoints(instance_id)
            if endpoint is not None:
                return endpoint
            if instance_id not in self._attached:
                self.attach_ssh_key(instance_id)
            if time.monotonic() >= deadline:
                raise SshUnavailableError(
                    f"instance {instance_id}: sshd unreachable after {budget:.0f}s -- {last}"
                )
            time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
            delay = min(delay * 1.5, 30.0)

    def execute(self, instance_id: int, command: str, *, timeout: float | None = None) -> str:
        """Run ``command`` in a **login** shell on the worker and return its stdout.

        A login shell matters: Vast images put python/torch in ``/venv/main`` and
        only ``/etc/profile.d`` puts it on ``PATH``. Raises on a non-zero exit --
        the silent-empty-string behaviour of ``vastai execute`` is what hid the
        first rehearsal's failure.
        """
        host, port = self.wait_for_ssh(instance_id)
        remote = f"bash -lc {shlex.quote(command)}"
        cmd = ["ssh", *self._ssh_opts(), "-p", str(port), f"{self._ssh_user}@{host}", remote]
        proc = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, timeout=timeout or SSH_EXEC_TIMEOUT_S,
        )
        if proc.returncode != 0:
            self._ssh_ok.discard(instance_id)  # force a re-probe next time
            raise RuntimeError(
                f"ssh execute on instance {instance_id} failed ({proc.returncode}): "
                f"{(proc.stderr or '').strip()[:2000]} | stdout={(proc.stdout or '')[:500]}"
            )
        return proc.stdout

    def copy(self, src: str, dst: str) -> None:
        """Push a local file/dir to ``<id>:<path>`` or pull ``<id>:<path>`` home."""
        src_id, src_path = split_remote(src)
        dst_id, dst_path = split_remote(dst)
        if src_id is None and dst_id is not None:
            self._push(dst_id, src_path, dst_path)
        elif src_id is not None and dst_id is None:
            self._pull(src_id, src_path, dst_path)
        elif src_id is None:
            raise RuntimeError(f"copy: neither side names an instance ({src!r} -> {dst!r})")
        else:
            raise RuntimeError(f"copy: instance-to-instance is not supported ({src!r} -> {dst!r})")

    # -- copy internals -------------------------------------------------------
    def _transfer(
        self,
        instance_id: int,
        build_args: Any,
        *,
        recursive: bool,
        attempts: int = SSH_COPY_ATTEMPTS,
    ) -> tuple[str, int]:
        """scp with retries, alternating endpoints between tries.

        A multi-minute 120 MB push over a home uplink gets reset often enough that
        one shot is not a transport. Each retry swaps direct <-> proxy, so a flaky
        NAT mapping on one of them cannot strand a worker.
        """
        self.wait_for_ssh(instance_id)
        known = self._endpoint.get(instance_id)
        candidates = ssh_endpoints(self.instance_row(instance_id))
        order = ([known] if known else []) + [c for c in candidates if c != known]
        if not order:
            raise SshUnavailableError(f"instance {instance_id} exposes no ssh endpoint")
        last = ""
        for attempt in range(attempts):
            host, port = order[attempt % len(order)]
            try:
                self._scp(build_args(host, port), recursive=recursive)
                self._endpoint[instance_id] = (host, port)
                return host, port
            except RuntimeError as exc:
                last = str(exc)
                log.warning(
                    "scp attempt %d/%d to instance %s via %s:%s failed: %s",
                    attempt + 1, attempts, instance_id, host, port, last,
                )
                if attempt + 1 < attempts:
                    time.sleep(5.0 * (attempt + 1))
        raise RuntimeError(
            f"scp to instance {instance_id} failed after {attempts} attempts: {last}"
        )

    def _scp(self, args: Sequence[str], *, recursive: bool) -> None:
        base = ["scp", *(["-r"] if recursive else []), "-p"]
        cmd = [*base, *self._ssh_opts(), *args]
        proc = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, timeout=SSH_COPY_TIMEOUT_S,
        )
        if proc.returncode != 0:
            err = (proc.stderr or "").strip()
            if "subsystem" in err.lower() or "sftp" in err.lower():
                # OpenSSH >=9 scp speaks SFTP; -O falls back to the legacy protocol.
                proc = subprocess.run(  # noqa: S603
                    [*base, "-O", *self._ssh_opts(), *args],
                    capture_output=True, text=True, timeout=SSH_COPY_TIMEOUT_S,
                )
                err = (proc.stderr or "").strip()
            if proc.returncode != 0:
                raise RuntimeError(f"scp failed ({proc.returncode}): {err[:2000]}")

    def _push(self, instance_id: int, local: str, remote: str) -> None:
        src = Path(local)
        if not src.exists():
            raise RuntimeError(f"copy: local source {local!r} does not exist")
        self.wait_for_ssh(instance_id)
        remote = remote.rstrip("/") or "/"
        # The push wipes its destination first; never let that be / or /workspace.
        if len([p for p in remote.split("/") if p]) < 2:
            raise RuntimeError(f"copy: refusing to overwrite the remote path {remote!r}")
        parent = posixpath.dirname(remote) or "/"
        staged = posixpath.join(parent, src.name)
        wipe = f"rm -rf {shlex.quote(remote)}"
        if staged != remote:
            wipe += f" {shlex.quote(staged)}"
        self.execute(instance_id, f"mkdir -p {shlex.quote(parent)} && {wipe}")

        started = time.monotonic()
        self._transfer(
            instance_id,
            lambda h, p: ["-P", str(p), str(src), f"{self._ssh_user}@{h}:{parent}/"],
            recursive=src.is_dir(),
        )
        if staged != remote:
            self.execute(instance_id, f"mv {shlex.quote(staged)} {shlex.quote(remote)}")
        elapsed = max(time.monotonic() - started, 1e-6)

        n_files, n_bytes = _local_tree_size(src)
        got_files, got_bytes = self._remote_tree_size(instance_id, remote)
        if (got_files, got_bytes) != (n_files, n_bytes):
            raise RuntimeError(
                f"copy verification FAILED for {local!r} -> {instance_id}:{remote}: "
                f"sent {n_files} files / {n_bytes} B, remote has {got_files} files / "
                f"{got_bytes} B"
            )
        log.info(
            "pushed %s -> %s:%s (%d files, %.1f MB in %.1fs = %.1f MB/s, size-verified)",
            local, instance_id, remote, n_files, n_bytes / 1e6, elapsed,
            n_bytes / 1e6 / elapsed,
        )

    def _remote_tree_size(self, instance_id: int, remote: str) -> tuple[int, int]:
        """``(file count, total bytes)`` under ``remote`` -- the push receipt."""
        q = shlex.quote(remote)
        out = self.execute(
            instance_id,
            f"if [ -d {q} ]; then find {q} -type f -exec stat -c '%s' {{}} + 2>/dev/null "
            f"| awk '{{s+=$1}} END {{print NR, s+0}}'; "
            f"elif [ -f {q} ]; then echo \"1 $(stat -c '%s' {q})\"; else echo '0 0'; fi",
        )
        parts = (out.strip().splitlines() or ["0 0"])[-1].split()
        if len(parts) != 2:
            raise RuntimeError(f"could not size {instance_id}:{remote}; got {out!r}")
        return int(parts[0]), int(parts[1])

    def _pull(self, instance_id: int, remote: str, local: str) -> None:
        target = Path(local)
        target.parent.mkdir(parents=True, exist_ok=True)
        # scp drops <basename(remote)> *inside* an existing local dir, and writes
        # the exact filename otherwise -- both are what the collector expects.
        self._transfer(
            instance_id,
            lambda h, p: ["-P", str(p), f"{self._ssh_user}@{h}:{remote}", str(target)],
            recursive=True,
            attempts=2,  # collect() runs every cycle; a miss is retried for free
        )


# --------------------------------------------------------------------------- #
# Pure parsing / selection logic (unit tested without a backend)
# --------------------------------------------------------------------------- #


def split_remote(spec: str) -> tuple[int | None, str]:
    """Split ``"1234:/workspace/x"`` into ``(1234, "/workspace/x")``.

    A local path -- including a Windows one like ``E:\\paper7\\bundle`` -- yields
    ``(None, spec)``: only a purely numeric prefix names an instance.
    """
    m = _REMOTE_SPEC.match(spec)
    return (int(m.group(1)), m.group(2)) if m else (None, spec)


def ssh_endpoints(row: Mapping[str, Any]) -> list[tuple[str, int]]:
    """Every way to reach an instance's sshd, best first.

    ``[0]`` is the **direct** mapping (``public_ipaddr`` + the host port bound to
    ``22/tcp``) when Vast has published one, then the SSH **proxy**
    (``ssh_host``/``ssh_port``). Both are returned rather than one being guessed:
    ``public_ipaddr`` is not always the host's address (during boot Vast echoes the
    *caller's* IP there), so the transport probes the candidates in order instead
    of trusting the field. Jupyter-runtype instances expose sshd one port above
    ``ssh_port`` (matching ``vastai ssh-url``).
    """
    out: list[tuple[str, int]] = []
    mapped = (row.get("ports") or {}).get("22/tcp") or []
    host = str(row.get("public_ipaddr") or "").strip()
    if mapped and host:
        port = int(mapped[0].get("HostPort") or 0)
        if port > 0:
            out.append((host, port))
    proxy, pport = row.get("ssh_host"), row.get("ssh_port")
    if proxy and pport:
        pport = int(pport)
        if "jupyter" in str(row.get("image_runtype") or ""):
            pport += 1
        out.append((str(proxy).strip(), pport))
    return out


def ssh_endpoint(row: Mapping[str, Any]) -> tuple[str, int]:
    """The preferred ``(host, port)`` from :func:`ssh_endpoints`."""
    candidates = ssh_endpoints(row)
    if not candidates:
        raise RuntimeError(
            f"instance {row.get('id')} exposes no ssh endpoint yet "
            f"(status={row.get('actual_status')!r})"
        )
    return candidates[0]


#: Everything a worker needs in the pushed bundle. ``experiments/`` is easy to
#: forget (the wheel only carries ``fedgrid``) and its absence is invisible until
#: 22 workers all die on ``No module named experiments``.
BUNDLE_REQUIRED = (
    "run_shard.sh",
    "onstart.sh",
    "manifest.jsonl",
    "experiments/run_one.py",
    "experiments/__init__.py",
)


def validate_bundle(bundle_dir: str | Path) -> list[str]:
    """Names missing from the worker bundle (empty == ready to push)."""
    root = Path(bundle_dir)
    missing = [name for name in BUNDLE_REQUIRED if not (root / name).exists()]
    if not list(root.glob("fedgrid-*.whl")):
        missing.append("fedgrid-*.whl")
    return missing


def _local_tree_size(path: Path) -> tuple[int, int]:
    """``(file count, total bytes)`` of a local file or directory tree."""
    if path.is_file():
        return 1, path.stat().st_size
    files = [p for p in path.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def _get(row: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """First present key from ``keys`` (Vast field names drift across versions)."""
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def parse_offer(row: Mapping[str, Any]) -> Offer:
    return Offer(
        id=int(_get(row, "id", "ask_contract_id", default=0)),
        dph_total=float(_get(row, "dph_total", "dph", "usd_per_gpu_hr", default=0.0)),
        gpu_name=str(_get(row, "gpu_name", default="")),
        num_gpus=int(_get(row, "num_gpus", default=1)),
        gpu_ram_gb=float(_get(row, "gpu_ram", "gpu_totalram", default=0.0)),
        reliability=float(_get(row, "reliability", "reliability2", "rel", default=0.0)),
        cpu_cores_effective=float(_get(row, "cpu_cores_effective", "cpu_cores", default=0.0)),
        inet_down_mbps=float(_get(row, "inet_down", "inet_down_mbps", default=0.0)),
        inet_up_cost=float(_get(row, "inet_up_cost", "up_usd_tb", default=0.0)),
        inet_down_cost=float(_get(row, "inet_down_cost", "down_usd_tb", default=0.0)),
        cuda_max=float(_get(row, "cuda_max_good", "cuda_max", "cuda_vers", default=0.0)),
        verified=bool(_get(row, "verified", default=False))
        or str(_get(row, "verification", default="")) == "verified",
        geolocation=str(_get(row, "geolocation", "geo", default="")),
        machine_id=int(_get(row, "machine_id", "machine", default=0)),
        raw=dict(row),
    )


def parse_instance(row: Mapping[str, Any]) -> Instance:
    start = _get(row, "start_date")
    return Instance(
        id=int(_get(row, "id", default=0)),
        label=str(_get(row, "label", default="") or ""),
        actual_status=str(_get(row, "actual_status", "cur_state", "status", default="unknown")),
        gpu_name=str(_get(row, "gpu_name", "gpu", default="")),
        num_gpus=int(_get(row, "num_gpus", default=1)),
        gpu_ram_gb=float(_get(row, "gpu_ram", "gpu_totalram", default=0.0)),
        dph_total=float(_get(row, "dph_total", "dph", default=0.0)),
        start_date=float(start) if start else None,
        machine_id=int(_get(row, "machine_id", "machine", default=0)),
        cur_state=str(_get(row, "cur_state", "intended_status", default="")),
        status_msg=str(_get(row, "status_msg", default="") or ""),
        raw=dict(row),
    )


def build_search_query(cfg: FleetConfig) -> str:
    """The ``vastai search offers`` filter string (VAST_BUDGET §5, amendment h).

    Egress is filtered client-side in :func:`filter_offers` (the query language has
    no reliable cost field name across versions), so it is not encoded here.
    """
    return (
        f"gpu_name={cfg.gpu_query_name} num_gpus=1 rentable=true verified=true "
        f"reliability>={cfg.min_reliability} cpu_cores_effective>={int(cfg.min_cpu_cores)} "
        f"inet_down>={int(cfg.min_inet_down)} cuda_vers>={cfg.cuda_min} "
        f"disk_space>={cfg.disk_gb} dph_total<={cfg.price_cap}"
    )


def filter_offers(offers: Iterable[Offer], cfg: FleetConfig) -> list[Offer]:
    """Apply every hard filter (some also live in the query; enforced here too)."""
    excluded = set(cfg.exclude_machine_ids)
    keep: list[Offer] = []
    for o in offers:
        if o.machine_id in excluded:
            continue
        if cfg.gpu_match_token not in o.gpu_name.replace("_", " "):
            continue
        if o.num_gpus != 1 or not o.verified:
            continue
        if o.reliability < cfg.min_reliability:
            continue
        if o.cpu_cores_effective < cfg.min_cpu_cores:
            continue
        if o.inet_down_mbps < cfg.min_inet_down:
            continue
        if o.cuda_max and o.cuda_max < cfg.cuda_min:
            continue
        if o.dph_total > cfg.price_cap:
            continue
        if max(o.inet_up_cost, o.inet_down_cost) > cfg.max_egress_usd_per_gb:
            continue
        keep.append(o)
    return keep


def rank_offers(offers: Iterable[Offer]) -> list[Offer]:
    """Cheapest first; ties broken by higher reliability then lower egress."""
    return sorted(offers, key=lambda o: (o.dph_total, -o.reliability, o.inet_up_cost))


def classify_instance(inst: Instance, cfg: FleetConfig) -> str:
    """One of ``"ok"`` | ``"pending"`` | ``"mismatch"``.

    ``"mismatch"`` is the *deceiving host* case (CONTRACT amendment h): the box
    reports running but shows the wrong GPU model or too little VRAM -- destroy and
    replace. ``"pending"`` = still booting; ``"ok"`` = verified as advertised.
    """
    if inst.is_offline:
        return "mismatch"
    if not inst.is_running:
        return "pending"
    named = inst.gpu_name and cfg.gpu_match_token not in inst.gpu_name.replace("_", " ")
    underdelivers = inst.gpu_ram_gb and inst.gpu_ram_gb < cfg.min_gpu_ram_gb
    if named or underdelivers:
        return "mismatch"
    return "ok"


def stride_shard(run_ids: Sequence[str], index: int, n_shards: int) -> list[str]:
    """Worker ``index`` owns lines ``index, index+n, index+2n, ...`` (static stride).

    Recomputable from ``(index, n_shards)`` alone -- no shared queue state.
    """
    if not 0 <= index < n_shards:
        raise ValueError(f"shard index {index} out of range for {n_shards} shards")
    return list(run_ids[index::n_shards])


def assign_shards(run_ids: Sequence[str], n_shards: int) -> list[list[str]]:
    """Partition ``run_ids`` into ``n_shards`` striped shards (pure, total, disjoint)."""
    if n_shards <= 0:
        raise ValueError("n_shards must be positive")
    return [stride_shard(run_ids, i, n_shards) for i in range(n_shards)]


def manifest_run_ids(manifest: Iterable[Mapping[str, Any]]) -> list[str]:
    return [str(line["run_id"]) for line in manifest]


def collected_run_ids(runs_dir: str | Path) -> set[str]:
    """Run ids for which a RunResult JSON already exists (``runs/<run_id>.json``)."""
    runs = Path(runs_dir)
    if not runs.is_dir():
        return set()
    # Skip the controller's own top-level files (fleet_status.json,
    # fleet_state.json); only true RunResult files count as collected.
    return {p.stem for p in runs.glob("*.json") if not p.name.startswith("fleet_")}


def find_missing_run_ids(
    manifest: Iterable[Mapping[str, Any]], runs_dir: str | Path
) -> list[str]:
    """Manifest lines with no collected result, in manifest order (the gap set)."""
    have = collected_run_ids(runs_dir)
    return [rid for rid in manifest_run_ids(manifest) if rid not in have]


def plan_gap_repair(
    missing: Sequence[str], worker_ids: Sequence[int]
) -> dict[int, list[str]]:
    """Round-robin the missing run ids over the (already speed-sorted) workers.

    Fastest-first ordering is the caller's responsibility; this just distributes so
    every missing id lands on exactly one alive worker.
    """
    plan: dict[int, list[str]] = {wid: [] for wid in worker_ids}
    if not worker_ids:
        return plan
    for i, rid in enumerate(missing):
        plan[worker_ids[i % len(worker_ids)]].append(rid)
    return plan


def run_one_command(
    line: Mapping[str, Any], data_dir: str, out_dir: str, *, device: str = "cuda"
) -> list[str]:
    """The exact worker entrypoint invocation (Runner's interface contract).

    ``device`` defaults to ``cuda`` because fleet workers are GPU boxes (``run_one``
    itself defaults to ``cpu``).
    """
    return [
        "python", "-m", RUN_ONE_MODULE,
        "--manifest-line", json.dumps(line, sort_keys=True),
        "--data-dir", data_dir, "--out-dir", out_dir, "--device", device,
    ]


# -- money (shared with the monitor) -------------------------------------------


def total_accrued_usd(instances: Iterable[Instance], now: float) -> float:
    return sum(i.accrued_usd(now) for i in instances)


def burn_rate_usd_per_hr(instances: Iterable[Instance]) -> float:
    """Current $/hr = sum of running instances' rates."""
    return sum(i.dph_total for i in instances if i.is_running)


def project_total_spend(accrued: float, rate_per_hr: float, hours_remaining: float) -> float:
    return accrued + rate_per_hr * max(0.0, hours_remaining)


def eta_hours(runs_done: int, runs_total: int, throughput_per_hr: float) -> float | None:
    """Hours to finish from throughput; ``None`` when throughput is unknown."""
    if throughput_per_hr <= 0 or runs_done >= runs_total:
        return 0.0 if runs_done >= runs_total else None
    return (runs_total - runs_done) / throughput_per_hr


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


@dataclass
class FleetResult:
    """Outcome of a :meth:`Fleet.run_fleet` invocation."""

    completed: bool
    runs_total: int
    runs_done: int
    missing: list[str]
    accrued_usd: float
    provisioning_frozen: bool
    cycles: int


class Fleet:
    """Resumable controller over a :class:`VastBackend`.

    Parameters
    ----------
    backend:
        Anything satisfying :class:`VastBackend` (mock in tests, CLI in production).
    manifest:
        Parsed manifest lines (``run_id, arm, model, seed, tier, block, config,
        config_hash``). Load with :func:`load_manifest`.
    runs_dir:
        Local canonical result store; collected ``runs/<run_id>.json`` land here.
    bundle_dir:
        Local dir with the fedgrid wheel + experiments package + shard scripts,
        pushed to each worker. Optional in dry-run.
    """

    def __init__(
        self,
        backend: VastBackend,
        manifest: Sequence[Mapping[str, Any]],
        *,
        runs_dir: str | Path,
        bundle_dir: str | Path | None = None,
        config: FleetConfig | None = None,
        state_path: str | Path | None = None,
        clock: Any = time.time,
    ) -> None:
        self.backend = backend
        self.manifest = list(manifest)
        self.run_ids = manifest_run_ids(self.manifest)
        self.runs_dir = Path(runs_dir)
        self.bundle_dir = Path(bundle_dir) if bundle_dir else None
        self.config = config or FleetConfig()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.incoming_dir = self.runs_dir / "incoming"
        self.status_dir = self.runs_dir / "status"
        self.state_path = Path(state_path) if state_path else self.runs_dir / "fleet_state.json"
        self._clock = clock
        self.provisioning_frozen = False
        self.state: dict[str, Any] = {"phase": "SEARCH", "shard_index": {}, "n_shards": 0}
        self._load_state()

    # -- state persistence (resume) ------------------------------------------
    def _load_state(self) -> None:
        if self.state_path.exists():
            try:
                self.state.update(json.loads(self.state_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                log.warning("could not read fleet state at %s; starting fresh", self.state_path)

    def _save_state(self) -> None:
        # Re-create the store on every save: a 6 h battery is long enough for the
        # runs dir to be moved, cleaned or synced away underneath us, and losing
        # the resume state (or crashing mid-provision, with instances already
        # rented) is far worse than an extra mkdir.
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.state_path)

    # -- live view ------------------------------------------------------------
    def live_instances(self) -> list[Instance]:
        """All instances carrying our label (the only ones we ever touch)."""
        return [parse_instance(r) for r in self.backend.show_instances(label=self.config.label)]

    def _running_ok_instances(self) -> list[Instance]:
        return [
            i
            for i in self.live_instances()
            if classify_instance(i, self.config) == "ok"
        ]

    # -- 1. SEARCH ------------------------------------------------------------
    def search_offers(self, n: int | None = None) -> list[Offer]:
        """Cheapest ``n`` credible on-demand RTX 3060 offers (default target + 2)."""
        want = n if n is not None else self.config.target_size + self.config.over_provision
        # Price the offers with the disk we actually rent: Vast quotes dph_total
        # against 5 GiB by default, hiding ~$0.008/hr/worker of storage.
        if hasattr(self.backend, "storage_gb"):
            self.backend.storage_gb = float(self.config.disk_gb)
        query = build_search_query(self.config)
        rows = self.backend.search_offers(query, order="dph_total")
        offers = rank_offers(filter_offers((parse_offer(r) for r in rows), self.config))
        if len(offers) < want:
            log.warning(
                "only %d credible offers matched (wanted %d); the 3060 book may be thin "
                "on launch day -- consider relaxing target_size to 16 (amendment h).",
                len(offers), want,
            )
        return offers[:want]

    # -- money guard ----------------------------------------------------------
    def spend_state(self, instances: Sequence[Instance] | None = None) -> tuple[float, float]:
        """Return ``(accrued_usd, rate_usd_per_hr)`` for the live fleet."""
        insts = list(instances) if instances is not None else self.live_instances()
        now = self._clock()
        return total_accrued_usd(insts, now), burn_rate_usd_per_hr(insts)

    def _cap_check(self, n_new: int, instances: Sequence[Instance]) -> tuple[bool, str]:
        """Would creating ``n_new`` instances risk crossing the hard $30 cap?"""
        accrued, _ = self.spend_state(instances)
        cap = self.config.spend_cap_usd
        if accrued >= cap:
            return False, f"accrued ${accrued:.2f} >= cap ${cap:.2f}"
        projected_add = n_new * self.config.price_cap * self.config.provision_horizon_hr
        if accrued + projected_add > cap:
            return False, (
                f"accrued ${accrued:.2f} + projected ${projected_add:.2f} for {n_new} new "
                f"instances > cap ${cap:.2f}"
            )
        return True, ""

    # -- 2. PROVISION ---------------------------------------------------------
    def provision(self, n: int | None = None) -> list[Instance]:
        """Bring the fleet up to ``n`` verified workers -- idempotent, cap-guarded.

        * Never double-creates: counts existing labelled instances first and only
          creates the deficit.
        * Verifies each booted instance reports the right GPU + VRAM; a mismatching
          ("deceiving") host is destroyed and replaced from the next offer.
        * Refuses to create anything that could cross the $30 cap (sets
          :attr:`provisioning_frozen`).
        """
        target = n if n is not None else self.config.target_size
        # Cheapest possible failure: refuse to rent anything if what we would push
        # is incomplete. (A bundle without `experiments/` kills every worker.)
        if self.bundle_dir is not None and self.bundle_dir.is_dir():
            missing = validate_bundle(self.bundle_dir)
            if missing:
                raise RuntimeError(
                    f"worker bundle {self.bundle_dir} is missing {missing}; "
                    "refusing to provision (fix the bundle first)."
                )
        existing = self.live_instances()
        deficit = target - len([i for i in existing if not i.is_offline])
        if deficit <= 0:
            return existing

        allowed, reason = self._cap_check(deficit, existing)
        if not allowed:
            self.provisioning_frozen = True
            log.error("SPEND-CAP: refusing to provision %d instances -- %s", deficit, reason)
            return existing

        # Over-fetch by `over_provision` so a deceiving host can be swapped for a
        # spare without a second search (amendment h: over-create +2).
        offers = self.search_offers(deficit + self.config.over_provision)
        created: list[int] = []
        for offer in offers[:deficit]:
            if self.config.dry_run:
                log.info("[dry-run] would create instance from offer %s @ $%.4f/hr",
                         offer.id, offer.dph_total)
                continue
            iid = self.backend.create_instance(
                offer.id,
                image=self.config.image,
                disk_gb=self.config.disk_gb,
                label=self.config.label,
                onstart_cmd=self._onstart_cmd(),
                env=self._instance_env(),
            )
            created.append(iid)
        if self.config.dry_run:
            return existing

        self._poll_until_ready(created)
        self._replace_deceiving_hosts(offers[deficit:])
        self.state["phase"] = "ASSIGN"
        self._save_state()
        return self.live_instances()

    def _ssh_reachable(self, instance_id: int) -> bool:
        """Is sshd answering? ``True`` for backends without the optional probe."""
        probe = getattr(self.backend, "ssh_ready", None)
        if probe is None:
            return True
        try:
            return bool(probe(instance_id))
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            log.debug("ssh probe on %s failed: %s", instance_id, exc)
            return False

    def _poll_until_ready(self, instance_ids: Sequence[int]) -> None:
        """Block until every new instance boots *and* answers on ssh.

        "Not reachable yet" is **pending**, never a failure: the image pull plus
        Vast's setup script routinely take minutes before sshd binds. Only
        ``boot_timeout_s`` ends the wait.
        """
        deadline = self._clock() + self.config.boot_timeout_s
        pending = set(instance_ids)
        while pending and self._clock() < deadline:
            by_id = {i.id: i for i in self.live_instances()}
            for iid in list(pending):
                inst = by_id.get(iid)
                if inst is None:
                    continue
                verdict = classify_instance(inst, self.config)
                if verdict == "pending":
                    continue
                if verdict == "ok" and not self._ssh_reachable(iid):
                    continue  # booted, but sshd is still coming up
                pending.discard(iid)
            if pending:
                time.sleep(self.config.poll_interval_s)
        if pending:
            log.warning(
                "boot timeout (%.0fs): %s still not ready (booting or sshd not up)",
                self.config.boot_timeout_s, sorted(pending),
            )

    def _replace_deceiving_hosts(self, spare_offers: Sequence[Offer]) -> list[int]:
        """Destroy every mismatching host and replace from ``spare_offers``.

        Returns the ids of freshly created replacements. This is the concrete
        "Vast instances sometimes deceive" mitigation the user named.
        """
        spares = list(spare_offers)
        replacements: list[int] = []
        for inst in self.live_instances():
            if classify_instance(inst, self.config) != "mismatch":
                continue
            log.warning(
                "DECEIVING-HOST: instance %s reports gpu=%r vram=%.1fGB status=%r -- "
                "destroying and replacing.",
                inst.id, inst.gpu_name, inst.gpu_ram_gb, inst.actual_status,
            )
            self.backend.destroy_instance(inst.id)
            if spares:
                offer = spares.pop(0)
                iid = self.backend.create_instance(
                    offer.id,
                    image=self.config.image,
                    disk_gb=self.config.disk_gb,
                    label=self.config.label,
                    onstart_cmd=self._onstart_cmd(),
                    env=self._instance_env(),
                )
                replacements.append(iid)
        if replacements:
            self._poll_until_ready(replacements)
        return replacements

    def _instance_env(self) -> str:
        """docker-style ``--env`` string: dataset integrity only, never a secret."""
        parts = []
        if self.config.data_url:
            parts.append(f"-e FEDGRID_DATA_URL={self.config.data_url}")
        if self.config.data_sha256:
            parts.append(f"-e FEDGRID_DATA_SHA256={self.config.data_sha256}")
        return " ".join(parts)

    def _onstart_cmd(self) -> str:
        """The onstart script contents passed via ``--onstart-cmd`` at create time.

        Ships the real ``experiments/fleet/onstart.sh`` (GPU verify -> dataset
        sha256 verify -> idle-wait) inline, so it runs at boot without needing the
        bundle yet. Falls back to a minimal bootstrap if the file is missing.
        """
        onstart = Path(__file__).with_name("onstart.sh")
        if onstart.exists():
            return onstart.read_text(encoding="utf-8")
        return (
            "mkdir -p /workspace/control /workspace/status /workspace/results; "
            "touch ~/.no_auto_tmux"
        )

    # -- 3. ASSIGN ------------------------------------------------------------
    def assign_shards(self) -> dict[int, ShardAssignment]:
        """Statically stripe the manifest over live OK workers; push + start each.

        Shard index is stable per instance (persisted), so a replaced worker can be
        handed the same slice, and the slice is recomputable from ``(index, n)``.
        """
        workers = sorted(self._running_ok_instances(), key=lambda i: i.id)
        if not workers:
            return {}
        idx_map: dict[str, int] = dict(self.state.get("shard_index", {}))
        used = set(idx_map.values())
        next_free = (i for i in range(len(self.run_ids)) if i not in used)
        for w in workers:
            if str(w.id) not in idx_map:
                idx_map[str(w.id)] = next(next_free)
        n_shards = self.state.get("n_shards") or len(workers)
        self.state["shard_index"] = idx_map
        self.state["n_shards"] = n_shards
        self._save_state()

        assignments: dict[int, ShardAssignment] = {}
        for w in workers:
            i = idx_map[str(w.id)]
            shard = tuple(stride_shard(self.run_ids, i, n_shards))
            assignments[w.id] = ShardAssignment(w.id, i, n_shards, shard)
            if not self.config.dry_run:
                try:
                    self._push_and_start(w, i, n_shards)
                except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
                    # One unreachable box must not abort the other 21 assignments;
                    # its slice is picked up by gap_repair on a later cycle.
                    log.error("assign: could not start shard %d on %s: %s", i, w.id, exc)
        return assignments

    def _push_and_start(self, worker: Instance, shard_index: int, n_shards: int) -> None:
        """Copy the bundle + manifest, then launch the shard loop over SSH."""
        if self.bundle_dir is not None:
            self.backend.copy(str(self.bundle_dir), f"{worker.id}:{REMOTE_BUNDLE_DIR}")
        spec = json.dumps({"shard_index": shard_index, "n_shards": n_shards})
        self.backend.execute(
            worker.id,
            f"mkdir -p {REMOTE_CONTROL_DIR} && printf '%s' '{spec}' "
            f"> {REMOTE_CONTROL_DIR}/shard.json && "
            f"nohup bash {REMOTE_BUNDLE_DIR}/run_shard.sh {shard_index} {n_shards} "
            f"> {REMOTE_WORKSPACE}/shard.log 2>&1 < /dev/null & "
            f"echo started $!",
        )

    # -- 4. COLLECT -----------------------------------------------------------
    def collect(self) -> int:
        """Pull results + status from every live worker; dedupe into ``runs/``.

        Idempotent: an existing ``runs/<run_id>.json`` is never overwritten, so a
        run collected once is never lost even if the worker later dies. Returns the
        number of *newly* landed results.
        """
        self.incoming_dir.mkdir(parents=True, exist_ok=True)
        self.status_dir.mkdir(parents=True, exist_ok=True)
        new = 0
        for inst in self.live_instances():
            dst = self.incoming_dir / str(inst.id)
            dst.mkdir(parents=True, exist_ok=True)
            try:
                self.backend.copy(f"{inst.id}:{REMOTE_RESULTS_DIR}", str(dst))
                self.backend.copy(
                    f"{inst.id}:{REMOTE_STATUS_FILE}", str(self.status_dir / f"{inst.id}.json")
                )
            except RuntimeError as exc:  # a box may be mid-reboot; try again next cycle
                log.info("collect: pull from %s failed this cycle: %s", inst.id, exc)
                continue
            new += self._ingest(dst)
        return new

    def _ingest(self, incoming_dir: Path) -> int:
        """Move new ``<run_id>.json`` files into the canonical store (dedup by id)."""
        added = 0
        for src in incoming_dir.rglob("*.json"):
            if src.name == "self.json":
                continue
            target = self.runs_dir / src.name
            if target.exists():
                continue
            shutil.copy2(src, target)
            added += 1
        return added

    # -- 5. GAP REPAIR --------------------------------------------------------
    def gap_repair(self, worker_speeds: Mapping[int, float] | None = None) -> list[str]:
        """Re-dispatch every manifest line lacking a result onto the fastest workers.

        ``worker_speeds`` maps instance id -> mean run wall-time (s); lower is
        faster. Missing when unknown (falls back to id order). Returns the set of
        run ids re-dispatched this call. The RUN loop calls this until it returns
        ``[]`` -- i.e. the collected set equals the manifest (convergence).
        """
        missing = find_missing_run_ids(self.manifest, self.runs_dir)
        if not missing:
            return []
        workers = self._running_ok_instances()
        if not workers:
            log.warning("gap_repair: %d runs missing but no alive workers", len(missing))
            return []
        speeds = worker_speeds or {}
        order = sorted(workers, key=lambda w: (speeds.get(w.id, float("inf")), w.id))
        plan = plan_gap_repair(missing, [w.id for w in order])
        by_index = {w.id: self.state.get("shard_index", {}).get(str(w.id), 0) for w in order}
        n_shards = self.state.get("n_shards") or len(order)
        for wid, ids in plan.items():
            if not ids or self.config.dry_run:
                continue
            try:
                self._dispatch_repair(wid, ids, by_index.get(wid, 0), n_shards)
            except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
                log.error("gap_repair: dispatch to %s failed: %s", wid, exc)
        return missing

    def _dispatch_repair(
        self, worker_id: int, run_ids: Sequence[str], shard_index: int, n_shards: int
    ) -> None:
        by_id = {str(line["run_id"]): line for line in self.manifest}
        payload = json.dumps([by_id[r] for r in run_ids])
        spec = json.dumps({"shard_index": shard_index, "n_shards": n_shards})
        self.backend.execute(
            worker_id,
            f"mkdir -p {REMOTE_CONTROL_DIR} && "
            f"printf '%s' '{spec}' > {REMOTE_CONTROL_DIR}/shard.json && "
            f"printf '%s' '{payload}' > {REMOTE_CONTROL_DIR}/repair.jsonl && "
            f"nohup bash {REMOTE_BUNDLE_DIR}/run_shard.sh --repair "
            f"{REMOTE_CONTROL_DIR}/repair.jsonl > {REMOTE_WORKSPACE}/repair.log 2>&1 "
            f"< /dev/null & echo started $!",
        )

    # -- 6. TEARDOWN ----------------------------------------------------------
    def teardown(self) -> float:
        """DESTROY every labelled instance (never *stop* -- storage keeps billing).

        Verifies none remain and returns the final accrued spend. This is also the
        kill-switch body.
        """
        instances = self.live_instances()
        accrued, _ = self.spend_state(instances)
        now = self._clock()
        for inst in instances:
            log.info("teardown: destroying %s (accrued ~$%.3f)", inst.id, inst.accrued_usd(now))
            self.backend.destroy_instance(inst.id)
        remaining = [i for i in self.live_instances() if not i.is_offline]
        if remaining:
            log.error("teardown: %d instances still present: %s", len(remaining),
                      [i.id for i in remaining])
        else:
            log.info("teardown complete: no instances remain. Final accrued ~$%.3f", accrued)
        self.state["phase"] = "DONE"
        self._save_state()
        return accrued

    # -- top-level state machine ---------------------------------------------
    def run_fleet(
        self,
        *,
        teardown_on_complete: bool = True,
        max_cycles: int = 10_000,
        cycle_sleep_s: float = 0.0,
        worker_speeds: Mapping[int, float] | None = None,
    ) -> FleetResult:
        """Drive SEARCH->PROVISION->ASSIGN->RUN->TEARDOWN to convergence.

        Resumable: reconciles live instances + collected results on entry, so a
        re-invocation after a crash continues rather than restarting. Converges when
        every manifest line has a collected result.
        """
        missing = find_missing_run_ids(self.manifest, self.runs_dir)
        if missing:
            self.provision()
            if not self.provisioning_frozen:
                self.assign_shards()

        cycles = 0
        while cycles < max_cycles:
            cycles += 1
            self.collect()
            missing = find_missing_run_ids(self.manifest, self.runs_dir)
            if not missing:
                break
            # Keep the fleet whole, then chase the gaps.
            self.provision()
            self.gap_repair(worker_speeds=worker_speeds)
            if cycle_sleep_s:
                time.sleep(cycle_sleep_s)

        accrued, _ = self.spend_state()
        completed = not missing
        if completed and teardown_on_complete:
            accrued = self.teardown()
        return FleetResult(
            completed=completed,
            runs_total=len(self.run_ids),
            runs_done=len(self.run_ids) - len(missing),
            missing=missing,
            accrued_usd=accrued,
            provisioning_frozen=self.provisioning_frozen,
            cycles=cycles,
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def load_manifest(path: str | Path) -> list[dict]:
    """Read Runner's ``experiments/manifest.jsonl`` (one JSON object per line)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]
