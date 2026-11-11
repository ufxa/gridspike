# FedGrid Vast.ai fleet — phase 4 runbook

The exact sequence the controller follows to run the phase 4 battery (~540 runs)
across a fleet of rented RTX 3060s + the server's own 3090, with **zero data loss,
full observability, and a hard $30 cap**. Everything below is exercised in dry-run /
against a mock in `tests/test_fleet.py`; on launch day it runs for real.

Package: `experiments/fleet/` — `fleet.py` (controller), `monitor.py` (monitoring
harness), `worker.Dockerfile` + `onstart.sh` + `run_shard.sh` (the worker image),
this runbook, and `tests/test_fleet.py`.

---

## 0. What it guarantees (maps to the user's asks)

| User ask | Mechanism |
|---|---|
| **Nothing is ever lost** | `run_one` is idempotent (SKIPs when `runs/<run_id>.json` exists, atomic write). Results are **pulled** to the controller every cycle and **never overwritten**. `gap_repair` re-runs any manifest line without a collected result until the collected set == the manifest. |
| **If an instance falls it comes back** | The monitor flags `dead`/`stalled` hosts; `provision` re-creates the deficit; the fallen host's unfinished shard lines simply reappear as gaps and are re-dispatched (`gap_repair`). Striding makes shards recomputable — no shared queue to lose. |
| **Map / monitor everything** | `monitor.py` writes `runs/fleet_status.json` (full per-instance + fleet + money + progress map) and one compact line per cycle. Vast account credit is read every cycle. |
| **Detect deceiving hosts** | GPU model + VRAM are verified at boot (`nvidia-smi` self-report vs advertised); a mismatch/underdeliver is **destroyed and replaced**. A host whose per-run wall-time exceeds **2× the fleet median** is flagged `slow` for destroy-and-replace. |
| **Never overspend** | Hard **$30** cap: `provision` refuses to create past it; the monitor freezes provisioning and prints a loud `!! SPEND-CAP !!` line if accrued **or projected** crosses it. Instances are always **DESTROYED, never stopped** (stopped instances keep billing storage). |
| **Key never leaks** | The Vast API key lives only in `VAST_API_KEY`, is passed to `vastai` per call (never `vastai set api-key`, which writes it to disk), and is **never** copied onto a community host. The SSH *private* key stays in `~/.ssh/vast_fedgrid` and is referenced by path — only its **public** half is ever attached to an instance. Results flow home by controller *pull*. |

---

## 1. Environment setup (once, controller machine only)

```bash
# Install the optional fleet CLI into the venv (kept out of core deps).
./.venv/Scripts/python.exe -m pip install -e '.[fleet]'    # pulls vastai

# The key is read from the environment; it is NEVER written to a file or instance.
export VAST_API_KEY='<paste the project key>'              # console -> Keys page
vastai show user --raw --api-key "$VAST_API_KEY" | python -c \
  "import json,sys;print('credit $', json.load(sys.stdin).get('credit'))"
```

**SSH key (mandatory — the data plane runs on it).** `vastai copy` silently
no-ops and `vastai execute` rejects anything outside its whitelist, so the
controller moves bytes and commands over plain `ssh`/`scp`. Create one dedicated
key, register it on the account, and never copy the private half anywhere:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/vast_fedgrid -N '' -C fedgrid-fleet
vastai create ssh-key "$(cat ~/.ssh/vast_fedgrid.pub)" --api-key "$VAST_API_KEY"
vastai show ssh-keys --api-key "$VAST_API_KEY"             # must NOT be []
# override the path with FEDGRID_VAST_SSH_KEY if it lives elsewhere
```

**Image.** `FleetConfig.image` must be a **`vastai/*`** image. SSH launch mode
replaces the image's ENTRYPOINT with Vast's setup script, which brings up sshd only
if the image ships one — `pytorch/pytorch:*` does not, so every command and copy
against it fails (a whole paid rehearsal was lost to this). The verified default is
`vastai/pytorch:2.11.0-cu128-cuda-12.9-mini-py312-2026-06-15`.

Build + publish the worker image (public tag, **no secrets baked**):

```bash
./.venv/Scripts/python.exe -m build --wheel                # -> dist/fedgrid-*.whl
docker build -f experiments/fleet/worker.Dockerfile -t <yourrepo>/fedgrid-worker:cu124 .
docker push <yourrepo>/fedgrid-worker:cu124
```

Stage the dataset for fetch-at-onstart (keeps the image slim, ~85 MB out-of-band):

```bash
tar -czf data.tar.gz -C data .                             # the generated dataset
sha256sum data.tar.gz                                      # -> FEDGRID_DATA_SHA256
# host data.tar.gz somewhere the workers can GET (its URL -> FEDGRID_DATA_URL)
```

---

## 2. The command sequence

The controller drives everything through `Fleet` (one resumable state machine).
The canonical driver:

```python
from experiments.fleet.fleet import CliVastBackend, Fleet, FleetConfig, load_manifest

cfg = FleetConfig(
    target_size=20, over_provision=2,          # amendment h; relax if the book is thin
    # image=...  leave the default: a vastai/* image (sshd + torch). See §1.
    # disk_gb=32 by default: the image alone unpacks to >13 GB.
    data_url="https://.../data.tar.gz",        # optional; the bundle can carry the data
    data_sha256="<sha256 of data.tar.gz>",
    spend_cap_usd=30.0,                          # HARD cap
)
fleet = Fleet(CliVastBackend(), load_manifest("experiments/manifest.jsonl"),
              runs_dir="runs", bundle_dir="bundle", config=cfg)

result = fleet.run_fleet()                       # SEARCH -> PROVISION -> ASSIGN -> RUN -> TEARDOWN
print(result)                                    # completed / runs_done / accrued_usd
```

`run_fleet()` internally performs, in order:

1. **`search_offers()`** — query on-demand RTX 3060, `verified`, `reliability≥0.95`,
   `cpu_cores_effective≥4`, `inet_down≥200`, egress `≤$5/TB`, price cap `~$0.08/hr`;
   ranked cheapest-first; returns the cheapest `N` (default `target+2`, over-provisioned).
2. **`provision()`** — creates the deficit only (idempotent by the `fedgrid-fleet`
   label — never double-creates), polls to `running`, verifies each reports the
   **right GPU + VRAM**, and **destroys+replaces** any deceiving/underdelivering host.
   Refuses to create anything that could cross the $30 cap.
3. **`assign_shards()`** — static striping: worker *i* owns manifest lines `i::N`
   (recomputable, no shared state). Pushes the bundle (dataset already fetched at
   onstart; wheel + `manifest.jsonl` + `run_shard.sh`) and starts the shard loop.
4. **`collect()`** — pulls every worker's `runs/<run_id>.json` + `self.json` status
   into local `runs/` (deduped by `run_id`, existing files never overwritten). Looped.
5. **`gap_repair()`** — diffs manifest vs collected; re-dispatches the missing lines
   onto the **fastest alive** instances; loop converges when nothing is missing.
6. **`teardown()`** — **DESTROYs** every labelled instance (never stops), verifies
   none remain, prints final spend.

Run the monitor **in parallel** (separate terminal / background task) so you get one
status line per cycle and the full `fleet_status.json` map:

```bash
python -m experiments.fleet.monitor --runs-dir runs --manifest experiments/manifest.jsonl \
       --interval 45 --target-size 20 --spend-cap 30
# each cycle prints e.g.:
# [2026-07-28T…Z] alive 19/20 (load1 off0) | runs 312/540 57.8% | $4.21/$30 proj$7.30
#   cred$93.77 | ETA 1.9h thr 84/h | slow0 stall0 dead0 mism0 | med 470s
# add --no-backend to tail local dirs only (no key / no Vast calls).
```

---

## 3. Safety checks — what each one does

- **GPU verification at boot** (`onstart.sh` → `classify_instance`): the worker
  writes its real `nvidia-smi` name + VRAM to `self.json`. If it isn't a ~12 GB
  "3060", `provision`/monitor mark it `gpu_mismatch` → destroy + replace. This is
  the "deceiving host" case the user named.
- **Dataset integrity** (`onstart.sh`): the dataset is fetched from
  `FEDGRID_DATA_URL` and its **sha256 is checked against `FEDGRID_DATA_SHA256`**
  before any run; a mismatch aborts the worker (surfaces as a gap, then reprovision).
- **Slow-host / median rule** (`monitor.FleetMonitor`): per-run wall-time vs the
  fleet median; `> 2×` (with ≥3 samples) ⇒ `slow` flag ⇒ candidate for
  destroy-and-replace. Catches hosts that are "real" but throttled/oversubscribed.
- **Stall / dead detection**: no fresh result within `stall_minutes` ⇒ `stalled`;
  heartbeat older than `dead_grace_s` or an offline status ⇒ `dead` ⇒ triggers
  `gap_repair` + reprovision.
- **Hard $30 cap**: `provision` pre-checks `accrued + n_new × price_cap × horizon`
  and refuses if it would cross `spend_cap_usd`; the monitor recomputes live
  `accrued` and `projected_total` every cycle and, on breach, freezes provisioning
  and emits `!! SPEND-CAP !!`. **Never exceeded.**
- **Destroy, never stop**: `teardown` and the kill-switch always `destroy` — a
  *stopped* instance keeps billing storage (VAST_BUDGET §2).
- **Key hygiene**: create/copy/destroy all run controller-side; the key is passed
  with `--api-key` per call and never lands on a community host.

---

## 4. How resume works (crash-safe)

`run_fleet()` is **idempotent and resumable**. On (re)entry it:

1. reads the collected results in `runs/` and diffs them against the manifest;
2. finds the still-live fleet by its `fedgrid-fleet` **label** (so it never
   double-creates) and reconciles the per-instance shard assignment from
   `runs/fleet_state.json`;
3. continues from where it was — completed runs are skipped (their `runs/<id>.json`
   exists), missing runs are re-dispatched, the fleet is topped back up to target.

So a killed controller, a rebooted laptop, or a mid-battery `Ctrl-C` all recover by
simply re-running the same driver. Nothing restarts from zero.

---

## 5. Kill-switch (destroy everything, now)

```python
# Safe: destroys ONLY our labelled fleet, verifies none remain, prints final spend.
from experiments.fleet.fleet import CliVastBackend, Fleet, FleetConfig
Fleet(CliVastBackend(), [], runs_dir="runs", config=FleetConfig()).teardown()
```

Raw fallback (also label-scoped, so it never touches other tenants' instances):

```bash
for id in $(vastai show instances --label fedgrid-fleet -q --api-key "$VAST_API_KEY"); do
  vastai destroy instance "$id" -y --api-key "$VAST_API_KEY"
done
vastai show instances --label fedgrid-fleet --api-key "$VAST_API_KEY"   # must be empty
```

---

## 6. Launch-day re-check list (VAST_BUDGET §7)

Run these **before** creating the fleet — total rehearsal cost is cents:

1. **Re-snapshot the market**: re-run the RTX 3060 on-demand sweep; confirm ≥ 2×
   fleet-size credible single-GPU offers ≤ price cap, and re-check egress of the
   chosen offers. (`search_offers()` already applies every filter.)
2. **Per-run cost from the phase 3 pilot** (the §3.9 timing gate): sizes the fleet /
   picks the budget row; if ~15 min/run, escalate toward the 24-fleet contingency.
3. **Exact docker tag**: confirm the `vastai/pytorch:<torch>-cu12x-…-py312-<date>`
   tag still exists and that its torch satisfies the repo pin (`torch>=2.6`).
   **Never** a bare `pytorch/pytorch` tag — no sshd (see §1).
4. **CLI flag spellings** on the installed `vastai` (`create instance --help`) —
   onstart/env/bid flags have shifted across releases. (Pinned to 1.5.1 here.)
5. **Minimum top-up still $5** and the card-verification flow (verify at first top-up).
6. **Driver mix** of the actually-chosen offers — logged into each run's `config`.
7. **Rehearse the round trip on ONE cheap 3060 before creating the fleet**:
   `backend.execute(id, "nvidia-smi -L")` must return real text, the bundle push must
   size-verify, and `fleet.collect()` must land a parseable `RunResult`.
8. **`vastai show ssh-keys` must not be empty** and `~/.ssh/vast_fedgrid` must exist —
   without them every copy is a silent no-op.
9. **Book depth**: `search_offers()` at the default filters returned only ~12 credible
   RTX 3060 offers on 2026-07-28 (the whole on-demand 3060 book ≤$0.12/hr was 27
   rows). The binding filter is the **egress cap** (`max_egress_usd_per_gb=0.005`);
   relaxing it alone recovers ~16. Size `target_size` from a fresh count, not from 20.

---

## 7. Interface contract with the module (coordinate, don't edit)

The fleet consumes Runner's outputs by contract:

- **`experiments/manifest.jsonl`** — one JSON object per line with fields
  `run_id, arm, model, seed, tier, block, config, config_hash`. `run_id` is the key
  (`runs/<run_id>.json`); `block` drives per-block progress.
- **Worker entrypoint** — `python -m experiments.run_one --manifest-line '<json>'
  --data-dir <data> --out-dir <runs>`: idempotent (SKIP if the output exists), atomic
  write, exit 0 ok / non-zero fail. `run_shard.sh` calls exactly this.

**Deviation from the VAST_BUDGET §5 sketch** (intentional): results land in the
canonical `runs/<run_id>.json` store (deduped, never overwritten). `runs/incoming/<id>/`
is only a per-instance staging area for the pull; `runs/status/<id>.json` holds the
pulled worker self-reports the monitor reads.
