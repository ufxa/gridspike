#!/usr/bin/env bash
# FedGrid worker onstart -- runs once at boot on every rented instance.
#
# Contract (shared with experiments/fleet/fleet.py REMOTE_* constants):
#   /workspace/data          verified dataset lives here
#   /workspace/status/self.json  the self-report the controller PULLS each cycle
#   /workspace/control/       controller drops shard.json / START markers here
#   /workspace/results/       run_one writes runs/<run_id>.json here (pulled home)
#
# Responsibilities (the fleet module deliverable 1):
#   1. verify the GPU via nvidia-smi and write its model + VRAM to the status file,
#   2. fetch + verify the dataset sha256 (fetch-at-onstart keeps the image slim),
#   3. idle-wait -- heart-beating -- until the controller pushes a shard and
#      launches run_shard.sh.
#
# NO secrets are baked in or expected here: only FEDGRID_DATA_URL / _SHA256, which
# are integrity metadata, never credentials. The Vast API key never reaches a worker.
#
# NOTE: `set -e` is deliberately NOT used. Vast's SSH launch mode replaces the
# image ENTRYPOINT with its setup script and runs THIS FILE in its place, so an
# early `exit 1` here would take the whole instance down with it -- including the
# hand-off in section 4 that starts sshd. Every step below is best-effort; the
# controller verifies the box over SSH afterwards anyway.
set -uo pipefail

WORKSPACE=/workspace
STATUS_DIR="$WORKSPACE/status"
DATA_DIR="$WORKSPACE/data"
CONTROL_DIR="$WORKSPACE/control"
RESULTS_DIR="$WORKSPACE/results"
STATUS_FILE="$STATUS_DIR/self.json"
UPDATER="$WORKSPACE/_status.py"

mkdir -p "$STATUS_DIR" "$DATA_DIR" "$CONTROL_DIR" "$RESULTS_DIR"
touch "$HOME/.no_auto_tmux" 2>/dev/null || true

# --- a tiny, dependency-free JSON status merger both scripts reuse ------------
cat > "$UPDATER" <<'PY'
import json, os, sys, tempfile, time
path = sys.argv[1]
try:
    with open(path) as fh:
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
    if v == "null" or v == "":
        return None
    return v
for pair in sys.argv[2:]:
    k, _, v = pair.partition("=")
    data[k] = coerce(v)
data.setdefault("instance_id", int(os.environ.get("CONTAINER_ID", 0) or 0))
if "heartbeat_ts" not in {p.split("=", 1)[0] for p in sys.argv[2:]}:
    data["heartbeat_ts"] = time.time()
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".")
with os.fdopen(fd, "w") as fh:
    json.dump(data, fh, sort_keys=True)
os.replace(tmp, path)
PY

set_status() { python3 "$UPDATER" "$STATUS_FILE" "$@" 2>/dev/null || true; }

gpu_stats() {
  # echoes: "<name>|<total_mb>|<used_mb>|<free_mb>|<util>"; empty on no GPU.
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' | tr ',' '|' || true
}

# --- 1. GPU verification -----------------------------------------------------
STATS="$(gpu_stats)"
if [ -z "$STATS" ]; then
  set_status phase=error error=gpu_missing dataset_sha_ok=false
  echo "FATAL: no GPU visible via nvidia-smi -- controller will destroy+replace." >&2
else
  IFS='|' read -r GNAME GTOTAL GUSED GFREE GUTIL <<EOF
$STATS
EOF
  set_status phase=booting gpu_name="$GNAME" vram_total_mb="$GTOTAL" \
    vram_used_mb="$GUSED" vram_free_mb="$GFREE" gpu_util_pct="$GUTIL"
  echo "GPU: $GNAME  VRAM ${GTOTAL}MB"
fi

# --- 2. dataset fetch + sha256 verification ----------------------------------
DATASET_OK=false
if [ -n "${FEDGRID_DATA_URL:-}" ]; then
  echo "Fetching dataset: $FEDGRID_DATA_URL"
  ARCHIVE="$WORKSPACE/data.tar.gz"
  curl -fsSL "$FEDGRID_DATA_URL" -o "$ARCHIVE"
  ACTUAL="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
  if [ -n "${FEDGRID_DATA_SHA256:-}" ] && [ "$ACTUAL" != "$FEDGRID_DATA_SHA256" ]; then
    # Do NOT exit: that would kill the boot before sshd starts and leave an
    # unreachable, still-billing box. Flag it and let the controller destroy it.
    set_status phase=error error=dataset_sha_mismatch dataset_sha_ok=false
    echo "FATAL: dataset sha256 mismatch: got $ACTUAL want $FEDGRID_DATA_SHA256" >&2
    rm -f "$ARCHIVE"
  else
    tar -xzf "$ARCHIVE" -C "$DATA_DIR" --strip-components=1 2>/dev/null \
      || tar -xzf "$ARCHIVE" -C "$DATA_DIR"
    DATASET_OK=true
  fi
elif [ -f "$DATA_DIR/manifest.json" ]; then
  # Baked-in dataset path: presence of the dataset manifest is the integrity gate.
  DATASET_OK=true
fi
set_status dataset_sha_ok="$DATASET_OK"
[ "$DATASET_OK" = "true" ] && echo "Dataset ready." || echo "WARN: dataset not verified." >&2

# --- 3. idle-wait, heart-beating, until run_shard.sh takes over --------------
# The loop runs DETACHED and this script exits immediately: SSH launch mode
# replaces the image entrypoint with Vast's setup script and runs onstart as part
# of it, so an onstart that never returns can hold up the rest of the boot (sshd
# included). Nothing downstream depends on onstart still being in the foreground.
set_status phase=idle
echo "Idle. Waiting for controller to push a shard and start run_shard.sh ..."
(
  while [ ! -f "$CONTROL_DIR/started" ]; do
    STATS="$(gpu_stats)"
    if [ -n "$STATS" ]; then
      IFS='|' read -r GNAME GTOTAL GUSED GFREE GUTIL <<EOF
$STATS
EOF
      set_status vram_used_mb="$GUSED" vram_free_mb="$GFREE" gpu_util_pct="$GUTIL"
    else
      set_status
    fi
    sleep 20
  done
  echo "run_shard.sh has started; onstart idle loop exiting."
) >> "$WORKSPACE/onstart-idle.log" 2>&1 &
echo "onstart complete; heartbeat detached (pid $!)."

# --- 4. MAKE THE BOX REACHABLE -- MUST BE LAST -------------------------------
# Vast's SSH/Jupyter launch modes REPLACE the image ENTRYPOINT with their setup
# script, which runs this file *instead of* it. On vastai/* images that entrypoint
# is what starts supervisor -> sshd (Vast's own "PyTorch (Vast)" template literally
# sets onstart = `entrypoint.sh`), so without the hand-off below port 22 never
# opens and every execute/copy dies with "connection refused" -- observed on two
# paid hosts on 2026-07-28.

# 4a. Safety net: bring up an sshd ourselves first, so the controller can reach
#     (and diagnose) the box even if the image entrypoint bails out. Harmless if
#     the entrypoint later starts its own -- the second bind just fails.
if [ -x /usr/sbin/sshd ]; then
  mkdir -p /run/sshd /root/.ssh
  ssh-keygen -A >/dev/null 2>&1 || true
  if /usr/sbin/sshd -e 2>>"$WORKSPACE/onstart-sshd.log"; then
    echo "sshd started by onstart."
  else
    echo "sshd not started by onstart (already running, or refused) -- see onstart-sshd.log"
  fi
fi

# 4b. Hand off. `exec` because the image entrypoint is meant to BE the boot process.
for EP in entrypoint.sh /opt/instance-tools/bin/entrypoint.sh /entrypoint.sh; do
  if command -v "$EP" >/dev/null 2>&1 || [ -x "$EP" ]; then
    echo "handing off to the image entrypoint: $EP"
    exec "$EP"
  fi
done
echo "onstart finished (no image entrypoint to hand off to)."
