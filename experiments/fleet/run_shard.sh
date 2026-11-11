#!/usr/bin/env bash
# FedGrid worker shard loop -- launched by the fleet controller (Fleet.assign_shards /
# Fleet.gap_repair) via `vastai execute`, AFTER onstart.sh has verified the box.
#
# Usage:
#   run_shard.sh <shard_index> <n_shards>      # static stride: lines i, i+N, ...
#   run_shard.sh --repair <repair.jsonl>       # gap-repair: an explicit line list
#
# It installs the fedgrid wheel from the pushed bundle, then loops its manifest
# lines through the idempotent worker entrypoint
#   python -m experiments.run_one --manifest-line '<json>' \
#          --data-dir /workspace/data --out-dir /workspace/results
# updating /workspace/status/self.json (heartbeat, current run, per-run wall-times,
# runs completed) after every run so the monitor can map and time this host.
set -euo pipefail

WORKSPACE=/workspace
BUNDLE_DIR="$WORKSPACE/bundle"
DATA_DIR="$WORKSPACE/data"
RESULTS_DIR="$WORKSPACE/results"
CONTROL_DIR="$WORKSPACE/control"
STATUS_FILE="$WORKSPACE/status/self.json"
UPDATER="$WORKSPACE/_status.py"

mkdir -p "$RESULTS_DIR" "$CONTROL_DIR" "$(dirname "$STATUS_FILE")"
touch "$CONTROL_DIR/started"

# The status updater is OPTIONAL (monitor heartbeat only). NEVER block training on it: running
# onstart.sh here can hang forever on its idle-heartbeat loop, wedging run_shard before it ever
# trains (observed: boxes stuck 24min, run_one never spawned). If _status.py is absent,
# set_status is a silent no-op -- training proceeds regardless.
set_status() { [ -f "$UPDATER" ] && python3 "$UPDATER" "$STATUS_FILE" "$@" 2>/dev/null || true; }

# --- background heartbeat: keeps the box "alive" during a long run -----------
heartbeat_loop() { while true; do set_status; sleep 20; done; }
heartbeat_loop & HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null || true' EXIT

# --- python: activate the image's venv ---------------------------------------
# Vast's images keep torch in /venv/main and NOTHING activates it for a
# non-interactive shell (/etc/profile.d doesn't, and ~/.bashrc returns early), so
# a bare `python3` is /usr/bin/python3 with no torch. Verified on
# vastai/pytorch:2.11.0-cu128-...-py312: /venv/main/bin/python -> 3.12.13,
# torch 2.11.0+cu128, cuda_available=True.
for V in "${FEDGRID_VENV:-}" /venv/main /opt/conda; do
  if [ -n "$V" ] && [ -f "$V/bin/activate" ]; then
    set +u; . "$V/bin/activate"; set -u
    echo "activated python venv: $V"
    break
  fi
done
echo "python3: $(command -v python3) $(python3 -V 2>&1)"
python3 -c 'import torch;print("torch", torch.__version__, "cuda", torch.cuda.is_available())' \
  || echo "WARN: torch not importable with this interpreter" >&2

# --- environment: install the wheel, expose the experiments package ----------
export PYTHONPATH="$BUNDLE_DIR:${PYTHONPATH:-}"
if ls "$BUNDLE_DIR"/fedgrid-*.whl >/dev/null 2>&1; then
  # --no-deps: the vastai/pytorch image already ships torch+numpy, and the training path never
  # touches pandas/OpenDSS/fastapi (simulation/api imports are lazy, and data is pre-staged).
  # This avoids the 526MB torch reinstall AND the py<3.12 wheel-reject that starved earlier runs.
  pip install --no-deps "$BUNDLE_DIR"/fedgrid-*.whl 2>&1 | tail -2 || true
fi

# --- bundle-borne dataset (no-URL mode): extract once into DATA_DIR ----------
# The multi-federation root holds <fed_id>/manifest.json subdirs; presence of any
# is the integrity gate (run_one resolves each run's own federation from it).
if ! ls "$DATA_DIR"/*/manifest.json >/dev/null 2>&1 && [ -f "$BUNDLE_DIR/data.tar.gz" ]; then
  echo "Extracting bundle-borne dataset into $DATA_DIR"
  tar -xzf "$BUNDLE_DIR/data.tar.gz" -C "$DATA_DIR"
fi
if ls "$DATA_DIR"/*/manifest.json >/dev/null 2>&1 || [ -f "$DATA_DIR/manifest.json" ]; then
  set_status dataset_sha_ok=true
fi
set_status phase=running

# --- assemble this worker's list of manifest lines ---------------------------
MANIFEST="$BUNDLE_DIR/manifest.jsonl"
LINES_FILE="$(mktemp)"
if [ "${1:-}" = "--repair" ]; then
  # repair file is a JSON array of manifest lines; explode to one per line.
  python3 -c "import json,sys;[print(json.dumps(x,sort_keys=True)) for x in json.load(open(sys.argv[1]))]" \
    "$2" > "$LINES_FILE"
else
  SHARD_INDEX="${1:?shard_index required}"; N_SHARDS="${2:?n_shards required}"
  set_status shard_index="$SHARD_INDEX" n_shards="$N_SHARDS"
  awk "NR % $N_SHARDS == ($SHARD_INDEX + 1) % $N_SHARDS" "$MANIFEST" > "$LINES_FILE"
fi
TOTAL=$(wc -l < "$LINES_FILE" | tr -d ' ')
echo "shard has $TOTAL runs"

# --- run loop: idempotent, timed, self-reporting -----------------------------
DONE=0; SUM=0
while IFS= read -r LINE; do
  [ -z "$LINE" ] && continue
  RID=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['run_id'])" "$LINE")
  set_status current_run_id="$RID"
  START=$(date +%s.%N)
  # run_one SKIPs (exit 0) if runs/<run_id>.json already exists -> safe to re-run.
  # --device cuda: these are GPU workers (run_one itself defaults to cpu).
  if python3 -m experiments.run_one --manifest-line "$LINE" \
        --data-dir "$DATA_DIR" --out-dir "$RESULTS_DIR" --device cuda; then
    :
  else
    echo "WARN: run $RID exited non-zero; leaving gap for controller gap_repair." >&2
  fi
  END=$(date +%s.%N)
  WALL=$(python3 -c "print(round($END-$START,3))")
  DONE=$((DONE+1)); SUM=$(python3 -c "print($SUM+$WALL)")
  MEAN=$(python3 -c "print(round($SUM/$DONE,3))")
  set_status runs_completed="$DONE" last_run_walltime_s="$WALL" \
    mean_run_walltime_s="$MEAN" last_run_finished_ts="$END" current_run_id=null
done < "$LINES_FILE"

set_status phase=done current_run_id=null
echo "shard complete: $DONE runs."
