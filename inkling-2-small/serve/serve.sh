#!/usr/bin/env bash
# up | down | status. Never stops ds41 or ds41-sidecar.
# `up` is the day-0 GB300 process only: no peer hook, no RTX sidecar.
# If a dsv41 container exists, `up` refuses. Removing it is the operator's
# command, after the owner's approval: ../deepseek-v4.1-flash/ds41 down.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck disable=SC1091
. "$ROOT/serve/docker.sh"
log() { printf '[inkling2] %s\n' "$*"; }
die() { printf '[inkling2] ERROR: %s\n' "$*" >&2; exit 1; }

gb300_uuid() {
  PYTHONPATH="$ROOT/peer${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -c 'from gpus import find; print(find("gb300")["uuid"])'
}

down() {
  local ink
  ink=$(container_state inkling2)
  if [ "$ink" = present ]; then
    _docker rm -f inkling2 >/dev/null
    log "removed inkling2"
  fi
  rm -f "${INKLING2_SEGMENT:-/dev/shm/inkling2-peer}"
}

up() {
  local state
  state=$(container_state ds41)
  if [ "$state" != absent ]; then
    die "ds41 or ds41-sidecar is present. This script will not stop dsv41. Bringing Inkling up needs that container removed, and that needs the owner's approval. After approval: (cd ../deepseek-v4.1-flash && ./ds41 down) && ./inkling2 up"
  fi
  state=$(container_state ds41-sidecar)
  if [ "$state" != absent ]; then
    die "ds41 or ds41-sidecar is present. This script will not stop dsv41. Bringing Inkling up needs that container removed, and that needs the owner's approval. After approval: (cd ../deepseek-v4.1-flash && ./ds41 down) && ./inkling2 up"
  fi
  PYTHONPATH="$ROOT/peer${PYTHONPATH:+:$PYTHONPATH}" python3 "$ROOT/peer/preflight.py"
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is not installed"
  [ -n "${INKLING2_MODEL_DIR:-}" ] || die "set INKLING2_MODEL_DIR"
  [ -f "$INKLING2_MODEL_DIR/config.json" ] || die "no config.json in $INKLING2_MODEL_DIR"
  local gb
  gb=$(gb300_uuid)
  [ -n "$gb" ] || die "no GB300 visible"
  down
  log "day-0 vLLM on the GB300 only (${gb}). No peer hook, no RTX sidecar, no Grace spill."
  log "UNVALIDATED on this Station's SM 10.3. First boot can spend most of the window compiling."
  _docker run -d --name inkling2 --gpus "device=${gb}" --ipc host --network host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$INKLING2_MODEL_DIR:/model:ro" \
    -e HF_HUB_OFFLINE=1 \
    -e VLLM_USE_V2_MODEL_RUNNER=1 \
    "$INKLING2_IMAGE" \
    --model /model \
    --served-model-name inkling-2-small \
    --tensor-parallel-size 1 \
    --host "${INKLING2_HOST}" --port "${INKLING2_PORT}" >/dev/null
  log "container inkling2 started. Health: curl -sf http://${INKLING2_HOST}:${INKLING2_PORT}/health"
  log "The peer sidecar is not started. See docs/SMOKE.md before enabling INKLING2_PEER."
}

status() {
  local a b ink
  a=$(container_state ds41)
  b=$(container_state ds41-sidecar)
  if [ "$a" = present ] || [ "$b" = present ]; then
    log "dsv41 containers are present. ./inkling2 up will refuse."
  else
    log "dsv41 containers are not present."
  fi
  ink=$(container_state inkling2)
  if [ "$ink" = present ]; then
    _docker ps -a --filter name=^inkling2$ --format '{{.Names}} {{.Status}}'
  else
    log "inkling2 container is not present."
  fi
}

case "${1:-}" in
  up) up ;;
  down) down ;;
  status) status ;;
  *) die "usage: serve.sh up|down|status" ;;
esac
