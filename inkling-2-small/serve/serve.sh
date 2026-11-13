#!/usr/bin/env bash
# up | down | status. Never stops ds41 or ds41-sidecar.
# `up` is the day-0 GB300 process only: no peer hook, no RTX sidecar.
# If a dsv41 container exists, `up` refuses. Removing it is the operator's
# command, after the owner's approval: ../deepseek-v4.1-flash/ds41 down.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
log() { printf '[inkling2] %s\n' "$*"; }
die() { printf '[inkling2] ERROR: %s\n' "$*" >&2; exit 1; }

dsv41_present() {
  command -v docker >/dev/null 2>&1 || return 1
  docker inspect ds41 >/dev/null 2>&1 && return 0
  docker inspect ds41-sidecar >/dev/null 2>&1 && return 0
  return 1
}

gb300_uuid() {
  nvidia-smi --query-gpu=name,uuid --format=csv,noheader | awk -F', ' 'index($1, "GB300") { print $2; exit }'
}

down() {
  command -v docker >/dev/null 2>&1 || { log "docker is not installed; nothing to stop"; return 0; }
  if docker inspect inkling2 >/dev/null 2>&1; then
    docker rm -f inkling2 >/dev/null
    log "removed inkling2"
  fi
  rm -f "${INKLING2_SEGMENT:-/dev/shm/inkling2-peer}"
}

up() {
  if dsv41_present; then
    die "ds41 or ds41-sidecar is present. This script will not stop dsv41. Bringing Inkling up needs that container removed, and that needs the owner's approval. After approval: (cd ../deepseek-v4.1-flash && ./ds41 down) && ./inkling2 up"
  fi
  command -v docker >/dev/null 2>&1 || die "docker is not installed"
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is not installed"
  [ -n "${INKLING2_MODEL_DIR:-}" ] || die "set INKLING2_MODEL_DIR"
  [ -f "$INKLING2_MODEL_DIR/config.json" ] || die "no config.json in $INKLING2_MODEL_DIR"
  local gb
  gb=$(gb300_uuid)
  [ -n "$gb" ] || die "no GB300 visible"
  down
  log "day-0 vLLM on the GB300 only. No peer hook, no RTX sidecar, no Grace spill."
  log "UNVALIDATED on this Station's SM 10.3. First boot can spend most of the window compiling."
  docker run -d --name inkling2 --gpus "device=${gb}" --ipc host --network host \
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
  if dsv41_present; then
    log "dsv41 containers are present. ./inkling2 up will refuse."
  else
    log "dsv41 containers are not present."
  fi
  if command -v docker >/dev/null 2>&1 && docker inspect inkling2 >/dev/null 2>&1; then
    docker ps -a --filter name=^inkling2$ --format '{{.Names}} {{.Status}}'
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
