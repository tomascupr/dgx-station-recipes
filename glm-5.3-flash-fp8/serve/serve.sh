#!/usr/bin/env bash
# The peer tier's lifecycle: up | down | logs [glm53|sidecar]. Called by ./glm53, which sets the GLM53_* variables.
# Two containers, because the Station's driver gives a process one GPU: glm53-sidecar on the RTX PRO 6000 computes the
# cold experts; glm53 (vLLM) on the GB300 computes the rest. They exchange each MoE layer's rows through
# /dev/shm/glm53-peer (peer/channel.py). Both mount this recipe read-only at /w.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CH=/dev/shm/glm53-peer
log() { printf '[glm53] %s\n' "$*"; }
die() { printf '[glm53] ERROR: %s\n' "$*" >&2; exit 1; }

uuid_of() {
  nvidia-smi --query-gpu=name,uuid --format=csv,noheader | awk -F', ' -v n="$1" 'index($1, n) { print $2; exit }'
}

down() {
  for c in glm53 glm53-sidecar; do
    if docker inspect "$c" >/dev/null 2>&1; then docker rm -f "$c" >/dev/null; log "removed $c"; fi
  done
  rm -f "$CH"
}

wait_healthy() {
  local deadline=$((SECONDS + 3600)) c
  until curl -sf -o /dev/null "http://127.0.0.1:$GLM53_PORT/health"; do
    for c in glm53 glm53-sidecar; do
      if [ "$(docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null)" != true ]; then
        docker logs --tail 30 "$c" 2>&1 | sed 's/^/    /'
        die "$c stopped during startup (last lines above)"
      fi
    done
    if docker logs glm53 2>&1 | grep -q 'GLM53_PEER FATAL'; then die "the sidecar stopped answering: ./glm53 logs"; fi
    [ $SECONDS -lt $deadline ] || die "not healthy after 60 min: ./glm53 logs"
    sleep 10
  done
}

up() {
  local gb rtx
  gb=$(uuid_of GB300) && [ -n "$gb" ] || die "no GB300 visible"
  rtx=$(uuid_of "RTX PRO 6000") && [ -n "$rtx" ] || die "no RTX PRO 6000 visible; the model does not fit in the GB300 alone"
  down
  if [ "$(id -u)" = 0 ]; then
    sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory
  else
    log "not root: page cache not dropped before load (the checkpoint read can spill into HBM's NUMA node)"
  fi
  mkdir -p "$GLM53_HOME/cache"
  docker run -d --name glm53-sidecar --device "nvidia.com/gpu=$rtx" --ipc host \
    -v "$GLM53_MODEL_DIR:/model:ro" -v "$ROOT:/w:ro" -v "$GLM53_HOME/cache:/root/.cache" \
    -e TRITON_CACHE_DIR=/root/.cache/triton -e VLLM_TUNED_CONFIG_FOLDER=/w/peer/tuned \
    --entrypoint python3 "$GLM53_IMAGE" /w/peer/sidecar.py /model "/w/routing/$GLM53_ROWMAP" "$CH" >/dev/null
  docker run -d --name glm53 --gpus "device=$gb" --ipc host --network host \
    --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --cap-add SYS_NICE --entrypoint numactl \
    -v "$GLM53_MODEL_DIR:/model:ro" -v "$ROOT:/w:ro" -v "$GLM53_HOME/cache:/root/.cache" \
    -e PYTHONPATH=/w/peer -e VLLM_TUNED_CONFIG_FOLDER=/w/peer/tuned -e GLM53_PEER_CHANNEL="$CH" \
    -e GLM53_PEER_ROWMAP="/w/routing/$GLM53_ROWMAP" -e GLM53_PEER_MODEL=/model \
    -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang \
    -e CUDA_CACHE_PATH=/root/.cache/nv-compute -e PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024 \
    -e VLLM_ENGINE_READY_TIMEOUT_S=7200 -e VLLM_USE_RUST_FRONTEND=1 -e HF_HUB_OFFLINE=1 \
    "$GLM53_IMAGE" --membind=0 vllm serve \
    --model /model --served-model-name glm-5.3-flash zai-org/GLM-5.3-Flash \
    --tokenizer-mode glm5_next --kv-cache-dtype fp8 \
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$GLM53_MTP_TOKENS}" \
    --tool-call-parser glm47 --reasoning-parser glm47 --enable-auto-tool-choice \
    --moe-backend triton --offload-backend uva --cpu-offload-gb 300 --cpu-offload-params w13_weight w2_weight \
    --gpu-memory-utilization "$GLM53_UTIL" --kv-cache-memory-bytes "$GLM53_KV_BYTES" \
    --max-model-len "$GLM53_MAXLEN" --max-num-seqs "$GLM53_MAX_SEQS" --max-num-batched-tokens 8192 \
    --compilation-config "{\"cudagraph_capture_sizes\":$GLM53_CAPTURE_SIZES}" \
    --host "$GLM53_HOST" --port "$GLM53_PORT" >/dev/null
  log "starting: sidecar on the RTX ($rtx), vLLM on the GB300 ($gb); the first start compiles kernels, later ones take ~4 min"
  wait_healthy
  log "ready: http://$GLM53_HOST:$GLM53_PORT/v1 (model glm-5.3-flash)"
}

case "${1:-}" in
  up) up ;;
  down) down ;;
  logs) docker logs -f "${2:-glm53}" ;;
  *) die "usage: serve.sh up|down|logs [glm53|glm53-sidecar]" ;;
esac
