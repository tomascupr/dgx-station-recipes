#!/bin/sh
# Stage 4 of the GLM-5.3-Flash DGX Station recipe: stock vLLM on the GB300 alone, the number the RTX sidecar must beat.
# Built like vLLM's DGX Station recipe for DeepSeek-V4.1-Flash on the GB300 alone: TP1 on the GB300; routed-expert weights beyond
# what fits stay in pinned Grace memory (UVA, read over NVLink-C2C every step); small CUDA graphs, a fixed KV budget
# and numactl --membind=0 so pinned memory and the checkpoint's page cache stay on Grace (node 0), not in HBM (node 1).
# The same image and model flags as the B300 reference (serve-glm53-b300.sh), so the two compare.
#  - GLM53_OFFLOAD_GB (90): 306 GiB of weights against ~243 GiB at util 0.97, less ~20 GiB of runtime.
#  - GLM53_KV_BYTES (10 GiB) and --max-model-len 524288: vLLM refuses to start unless one full-length request fits.
#  - --max-num-batched-tokens 8192 and capture sizes up to 192 tokens (32 streams x 6 tokens with MTP 5): stock
#    settings took 24 GiB of CUDA graphs and 8.4 GiB of activations per GPU on the B300s (RESULTS.md).
# Run as root on the DGX, with nothing else on the GB300. Localhost only (port 30004), no API key. First start compiles and autotunes.
set -e
IMAGE=vllm/vllm-openai:nightly-a2bcb5657a881a002e79d0adab3c6b5e39436121
GB300=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | awk -F', ' '/GB300/ { print $2; exit }')
[ -n "$GB300" ] || { echo 'no GB300 visible' >&2; exit 1; }
: "${GLM53_OFFLOAD_GB:=90}" "${GLM53_KV_BYTES:=10737418240}" "${GLM53_HOME:=$HOME/.glm53}"
: "${GLM53_MODEL_DIR:=$GLM53_HOME/models/GLM-5.3-Flash}"
sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory
mkdir -p "$GLM53_HOME/cache"
docker run -d --name glm53 --gpus device=$GB300 --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --cap-add SYS_NICE --entrypoint numactl \
  -v "$GLM53_MODEL_DIR:/model:ro" -v "$GLM53_HOME/cache:/root/.cache" \
  -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang \
  -e CUDA_CACHE_PATH=/root/.cache/nv-compute -e PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=7200 -e VLLM_USE_RUST_FRONTEND=1 -e HF_HUB_OFFLINE=1 \
  "$IMAGE" --membind=0 vllm serve \
  --model /model --served-model-name glm-5.3-flash zai-org/GLM-5.3-Flash \
  --tokenizer-mode glm5_next --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --tool-call-parser glm47 --reasoning-parser glm47 --enable-auto-tool-choice \
  --offload-backend uva --cpu-offload-gb "$GLM53_OFFLOAD_GB" --cpu-offload-params w13_weight w2_weight \
  --gpu-memory-utilization 0.97 --kv-cache-memory-bytes "$GLM53_KV_BYTES" --max-model-len 524288 \
  --max-num-seqs 32 --max-num-batched-tokens 8192 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,6,8,12,16,18,24,30,36,48,60,72,96,120,144,192]}' \
  --host 127.0.0.1 --port 30004 "$@"
