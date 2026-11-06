#!/bin/sh
# DeepSeek-V4.1-Flash-1017 on the GB300 alone, for the root README's comparison: what the RTX PRO 6000 recipe
# (../../deepseek-v4.1-flash/, Garner McCloud's) is measured against. Not part of the GLM recipe.
# The DGX Station (GB300) row of recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash: TP1; Engram tables and 60 GiB of
# routed experts in pinned Grace memory (UVA, read over NVLink-C2C every step); FP8 KV; DSpark speculation. Changes
# from that row, kept because each made the GB300 alone faster or able to start:
#  - numactl --membind=0: pinned memory and the checkpoint's page cache stay on Grace (node 0), not in HBM (node 1).
#  - --kv-cache-memory-bytes 6 GiB (~3.1M tokens), the recipe's 0.97 budget, sized explicitly.
#  - draft schedule [[1,24,3]] instead of [[1,4,5],[5,24,1]]: drafts past the 3rd are rarely accepted here and each
#    verified token pulls more experts over C2C (single stream 93 -> 103-107 tok/s, 8 streams 251 -> 308).
# --max-num-seqs 24: at 32 streams, 8 requests wait. Image v0.32.0 (the RTX recipe uses v0.31.0).
# First start on a new image or flag set autotunes FlashInfer for 75-90 min; the cache directory keeps the result.
# Remove VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS for that first start, or those two ops serve on untuned defaults.
# Run as root on the DGX, with nothing else on the GB300. Localhost only (port 8000), no API key.
set -e
GB300=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | awk -F', ' '/GB300/ { print $2; exit }')
[ -n "$GB300" ] || { echo 'no GB300 visible' >&2; exit 1; }
: "${DS41_MODEL_DIR:?set DS41_MODEL_DIR to the DeepSeek-V4.1-Flash-1017 checkpoint}" "${DSV41_CACHE:=$HOME/.dsv41-cache}"
sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory
mkdir -p "$DSV41_CACHE/triton"
docker rm -f dsv41 2>/dev/null || true
docker run -d --name dsv41 --gpus device=$GB300 --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --cap-add SYS_NICE --entrypoint numactl \
  -v "$DS41_MODEL_DIR:/model:ro" -v "$DSV41_CACHE:/root/.cache" -v "$DSV41_CACHE/triton:/root/.triton" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=7200 -e VLLM_USE_RUST_FRONTEND=1 -e HF_HUB_OFFLINE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024 \
  -e VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS=mxfp8_gemm,flashinfer::trtllm_fp4_block_scale_moe \
  vllm/vllm-openai:v0.32.0 --membind=0 vllm serve \
  --model /model --served-model-name deepseek-v4.1-flash \
  --tokenizer-mode deepseek_v41 --tensor-parallel-size 1 \
  --attention-config '{"backend":"FLASHMLA_MEGA_ATTN_DSV41"}' --kv-cache-dtype fp8_ds_mla --max-num-seqs 24 \
  --offload-backend uva --cpu-offload-gb 60 --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight \
  --engram-config '{"cpu_offload":true}' --gpu-memory-utilization 0.97 --kv-cache-memory-bytes 6442450944 \
  --max-model-len 1048576 --max-num-batched-tokens 8192 --long-prefill-token-threshold 6144 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,6,8,12,16,18,24,32,40,48,64,96,128]}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,24,3]]}' \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41 \
  --host 127.0.0.1 --port 8000
