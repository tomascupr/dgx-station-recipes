#!/usr/bin/env bash
# Reference server for the DGX Station recipe: GLM-5.3-Flash (native FP8) on two B300s, stock vLLM.
# vLLM's recipe command (recipes.vllm.ai/zai-org/GLM-5.3-Flash: b300 row + the GB200-tray MTP example), except:
#  - TP2 on GPUs 4-5, CPUs on their NUMA node, port on 127.0.0.1 only, weights at the pinned revision, offline;
#  - --gpu-memory-utilization 0.85: at the default 0.9 the KV cache took 101 GiB per GPU and CUDA-graph capture
#    then ran out of memory (2026-11-05); 0.85 gives up KV space;
#  - every JIT cache (vLLM, FlashInfer, Triton, TileLang, CUDA) under /data/cache/glm53, so restarts skip the warmup.
# Run: serve-glm53-b300.sh [extra vLLM flags]. GLM53_MOUNTS adds docker -v/-e flags; GLM53_CPUS
# narrows the NUMA-1 cores when two of these servers run side by side.
set -euo pipefail
IMAGE=vllm/vllm-openai:nightly-a2bcb5657a881a002e79d0adab3c6b5e39436121
REV=46a050bec13002ad29ee795c16e61cf4fa3c4a9c
: "${GLM53_GPUS:=4,5}" "${GLM53_PORT:=8500}" "${GLM53_NAME:=glm53}" "${GLM53_CPUS:=64-127,192-255}"
docker run -d --name "$GLM53_NAME" --gpus "\"device=$GLM53_GPUS\"" --cpuset-cpus "$GLM53_CPUS" --ipc=host \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_USE_RUST_FRONTEND=1 -e HF_HUB_OFFLINE=1 \
  -e TRITON_CACHE_DIR=/root/.cache/triton -e TILELANG_CACHE_DIR=/root/.cache/tilelang \
  -e CUDA_CACHE_PATH=/root/.cache/nv-compute \
  -v /data/cache/glm53:/root/.cache -v /data/hf:/root/.cache/huggingface ${GLM53_MOUNTS:-} \
  -p "127.0.0.1:$GLM53_PORT:8000" "$IMAGE" \
  zai-org/GLM-5.3-Flash --revision "$REV" --served-model-name glm-5.3-flash zai-org/GLM-5.3-Flash \
  --tokenizer-mode glm5_next --tensor-parallel-size 2 --kv-cache-dtype fp8 --gpu-memory-utilization 0.85 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --tool-call-parser glm47 --reasoning-parser glm47 --enable-auto-tool-choice "$@"
