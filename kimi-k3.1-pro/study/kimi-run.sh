#!/bin/sh
# sudo sh kimi-run.sh NAME 'shell command'   one-off container on the GB300 in the vLLM nightly image the GLM recipe uses
# KIMI_HOME: working directory holding study/ (this directory), pydeps/ (fla-core, einops) and the outputs;
# KIMI_MODELS: the directory with the downloaded model; KIMI_MOUNT: an optional extra docker volume (src:dst)
: "${KIMI_HOME:?set KIMI_HOME}" "${KIMI_MODELS:?set KIMI_MODELS}"
GB300=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | awk -F', ' '/GB300/ { print $2; exit }')
[ -n "$GB300" ] || { echo 'no GB300 visible' >&2; exit 1; }
exec docker run --rm --name "$1" --gpus device=$GB300 --ipc host --network host --entrypoint bash \
  -v $KIMI_HOME:$KIMI_HOME -v $KIMI_MODELS:$KIMI_MODELS:ro ${KIMI_MOUNT:+-v $KIMI_MOUNT} \
  -e PYTHONPATH=$KIMI_HOME/pydeps -e TRITON_CACHE_DIR=$KIMI_HOME/cache/triton -e HF_HUB_OFFLINE=1 \
  -w $KIMI_HOME/study vllm/vllm-openai:nightly-a2bcb5657a881a002e79d0adab3c6b5e39436121 -c "$2"
