#!/usr/bin/env bash
# M3 window. This script does not stop or start dsv41.
# Between steps it waits until you type: go
# Logs go to $INKLING2_HOME/m3/<timestamp>/
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
. "$ROOT/config/recipe.env"
: "${INKLING2_HOME:=$HOME/.inkling2}"
: "${INKLING2_MODEL_DIR:=/data/models/Inkling-2-Small}"
: "${INKLING2_PLACEMENT:=$INKLING2_HOME/placement.json}"
: "${INKLING2_PORT:=${INKLING2_DEFAULT_PORT}}"
: "${INKLING2_HOST:=${INKLING2_DEFAULT_HOST}}"
: "${INKLING2_SEGMENT:=${INKLING2_DEFAULT_SEGMENT}}"
YES=0
if [ "${1:-}" = --yes ]; then YES=1; fi
if [ "${1:-}" = --dry-run ]; then
  cat <<EOF
m3 steps, each an estimate, not a measurement. The plan's whole window is 4-6 h.
  1 preflight          about 2 min
  2 kernels/build.sh   about 15 min
  3 inkling2 validate  about 60-90 min
  4 routing profile    about 20-40 min, including eager model load
  5 inkling2 plan      about 2 min
  6 split smoke        about 30-180 min; a first image compile can take the rest of the window
  7 eval spot check    about 10 min
  8 teardown           about 1 min, then the human restores dsv41
EOF
  exit 0
fi
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$INKLING2_HOME/m3/$STAMP"
mkdir -p "$LOG"
exec > >(tee -a "$LOG/run.log") 2>&1
echo "log dir $LOG"

pause() {
  echo "estimate: $1 (not a measurement)"
  if [ "$YES" = 1 ]; then return 0; fi
  printf 'type go to continue> '
  read -r ans
  [ "$ans" = go ]
}

gpu() {
  nvidia-smi --query-gpu=index,name,memory.free,uuid --format=csv,noheader
}

echo "step 1 preflight"
pause "about 2 min"
if command -v docker >/dev/null 2>&1; then
  if docker inspect ds41 >/dev/null 2>&1 || docker inspect ds41-sidecar >/dev/null 2>&1; then
    echo "ds41 containers are still present. This script will not stop them." >&2
    exit 1
  fi
fi
echo "host MemAvailable: $(awk '/MemAvailable/ {print $2}' /proc/meminfo) kB"
echo "Grace is about 494 GiB. dsv41 normally holds a large part of it, which is why it has to be down before this step."
export INKLING2_PLACEMENT
python3 - << 'PY'
import json
import os
info = open("/proc/meminfo").read()
avail = int(info.split("MemAvailable:")[1].split()[0]) * 1024
total = int(info.split("MemTotal:")[1].split()[0]) * 1024
# Default spill tier: 96 NVFP4 experts across 40 layers.
need = 96 * 40 * 17915912
placement = os.environ.get("INKLING2_PLACEMENT", "")
if placement and os.path.isfile(placement):
    doc = json.load(open(placement))
    need = 0
    for row in doc["layers"].values():
        n = len(row.get("spill") or [])
        need += n * (17915912 if row["quant"] == "nvfp4" else 32845824)
    print(f"spill bytes from {placement}")
else:
    print("no placement file yet; using the default 96 x 40 NVFP4 spill tier")
print(f"MemTotal {total} bytes ({total/2**30:.2f} GiB); Grace is about 494 GiB")
print(f"spill tier needs at least {need} bytes of packed experts ({need/2**30:.2f} GiB)")
print(f"MemAvailable {avail} bytes ({avail/2**30:.2f} GiB)")
if avail < need:
    raise SystemExit("not enough free RAM for the Grace spill tier; dsv41 has to be down already")
PY
gpu | tee "$LOG/gpus.txt"
[ -f "$INKLING2_MODEL_DIR/config.json" ] || { echo "missing $INKLING2_MODEL_DIR/config.json" >&2; exit 1; }
rev=$(python3 -c "import json; print(json.load(open('$INKLING2_MODEL_DIR/config.json')).get('revision',''))" || true)
echo "model dir $INKLING2_MODEL_DIR"

echo "step 2 compile"
pause "about 15 min"
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.2} "$ROOT/kernels/build.sh" | tee "$LOG/build.log"

echo "step 3 validate"
pause "about 60-90 min"
INKLING2_MODEL_DIR="$INKLING2_MODEL_DIR" python3 "$ROOT/peer/validate.py" --model-dir "$INKLING2_MODEL_DIR" \
  | tee "$LOG/validate.log"

echo "step 4 routing profile"
pause "about 20-40 min, including eager model load"
GB=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | awk -F', ' 'index($1,"GB300"){print $2; exit}')
[ -n "$GB" ] || { echo "no GB300" >&2; exit 1; }
docker rm -f inkling2 >/dev/null 2>&1 || true
docker run -d --name inkling2 --gpus "device=${GB}" --ipc host --network host \
  -v "$INKLING2_MODEL_DIR:/model:ro" -v "$ROOT:/w:ro" -v "$INKLING2_HOME:/out" \
  -e PYTHONPATH=/w/routing \
  -e INKLING2_ROUTE_OUT=/out/profile.json \
  -e HF_HUB_OFFLINE=1 -e VLLM_USE_V2_MODEL_RUNNER=1 \
  "$INKLING2_SUPPORTED_IMAGE" \
  --model /model --served-model-name inkling-2-small \
  --tensor-parallel-size 1 --enforce-eager \
  --host "$INKLING2_HOST" --port "$INKLING2_PORT"
for _ in $(seq 1 120); do
  curl -sf "http://${INKLING2_HOST}:${INKLING2_PORT}/health" && break
  sleep 10
done
curl -sf "http://${INKLING2_HOST}:${INKLING2_PORT}/health" >/dev/null
while IFS= read -r line; do
  [ -z "$line" ] && continue
  case $line in \#*) continue ;; esac
  curl -s "http://${INKLING2_HOST}:${INKLING2_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"model":"inkling-2-small","messages":[{"role":"user","content":sys.argv[1]}],"max_tokens":8}))' "$line")" \
    >> "$LOG/calibration.jsonl" || true
done < "$ROOT/routing/calibration.txt"
docker rm -f inkling2 >/dev/null
echo "profile $INKLING2_HOME/profile.json"

echo "step 5 plan"
pause "about 2 min"
"$ROOT/inkling2" plan --from-hf --profile "$INKLING2_HOME/profile.json" --out "$INKLING2_PLACEMENT" \
  | tee "$LOG/plan.log"

echo "step 6 split smoke"
pause "about 30-180 min; a first image compile can take the rest of the window"
RTX=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader | awk -F', ' 'index($1,"RTX"){print $2; exit}')
[ -n "$RTX" ] || { echo "no RTX" >&2; exit 1; }
rm -f "$INKLING2_SEGMENT"
python3 - << PY
import os, sys
sys.path[:0] = ["$ROOT/peer"]
from segment import Layout, Segment
Segment(os.environ.get("INKLING2_SEGMENT", "$INKLING2_SEGMENT"), Layout(), create=True).close()
PY
CUDA_VISIBLE_DEVICES="$RTX" INKLING2_ALLOW_UNVALIDATED=1 INKLING2_MODEL_DIR="$INKLING2_MODEL_DIR" \
  python3 "$ROOT/peer/sidecar.py" --segment "$INKLING2_SEGMENT" --placement "$INKLING2_PLACEMENT" \
  --model-dir "$INKLING2_MODEL_DIR" --so "$ROOT/peer/prebuilt/peer_inkling_rtx_sm120.so" \
  > "$LOG/sidecar.log" 2>&1 &
echo $! > "$LOG/sidecar.pid"
docker run -d --name inkling2 --gpus "device=${GB}" --ipc host --network host \
  -v "$INKLING2_MODEL_DIR:/model:ro" -v "$ROOT:/w:ro" -v "$INKLING2_SEGMENT:/dev/shm/inkling2-peer" \
  -e PYTHONPATH=/w/routing \
  -e INKLING2_PEER=1 -e INKLING2_ALLOW_UNVALIDATED=1 \
  -e INKLING2_SPILL_POOL_GIB="${INKLING2_SPILL_POOL_GIB:-4}" \
  -e INKLING2_MODEL_DIR=/model -e INKLING2_PLACEMENT=/w/placement-mount.json \
  -e INKLING2_GB300_SO=/w/peer/prebuilt/peer_inkling_gb300_sm103.so \
  -e INKLING2_SEGMENT=/dev/shm/inkling2-peer \
  -e HF_HUB_OFFLINE=1 -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -v "$INKLING2_PLACEMENT:/w/placement-mount.json:ro" \
  "$INKLING2_SUPPORTED_IMAGE" \
  --model /model --served-model-name inkling-2-small \
  --tensor-parallel-size 1 --enforce-eager \
  --host "$INKLING2_HOST" --port "$INKLING2_PORT"
for _ in $(seq 1 180); do
  curl -sf "http://${INKLING2_HOST}:${INKLING2_PORT}/health" && break
  sleep 10
done
curl -s "http://${INKLING2_HOST}:${INKLING2_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-small","messages":[{"role":"user","content":"Say hi."}],"max_tokens":16}' \
  | tee "$LOG/smoke.json"

echo "step 7 spot check"
pause "about 10 min"
python3 - << PY | tee "$LOG/eval-excerpt.txt"
import os, pathlib
root = pathlib.Path(os.environ.get("INKLING2_MODEL_DIR", "$INKLING2_MODEL_DIR")) / ".eval_results"
print("dir", root, "exists", root.is_dir())
if root.is_dir():
    files = sorted(root.rglob("*"))[:8]
    for path in files:
        if path.is_file():
            print("---", path.name)
            print(path.read_text(errors="replace")[:800])
PY
curl -s "http://${INKLING2_HOST}:${INKLING2_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-small","messages":[{"role":"user","content":"Name one benchmark a model card might report. One line."}],"max_tokens":32}' \
  | tee "$LOG/spot.json"
echo "spot check saved the excerpt and one generation. It is not an eval score."

echo "step 8 teardown"
pause "about 1 min"
if [ -f "$LOG/sidecar.pid" ]; then kill "$(cat "$LOG/sidecar.pid")" 2>/dev/null || true; fi
docker rm -f inkling2 >/dev/null 2>&1 || true
rm -f "$INKLING2_SEGMENT"
echo "inkling2 is stopped. This script did not start dsv41."
echo "When you want dsv41 back, you run: sudo sh ~/ds41.sh uncensored up"
echo "done $LOG"
