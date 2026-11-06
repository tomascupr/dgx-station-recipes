#!/usr/bin/env bash
# One-screen PASS/FAIL for a running glm53 server, against the published numbers (verify/published.json, RESULTS.md).
#   verify.sh [--full]     default: windows of 30 s, decode at 1 and 4 streams, one coding run (~6 min)
#                          --full: 60 s windows, decode at 1/4/8/16 streams, three coding runs (~12 min)
# Speed checks pass at 80 % of the published number, GSM8K-200 at 95 %.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
. "$ROOT/config/recipe.env"
: "${GLM53_PORT:=30004}" "${GLM53_IMAGE:=$GLM53_SUPPORTED_IMAGE}"
B="http://127.0.0.1:$GLM53_PORT"
common=(--base-url "$B" --key-file "$HERE/no-key" --model glm-5.3-flash)
window=30 warmup=10 conc=1,4 runs=1
[ "${1:-}" = --full ] && { window=60 warmup=20 conc=1,4,8,16 runs=3; }
pass=0 fail=0
check() { local name=$1; shift; if "$@"; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "$name FAIL"; fi; }

pre_image() {
  docker image inspect -f '{{join .RepoDigests " "}}' "$GLM53_IMAGE" 2>/dev/null | grep -q "$GLM53_IMAGE_DIGEST" &&
    echo "preflight image PASS ($GLM53_IMAGE)"
}
pre_gpus() {
  local n
  n=$(nvidia-smi --query-gpu=name --format=csv,noheader) && grep -q GB300 <<< "$n" && grep -q 'RTX PRO 6000' <<< "$n" &&
    echo "preflight GPUs PASS (GB300 + RTX PRO 6000)"
}
pre_boot() {
  local v s
  v=$(docker logs glm53 2>&1) && s=$(docker logs glm53-sidecar 2>&1) || return 1
  [ "$(grep -c 'GLM53_PEER ACTIVE: 42 layers split' <<< "$v")" = 1 ] || { echo "boot: no single GLM53_PEER ACTIVE line"; return 1; }
  grep -q 'GLM53_SIDECAR ACTIVE' <<< "$s" || { echo "boot: the sidecar never went ACTIVE"; return 1; }
  ! grep -q 'GLM53_PEER FATAL' <<< "$v" || { echo "boot: GLM53_PEER FATAL in the log"; return 1; }
  curl -sf -o /dev/null "$B/health" || { echo "boot: $B/health does not answer"; return 1; }
  echo "preflight boot PASS (peer tier ACTIVE on 42 layers, sidecar ACTIVE, healthy)"
}

echo "== glm53 verify: $B"
check "preflight image" pre_image
check "preflight GPUs" pre_gpus
check "preflight boot" pre_boot
if [ "$fail" != 0 ]; then echo "== $pass PASS, $fail FAIL (preflight; fix the FAIL lines above)"; exit 1; fi
check "tool call" python3 "$HERE/tool.py" "${common[@]}"
check "needles 200K" python3 "$HERE/needle.py" "${common[@]}" --tokens 200000
check "GSM8K-200" python3 "$HERE/gsm8k.py" "${common[@]}" --min-acc 95 --out "${TMPDIR:-/tmp}/glm53-gsm8k-200.jsonl"
check "decode" python3 "$HERE/decode.py" "${common[@]}" --conc "$conc" --window "$window" --warmup "$warmup" --min-pct 80
check "coding" python3 "$HERE/coding.py" "${common[@]}" --runs "$runs" --min-pct 80
echo "== $pass PASS, $fail FAIL"
[ "$fail" = 0 ]
