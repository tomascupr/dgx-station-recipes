# First GPU window

The plan's estimate for this window is **4–6 hours** of wall time. That is an
estimate of the procedure, not a measurement. Do not start it until the owner
has approved taking dsv41 down. This script will not take it down for you.

## 0. Before the window

On a CPU, from this directory:

```bash
./inkling2 test
./inkling2 plan --from-hf --out "$HOME/.inkling2/placement.json"
```

`plan` writes the 158/130/96 map from Hub headers. It does not download
weights. Confirm the checkpoint at `/data/models/Inkling-2-Small` is revision
`8361e7a31ae7e9966a49a91d3994fde618936c2b` and the directory size is about
393,509,649,692 bytes (the Hub `usedStorage` figure in the plan).

Compile only. The script does not launch a kernel and does not change
`CUDA_VISIBLE_DEVICES`. nvcc is not on PATH on the Station:

```bash
CUDA_HOME=/usr/local/cuda-13.2 ./kernels/build.sh
```

That is `-arch=sm_103a` and `-arch=sm_120a`, with CUTLASS `b2f249acd203`
(cloned into `kernels/third_party/cutlass` if `CUTLASS_ROOT` is unset).
`./kernels/build.sh --dry-run` prints the commands and does not download.

No `.so` is produced on an x86 machine without `nvcc`. That is expected. The
objects are `peer/prebuilt/peer_inkling_gb300_sm103.so` and
`peer_inkling_rtx_sm120.so`. They are UNVALIDATED even after a compile: they
have not been launched.

## 1. Stop dsv41 (owner approval, ~30–60 min in the plan)

```bash
cd ../deepseek-v4.1-flash && ./ds41 down
```

Confirm the GB300 and the RTX are free (`nvidia-smi`). Leave enough time at
the end of the window to bring dsv41 back and see it serve. Its first start
can recompile.

## 2. Day-0 smoke on the GB300 (~the rest of the window)

```bash
cd ../inkling-2-small
export INKLING2_MODEL_DIR=/data/models/Inkling-2-Small
./inkling2 up
```

This starts `vllm/vllm-openai:inkling-2-small` on the GB300 only. It does not
mount the peer hook and does not start a sidecar. First launch compiles and
can take hours; watch `docker logs -f inkling2`. Success for this step is one
health check and one short generate:

```bash
curl -sf http://127.0.0.1:${INKLING2_PORT:-30008}/health
curl -s http://127.0.0.1:${INKLING2_PORT:-30008}/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-small","messages":[{"role":"user","content":"Say hi."}],"max_tokens":16}'
```

If the process dies in an SM 10.3 kernel, stop. That is the plan's signal to
try `lmsysorg/sglang:dev-inkling-2-small` later, not to debug the unvalidated
`.cu` in the same window. Save the log.

## 3. Routing profile (same window, if the day-0 process runs)

Restart with the collector, eager, no CUDA graphs. The collector imports
`vllm.models.inkling2.moe.Inkling2MoE` and writes
`inkling2-expert-profile-v1` JSON. It is unvalidated against the image.

```bash
./inkling2 down
docker run --rm --gpus device=<GB300-uuid> --network host --ipc host \
  -v "$INKLING2_MODEL_DIR:/model:ro" -v "$PWD:/w:ro" \
  -v "$HOME/.inkling2:/out" \
  -e PYTHONPATH=/w/routing \
  -e INKLING2_ROUTE_OUT=/out/profile.json \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  vllm/vllm-openai:inkling-2-small \
  --model /model --served-model-name inkling-2-small \
  --tensor-parallel-size 1 --enforce-eager \
  --host 127.0.0.1 --port "${INKLING2_PORT:-30008}"
```

Send a calibration set (a few thousand tokens of the prompts you care about),
then `./inkling2 down` so the process exits and the profile is flushed.
Rebuild the map:

```bash
./inkling2 plan --from-hf --profile "$HOME/.inkling2/profile.json" \
  --out "$HOME/.inkling2/placement.json"
```

The new file still keeps every MXFP8 expert on the GB300. Read the printed
byte totals before trusting it.

## 4. Peer hook

Do not enable `INKLING2_PEER` in this first window unless the day-0 generate
already worked and time remains. With the variable set, `peer/hook.py`
replaces `Inkling2MoE.forward` and the hot GEMM raises, on purpose, because
`kernels/csrc` has not been compared to `kernels/reference/quant.py`. A later
window can load the `.so` and diff a handful of tokens against that reference
before any benchmark.

## 5. Restore dsv41

```bash
./inkling2 down
cd ../deepseek-v4.1-flash && ./ds41 up
```

Confirm a short generate on the DeepSeek port. Do not leave the box on Inkling
if dsv41 was the service the owner expects.

## Quality parity (later window, plan estimate 4–8 h)

The bar is the checkpoint's own `.eval_results/` YAML (HMMT, GPQA, SWE-bench,
Terminal-Bench, and the rest of that directory), not a visual sample. Run the
same tasks the YAML names, at the same agent versions it names, and write the
delta down. Peer numerics are not bit-identical to the day-0 kernel until
someone shows they are. There is no script in this tree that runs those
evals; they need the GPU window and the eval harnesses, which are not vendored
here.

## Perf (later window, plan estimate 2–3 h)

Batch 1 and 8 streams, after quality. Report the measured tok/s next to the
plan's bandwidth estimates and say which is which. Do not describe a match to
those estimates as hitting a forecast. This tree does not print a tok/s
figure.
