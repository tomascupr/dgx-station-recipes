# Inkling 2 Small on a DGX Station (native NVFP4 + MXFP8)

By Tomas Cupr, following the expert split in Garner McCloud's
[`deepseek-v4.1-flash`](../deepseek-v4.1-flash/) and the recipe layout of
[`glm-5.3-flash-fp8`](../glm-5.3-flash-fp8/).

This directory is the CPU-side port of that split onto
[thinkingmachines/Inkling-2-Small](https://huggingface.co/thinkingmachines/Inkling-2-Small)
at revision `8361e7a31ae7e9966a49a91d3994fde618936c2b`. Weights stay as shipped:
NVFP4 on layers 0–39, MXFP8 on layers 40–47. No GGUF, no requant.

Nothing here has been run on the Station. There is no measured tok/s. The
bandwidth table in the 14 Nov port plan is an estimate, not a result, and it
is not copied here as a forecast.

## What is validated on a CPU

`./inkling2 test` checks, with numpy and synthetic safetensors headers:

* Per-expert sizes from the real header shapes: **17,915,912** bytes (NVFP4) and
  **32,845,824** bytes (MXFP8), including scales and amax, not the 8-byte
  layer-shared denominator.
* The default placement: all 384 MXFP8 experts on the GB300, and NVFP4 ids
  **158 / 130 / 96** on GB300 / RTX / Grace. Those counts are the byte totals
  **214,130,935,168 / 93,162,742,400 / 68,797,102,080**.
* A frequency profile reorders NVFP4 only. MXFP8 stays on the GB300.
* E2M1 packing (low nibble first), E4M3, NVFP4 `amax / denominator`, MXFP8
  `2^(ue8m0-127)`, the sigmoid-bias top-8 router, interleaved-row SwiGLU, and
  that hot + cold + spill equals the full layer.
* One expert row can be read out of a packed tensor without the other rows.
* The `/dev/shm` record layout matches `kernels/csrc/inkling_proto.cuh`, and a
  mock sidecar round-trip matches the reference.
* `up` / `down` do not contain a command that stops `ds41`.

## What is not validated

* The CUTLASS serving GEMMs (`kernels/csrc/inkling_grouped.cuh`, built
  `-arch=sm_103a` and `-arch=sm_120a`). They have not been executed. The FP64
  kernels remain the numeric reference. `./kernels/build.sh` needs
  `CUDA_HOME` on the Station (`/usr/local/cuda-13.2`) and CUTLASS
  `b2f249acd203`.
* The vLLM hook (`peer/hook.py`, `routing/collect.py`) against image
  `vllm/vllm-openai:inkling-2-small` or PR
  [vllm#67325](https://github.com/vllm-project/vllm/pull/67325) head `d664cd1e`.
  Class names were read from that commit. The PR was open and not mergeable.
* SM 10.3 and SM 12.0, CUDA graphs, Grace KV offload, DSpark acceptance, and
  any comparison with the checkpoint's `.eval_results` YAML.

## Commands

```bash
./inkling2 test
./inkling2 plan --from-hf          # headers only, via HTTP Range
./inkling2 status
./inkling2 up                      # refuses while ds41 or ds41-sidecar exists
./inkling2 down                    # removes the inkling2 container only
```

`up` listens on port 30008 unless `INKLING2_PORT` is set. It is day-0 vLLM
on the GB300, with no peer hook and no RTX sidecar. It does not stop dsv41.
If those containers exist, it exits. Taking dsv41 down needs the owner's
approval, and then that recipe's own `./ds41 down`.

The first GPU window is `docs/SMOKE.md` (the plan's 4–6 hour estimate for
stop, profile, one generate, restore). `docs/ARCHITECTURE.md` records where
the port disagrees with the 14 Nov plan.

## Licence

MIT (`../LICENSE`), copyright 2026 Tomas Cupr, apart from the design taken from
Garner's DeepSeek recipe. Weights stay on the Hub. See `NOTICE`.
