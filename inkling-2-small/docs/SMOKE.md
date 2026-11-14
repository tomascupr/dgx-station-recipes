# First GPU window

The plan's estimate for this window is **4–6 hours** of wall time. That is an
estimate of the procedure, not a measurement. Do not start it until the owner
has approved taking dsv41 down. This script will not take it down for you.

## Commands

From `inkling-2-small`, after the owner has already stopped dsv41:

```bash
./inkling2 m3
```

That is the whole window. It stops between steps until you type `go`, and
writes `$HOME/.inkling2/m3/<timestamp>/`. `./inkling2 m3 --dry-run` prints
the steps and does not touch a GPU. Estimates below are not measurements.
The plan's figure for the whole window is 4–6 h.

| Step | Command inside m3 | Estimate |
|---|---|---|
| 1 | preflight: dsv41 already gone, free memory | about 2 min |
| 2 | `CUDA_HOME=/usr/local/cuda-13.2 ./kernels/build.sh` | about 15 min |
| 3 | `./inkling2 validate` | about 60–90 min |
| 4 | eager vLLM, `routing/calibration.txt` (about 4k tokens, chars/4) | about 20–40 min, including model load |
| 5 | `./inkling2 plan --from-hf --profile ...` | about 2 min |
| 6 | sidecar on the RTX plus vLLM with `INKLING2_ALLOW_UNVALIDATED=1`, one generate | about 30–180 min; a first image compile can take the rest of the window |
| 7 | print a few `.eval_results` files and one more generate | about 10 min |
| 8 | stop inkling2 and the sidecar | about 1 min |

Step 8 does not start dsv41. You do that yourself:

```bash
sudo sh ~/ds41.sh uncensored up
```

`./inkling2 validate --thresholds` prints the pass bar without a GPU. The bar
is max absolute error at most 2.5% of max(|fp64 reference|, 1). The serving
kernel stores bf16 three times (gate/up, SiTU, down); one bf16 ulp is 1/128.

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

That is `compute_103a/sm_103a` and `compute_120a/sm_120a` (not a plain
`compute_103` PTX, which rejects the block-scaled instructions), with
CUTLASS `b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd`
(cloned into `kernels/third_party/cutlass` if `CUTLASS_ROOT` is unset).
`./kernels/build.sh --dry-run` prints the commands and does not download.

No `.so` is produced on an x86 machine without `nvcc`. That is expected. The
objects are `peer/prebuilt/peer_inkling_gb300_sm103.so` and
`peer_inkling_rtx_sm120.so`. They are UNVALIDATED even after a compile: they
have not been launched.

The operator stops dsv41 before `./inkling2 m3`. The script checks that the
ds41 containers are already gone and refuses if they are not. It never stops
them. Step 6 is the split: the GB300 process loads `peer/hook.py` with
`INKLING2_ALLOW_UNVALIDATED=1`, and the RTX sidecar loads the sm_120a object.
Without that flag the hook raises and the sidecar stays on the numpy
reference. If the GB300 process dies in an SM 10.3 kernel during step 4 or 6,
stop the window. That is the plan's signal to try
`lmsysorg/sglang:dev-inkling-2-small` later. The planner still keeps every
MXFP8 expert on the GB300.

## Quality parity (later window, plan estimate 4–8 h)

The bar is the checkpoint's own `.eval_results/` YAML (HMMT, GPQA, SWE-bench,
Terminal-Bench, and the rest of that directory), not a visual sample. Run the
same tasks the YAML names, at the same agent versions it names, and write the
delta down. Peer numerics are not bit-identical to the day-0 kernel until
someone shows they are. There is no script in this tree that runs those
evals; they need the GPU window and the eval harnesses, which are not vendored
here.

## Perf (later window, plan estimate 2–3 h)

Batch 1 and 8 streams, after quality. `./inkling2 validate` prints a measured
microsecond time and TB/s for each expert GEMM call (weights, scales, x, and
y of that call). Those figures are measurements of the call. The plan's
bandwidth numbers stay estimates. Do not describe a match between them as
hitting a forecast. This tree does not print a tok/s figure.
