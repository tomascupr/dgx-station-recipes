# Results

Work in progress. Measured so far: on an 8x B300 server, the reference server (stock vLLM on two B300s) and the expert routing
the rowmap is built from; on the DGX Station, stock vLLM on the GB300 alone (stage 4), the peer tier with the RTX
sidecar (stage 5, v1-v3), today's ds41 for context, and DeepSeek on the GB300 alone for the root README. Every number below names the run it came
from; the raw records are in `results/b300/`.

## Summary

| shape (tok/s unless noted) | GB300 alone | **GB300 + RTX** | **speedup** | 2x B300 (reference) | GB300 + RTX as % of 2x B300 |
|---|---|---|---|---|---|
| fixed prompt, 1 stream, greedy | 74.6 | **216.7** | **2.9x** | 350.6 | 62 % |
| coding prompt, 1 stream, greedy | 64.2 | **249.8** | **3.9x** | 377.1 | 66 % |
| GSM8K, 1 stream, T=1.0 | 81.6 | **242.2** | **3.0x** | 373.4 | 65 % |
| GSM8K, 1 stream, greedy | 63.4 | **197.4** | **3.1x** | 302.5 | 65 % |
| 4 streams, aggregate | 113 | **556** | **4.9x** | 880 | 63 % |
| 8 streams, aggregate | 147 | **617** | **4.2x** | 1,375 | 45 % |
| 16 streams, aggregate | 173 | **640** | **3.7x** | 2,175 | 29 % |
| 32 streams, aggregate | 176 | **633** | **3.6x** | 3,481 | 18 % |
| prefill, ~15K-token prompt | 3.8K | **14.3K** | **3.8x** | 38.0K | 38 % |
| prefill, ~120K-token prompt | 7.9K | **15.6K** | **2.0x** | 36.1K | 43 % |
| GSM8K-200 accuracy | 99.0 % | **99.0 %** | same | 99.0 % | same |

Every row is the same prompts, sampling, MTP setting (5 drafts), image and model flags on all three. The speedup
is GB300 + RTX against the GB300 alone: the same box and model, with and without the sidecar. Two B300s are a
ceiling, not a competitor: two larger GPUs that hold the whole model in HBM. The Station gets about two thirds of
them at one stream, less as streams grow, because the RTX's bandwidth runs out (LIMITATIONS.md, 1). The Station runs
used 30 s windows and 3 runs per shape, the B300 runs 60 s and 10: the same shapes, fewer samples. Runs: `bench-gb300-stock-20261105T235133Z`, `bench-gb300-peer-v3-20261106T010643Z` (accuracy:
`bench-gb300-peer-v3-quality-20261106T011552Z`) in `results/dgx/`, and `bench-20261105T223458Z` (accuracy:
`bench-20261105T221649Z`) in `results/b300/`. The sections below have each run in full.

## Reference: stock vLLM on 2x B300, 2026-11-05

This is the number the DGX Station has to beat and the outputs it is checked against. The B300 is the GB300's GPU
(compute capability 10.3), but this server splits the model over two of them (TP2, NVLink all-reduce), while the
Station runs it on one GB300 plus the RTX. So these are a reference point, not a prediction.

| | |
|---|---|
| Model | `zai-org/GLM-5.3-Flash` at `46a050bec13002ad29ee795c16e61cf4fa3c4a9c`, native FP8, 306 GiB, 74/74 files hash-checked against the Hub |
| Image | `vllm/vllm-openai:nightly-a2bcb5657a881a002e79d0adab3c6b5e39436121` (`sha256:bd419f16…`, vLLM 0.33.0rc1.dev806, FlashInfer 0.7.2.post1, torch 2.14.0) |
| Server | `baseline/serve-glm53-b300.sh`: vLLM's recipe command at TP2 on GPUs 4-5, FP8 KV, MTP with 5 draft tokens, `glm47` tool and reasoning parsers, Rust frontend, `--gpu-memory-utilization 0.85` (0.9 ran out of memory in CUDA-graph capture) |
| Memory per GPU | 158.1 GiB weights and non-torch, 24.0 GiB CUDA graphs, 8.4 GiB peak activations, 31.4 GiB KV cache (vLLM's sizing after warmup) |
| Driver / GPUs | 580.178.04, 2x B300 SXM6 AC (275 GB) |
| Checks | `verify/` at commit `18cb33a`: Garner's grid and prompts, with GLM's effort switch in place of his thinking switch (below) |

GLM-5.3-Flash always reasons: its chat template opens `<think>` unconditionally, and `reasoning_effort` low / high /
max (default max) is the only control. Where Garner's checks turn thinking off, these run at effort low; where they
turn it on, at effort max. Tokens per step is the MTP acceptance: how many tokens each verify step committed.

### Single stream

| shape | run 1: output tok/s | run 2: output tok/s | TTFT ms | tokens/step | steps/s |
|---|---|---|---|---|---|
| fixed prompt (~2.6K in, 2,048 out, greedy, effort low; 10 runs) | 346.6 (339.6-358.6) | 350.6 (341.5-355.6) | 110 / 111 | 3.58 / 3.73 | 96.7 / 93.9 |
| coding prompt (2,048 out, greedy, effort low; 5 runs) | 382.5 (367.7-384.0) | 377.1 (366.6-387.5) | 132 / 132 | 4.11 / 4.05 | 93.0 / 93.3 |
| GSM8K, T=1.0, effort max (60 s window) | 379.4 | 373.4 | — | 4.27 / 4.30 | — |
| GSM8K, greedy, effort low (60 s window) | 311.1 | 302.5 | — | 3.97 / 3.98 | — |

### Many streams (GSM8K, T=1.0, effort max, 60 s windows)

| streams | run 1: aggregate tok/s | run 2: aggregate tok/s | run 2: per stream | tokens/step |
|---|---|---|---|---|
| 4 | 896 | 880 | 220 | 4.20 / 4.12 |
| 8 | 1,396 | 1,375 | 172 | 4.23 / 4.16 |
| 16 | 2,213 | 2,175 | 136 | 4.28 / 4.23 |
| 32 | 3,411 | 3,481 | 109 | 4.18 / 4.26 |

No preemptions or request errors in any window.

### Prefill (uncached, distinct filler per request)

| prompt tokens | TTFT ms | prefill tok/s |
|---|---|---|
| ~14.9K | 399 / 393 | ~37-38K |
| ~59.7K | 1,649 / 1,651 | ~36.2K |
| ~119.4K | 3,301 / 3,303 | ~36.1K |

### Quality and capacity (run 1)

| check | result |
|---|---|
| GSM8K-200 (effort max, T=0.3, 16 at a time) | 198/200 = 99.0 %, 0 runaways, 0 errors, median 179 completion tokens, 19 s wall |
| needles at 182,183 prompt tokens, 10 / 50 / 90 % depth | 3/3, 5 s each |
| tool call | PASS: `get_weather {"city":"Denver"}` |
| 16 distinct ~110K-token contexts decoding 2,500 forced tokens + a ~500K-token needle | all correct, 0 preemptions, peak KV use 62 %, 82 s wall; at most 16 of the 17 ran at once |

### Runs

* **Run 1**, `bench-20261105T221649Z`: the full suite. The routing server (below) was starting up on GPUs 6-7
  during it, on the same CPU cores.
* Both runs: `glm53` was started from an earlier version of `baseline/serve-glm53-b300.sh` with the same vLLM
  flags (the environment files record them); only its cache mounts differ from the committed script.
* **Run 2**, `bench-20261105T223458Z`: the speed stages again (`--skip capacity,quality`) after the two servers were
  given separate cores (`docker update --cpuset-cpus`), with the routing server serving on GPUs 6-7. Runs 1 and 2
  agree within 1-3 %, which is about what MTP acceptance alone moves between runs, so the overlap did not matter.

### Reference outputs

`baseline/reference/reference-b300-20261105.jsonl`: 33 temperature-0 answers with their full reasoning, from
`baseline/reference.py` (verify/data prompts only). GSM8K 19/20 at effort max (median 214 completion tokens) and
10/10 at effort low (median 68); the tool call is correct. Compare answers and tool calls, not bytes: temperature-0
output is not bit-reproducible across servers.

### How to reproduce

```bash
# on a server with B300s
baseline/serve-glm53-b300.sh                              # ~13 min cold, ~6.5 warm
cd glm-5.3-flash-fp8/verify                               # at commit 18cb33a
python3 bench_full.py --base-url http://127.0.0.1:8500 --key-file NO_KEY_FILE --model glm-5.3-flash \
  --env-file bench-STAMP.env --out bench-STAMP.json [--skip capacity,quality]
python3 ../baseline/reference.py reference.jsonl --base-url http://127.0.0.1:8500
```

The environment file records what was running (image, digest, driver, GPUs, flags); `results/b300/` has both
runs' JSON, table, log and environment, the GSM8K answers and the capacity record.

## Expert routing and the rowmap, 2026-11-05

`routing/rowmap-mix-v1-h202.json` keeps 202 of each MoE layer's 288 experts on the GB300 and puts 86 on the RTX
(84.7 GiB, the most that fits in the ~85 GiB the RTX can give to weights). The MTP layer stays whole on the GB300.
Chosen from measured routes, it sends **16.8 % of decode and 17.8 % of prefill routed tokens to the RTX**, against
29.9 % for a split that ignores routing.

| | |
|---|---|
| Counting | `routing/sitecustomize.py` in a second copy of the reference server (`--enforce-eager`, GPUs 6-7): every MoE call adds its top-k expert ids per layer, bucketed by call size. Calls of up to 64 tokens count as decode (the peer tier's fused path); with MTP each stream sends 6 tokens per step, so the mix ran at 10 streams |
| Sanity check | three single-token requests between two snapshots: every one of the 42 layers gained exactly the prompts' token counts and 8 hits per token, and no invalid ids |
| Traffic | the first 490 requests of `routing/build_mix.py`'s shuffled 2,471-request mix (`mix-v1`, sha256 `9d436e78…`): 102 reasoning, 89 coding, 87 tool-call, 83 agentic coding, 75 chat, 54 multilingual; effort low 30 % / high 30 % / max 40 %, T=1.0. 710,919 prompt and 388,695 completion tokens, 0 errors. Through the MoE: 764,370 decode-call and 761,773 prefill-call tokens per layer |
| Scoring | per layer, 0.6 x decode share + 0.4 x prefill share (Garner's weighting); the 86 lowest go cold |

The run stopped at 490 requests because more traffic stopped changing the answer. Each check below picks cold sets
from earlier traffic and applies them to later traffic, against the best possible cold sets for that later traffic
(which overfit it, so they are an optimistic bound):

| cold sets from | tested on | decode to RTX | best possible | prefill to RTX | best possible | same cold experts |
|---|---|---|---|---|---|---|
| requests 0-150 | 150-300 | 16.9 % | 15.2 % | 18.3 % | 17.2 % | 80 % |
| requests 0-300 | 300-489 | 18.5 % | 16.6 % | 18.2 % | 18.5 % | 83 % |

Doubling the calibration did not close the gap, so what remains is traffic varying between slices, not too little
data. Expect 17-19 % of routed tokens on the RTX in use. The early layers route more evenly: layers 3-6 send 22-25 %
of their decode tokens to the RTX, the others less (a candidate for uneven per-layer splits).

Reproduce: `routing/build_mix.py` (the mix), the reference server with `GLM53_MOUNTS` for the hook and
`--enforce-eager`, `routing/drive.py MIX LOG --conc 10`, then
`routing/rowmap.py results/b300/routing/snap-before.json results/b300/routing/snap-final.json OUT`. The
snapshots and the per-request log are in `results/b300/routing/`.

## GB300 memory: the split fits only with small CUDA graphs (estimate; stage 4 measured part of it, below)

An estimate from the checkpoint's headers, this reference server's logs and the logs of a GB300-only DeepSeek-V4.1-Flash server on the Station (vLLM's DGX
Station recipe); stage 4's
boot log will measure it. With this rowmap the GB300 holds 221.1 GiB of weights: 202 experts per layer (198.9 GiB), the MTP layer's experts
(6.75 GiB) and everything that is not a routed expert (15.4 GiB, from the checkpoint's headers). vLLM gets about
243 GiB of the GB300 at `--gpu-memory-utilization 0.97` (that DeepSeek server runs there). That leaves ~22 GiB for the runtime.

* The reference server's runtime per GPU was 24.0 GiB of CUDA graphs, 8.4 GiB of peak activations and ~2 GiB outside
  torch (vLLM's own accounting, before the KV cache). Stock settings do not fit.
* That DeepSeek server captures graphs only up to batch 128 and needs 2.2 GiB for them; with that, GLM's runtime is
  roughly 3-5 GiB of graphs, ~8 GiB of activations and non-torch, and ~6 GiB of KV cache. That fits with 3-5 GiB to
  spare.
* Measured since (stage 5's boot log): 6 GiB of KV cache holds 680,326 tokens on the GB300 (10 GiB held 1,135,957 in
  stage 4). Garner's capacity check keeps ~2.26M tokens resident (16 x 110K plus a 500K needle), so it does not run
  on the Station unless weights come off the GB300 or the KV cache overflows to host memory.
* More KV cache needs weights off the GB300: the MTP layer's experts (6.75 GiB) or the least-used experts into Grace
  memory, or the vision tower (1.05 GiB) for text-only serving.

## DGX Station, GB300 alone (stage 4), 2026-11-06

Stock vLLM on the GB300 with no help from the RTX: the number the sidecar has to beat. Same image and model flags as
the reference, built like vLLM's DGX Station recipe for DeepSeek-V4.1-Flash (`baseline/serve-glm53-gb300-stock.sh` at `09401c6`): vLLM put 92.25 GiB
of routed-expert weights in pinned Grace memory (UVA) and kept 215.96 GiB on the GB300; CUDA graphs captured up to
192 tokens took 1.30 GiB; a 10 GiB KV cache holds 1,135,957 tokens (~9.2 KB per token at TP1); init took 673 s on a
cold cache. arm64 mainline vLLM serves the model.

**Quick protocol**, unlike the reference runs: 30 s windows after 10 s of warm-up, 3 fixed-prompt and 2 coding runs,
one prefill pass per size, no capacity stage (`bench_full.py --skip capacity --reps 3 --coding-reps 2 --prefill-reps 1
--window 30 --warmup 10`, run `bench-gb300-stock-20261105T235133Z` in `results/dgx/`).

| shape | GB300 alone | 2x B300 reference (run 2) |
|---|---|---|
| fixed prompt, 1 stream (tok/s) | 74.6 (3 runs: 74.6 / 78.0 / 72.6) | 350.6 |
| coding prompt, 1 stream | 64.2 | 377.1 |
| GSM8K, T=1.0, effort max, 1 stream | 81.6 | 373.4 |
| GSM8K, greedy, effort low, 1 stream | 63.4 | 302.5 |
| 4 / 8 / 16 / 32 streams, aggregate | 113 / 147 / 173 / 176 | 880 / 1,375 / 2,175 / 3,481 |
| prefill at ~15K / 60K / 120K tokens | 3.8K / 8.0K / 7.9K tok/s | ~37K / 36K / 36K tok/s |
| GSM8K-200 | 198/200 (99.0 %), 1 runaway | 198/200 (99.0 %) |
| needles at 182,183 tokens | 3/3, 21-24 s each | 3/3, 5 s each |
| tool call | PASS | PASS |

Steps run at ~20/s (48 ms each) against ~94/s on the B300s, with the same MTP acceptance (3.5-4.4 tokens per step),
so the model computes the same and the time goes to reading the Grace-resident experts over NVLink-C2C every step.
Aggregate throughput stops growing past 16 streams for the same reason. This is the traffic the RTX sidecar removes.

## DGX Station, GB300 + RTX PRO 6000 peer tier (stage 5), 2026-11-06

`peer/serve-glm53-peer.sh` (at `adc5a4d`; now `./glm53 up`, `serve/serve.sh`): vLLM on the GB300 keeps the rowmap's 202 hot experts per layer in HBM; a sidecar process
on the RTX holds the 86 cold ones (84.7 GiB) and computes them; the two exchange each MoE layer's rows through
`/dev/shm` with graph-capturable spin and flag kernels (`peer/channel.py`). One process cannot use both GPUs on the
Station (CUDA reports one device), hence two. Experts run on vLLM's Triton FP8 block-scale kernel on both GPUs with
GLM's clamped SwiGLU (`peer/peer_moe.py`). Same quick protocol as stage 4 (30 s windows, 3 / 2 runs, no capacity).

| | GB300 alone | peer v1 | peer v2 | **peer v3** | 2x B300 reference |
|---|---|---|---|---|---|
| what changed | stage 4 | first version: sidecar launches from Python | sidecar requests as CUDA graphs, tiles tuned | tiles re-tuned as CUDA graphs | |
| fixed prompt, 1 stream | 74.6 | 129.5 | 198.6 | **216.7** | 350.6 |
| coding prompt, 1 stream | 64.2 | 152.9 | 220.4 | **249.8** | 377.1 |
| GSM8K T=1.0, 1 stream | 81.6 | 148.4 | 222.0 | **242.2** | 373.4 |
| GSM8K greedy, 1 stream | 63.4 | 123.8 | 187.2 | **197.4** | 302.5 |
| steps/s (fixed prompt) | 20.6 | 36.3 | 54.9 | **60.2** | 93.9 |
| 4 / 8 / 16 / 32 streams | 113 / 147 / 173 / 176 | 431 / 502 / 491 / 504 | 553 / 643 / 612 / 631 | **556 / 617 / 640 / 633** | 880 / 1,375 / 2,175 / 3,481 |
| prefill ~15K / 60K / 120K | 3.8K / 8.0K / 7.9K | — | 14.3K / 15.9K / 15.7K | **14.3K / 15.7K / 15.6K** | ~37K / 36K / 36K |
| run | `bench-gb300-stock-…235133Z` | `…peer-v1-…002401Z` | `…peer-v2-…004115Z` | `…peer-v3-…010643Z` | run 2 |

MTP acceptance is 3.5-4.3 tokens per step in every column, the same as the reference: the split computes the same
model. Quality: **v3 GSM8K-200 198/200 (99.0 %), needles 3/3 at 182,183 tokens, tool call PASS**
(`bench-gb300-peer-v3-quality-20261106T011552Z`); v2 199/200, 3/3, PASS. v1 stopped after its stream windows.

**Where the time goes (v3, single stream).** Per MoE layer the GB300's hot experts take 193 us as a CUDA graph at 6
tokens (~1 GB of weights, 5.2 TB/s) and the RTX round trip ~185 us (sidecar: 144 us of kernels plus copies and the
wake-up), so the two sides are balanced; the GB300's other work per layer (attention, mHC, router) is ~170 us. 42
layers x ~395 us ~ 16.6 ms per step. **Many streams** stop at ~640 tok/s for a different reason: from ~8 streams on
almost every cold expert is touched every step, so the RTX streams ~85 GB per step at ~1.6 TB/s (~50 ms). That is
the FP8 checkpoint against the RTX's bandwidth, not overhead; 4-bit cold experts would halve it.

**Tried, not adopted (2026-11-06).**

* *Fewer MTP drafts.* 3 drafts make steps cheaper (70.9 steps/s against 60.2) but acceptance falls from 3.6 to 3.15
  tokens per step: fixed prompt 223.8 (+3 %), coding 236.7 (-5 %), GSM8K 228.2 (-6 %), greedy 198.4 (=), 4 streams 547
  (-2 %). 5 drafts stay. A 4-draft run made 5-token calls, which the sidecar had no CUDA graph for, so it fell back to
  its Python path (37 steps/s, v1 speed); not a fair point, and not rerun. (`bench-gb300-peer-v3-mtp3-*`, `-mtp4-*`.)
* *FlashInfer's TRT-LLM kernel for the hot experts* (`peer/trtllm_hot.py`, `peer/trt_test.py`): 8 % faster than the
  graph-tuned Triton kernel at 6 tokens (181 against 197 us, the same margin up to 192 tokens), but the two outputs
  differ by 5 % relative, against ~4 % for either against float32; at ~0.7 ms per step it is not worth changing the
  numerics before that difference is understood.

## The packaged recipe, end to end, 2026-11-06

`./glm53 up` from commit `f90d00a` (weights at an existing path, warm caches) was ready in ~3 min; `./glm53 verify`
then passed preflight and boot gate, the tool call, needles 3/3 at 182,183 tokens, GSM8K-200 198/200 and coding
248.4 tok/s (99 % of published), and failed decode on a bug in `verify/decode.py`'s report line (a string followed by
a parenthesized expression parsed as a call), fixed in the next commit; rerun, decode gave 239 tok/s at 1 stream and
556 at 4 (99 % and 100 %). `results/dgx/verify-packaged-f90d00a.log`.

Reference outputs, the same 33 temperature-0 prompts as on the B300s (`baseline/reference/`): GSM8K at effort max
19/20 on both, with the same final answer on 20/20; the tool call identical; at effort low the B300s got 10/10 and
the Station 8/10 (the two misses are short, well-formed wrong answers after 76 and 86 tokens of reasoning, where
the B300s reasoned 93 and 115). Free text diverges after the first few dozen characters, as temperature 0 does across
servers.

## DGX Station today: ds41 on Garner's suite, 2026-11-05

For context only (a different model): Garner's own `verify/bench_full.py` against a live ds41 (a -1017-based checkpoint,
v0.31.0), same prompts and 60 s windows as the reference runs, stopped after the 16-stream window to keep it short and
run without the 32-stream window (v0.31.0 lacks vLLM #60945). `results/dgx/ds41-grid-20261105T232843Z`.

| shape | ds41 |
|---|---|
| fixed prompt, 1 stream (greedy, thinking off) | 741.3 tok/s, 5.45 tokens per step, 136 steps/s |
| coding prompt, 1 stream | 803.4 |
| GSM8K, T=1.0, thinking on, 1 stream | 506.4 |
| GSM8K, greedy, thinking off, 1 stream | 593.7 |
| 4 / 8 / 16 streams, aggregate | 1,024 / 1,467 / 2,138 |

## DeepSeek-V4.1-Flash on the GB300 alone, for the root README, 2026-11-06

The root README compares each recipe with its model on the GB300 alone. For DeepSeek that baseline is vLLM's own
DGX Station configuration (the GB300 row of recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash, `v0.32.0`), with the
local changes that made it faster on this box: `baseline/serve-deepseek-gb300-stock.sh`. It serves the official
`-1017` checkpoint, the one Garner's release run used. Measured with Garner's `verify/bench_full.py` at `8043a68`,
unchanged, release protocol (60 s windows, 20 s warm-up), only the shapes the README shows:

```
python3 -I ../deepseek-v4.1-flash/verify/bench_full.py --base-url http://127.0.0.1:8000 --key-file <any file> \
  --out bench-dsv41-gb300-alone.json --skip greedy_c1,coding,prefill,capacity,quality --conc 8,32 --reps 5
```

| shape (tok/s) | GB300 alone | GB300 + RTX (Garner's release run) | speedup |
|---|---|---|---|
| fixed prompt, 1 stream (greedy, thinking off) | 171.1 (3.55 tokens/step, 48 steps/s) | 733.6 (5.43 tokens/step, 135 steps/s) | 4.3x |
| GSM8K, T=1.0, thinking on, 1 stream | 130.4 | 500.5 | 3.8x |
| 8 streams, aggregate | 326 | 1,373 | 4.2x |
| 32 streams, aggregate | 584 (at most 24 run at once) | 2,524 | 4.3x |

Not the same image or flags on both sides, unlike the GLM comparison: each side is its own tuned configuration
(the GB300 alone uses 3 speculative drafts, since drafts past the third rarely pay when every verified token pulls
experts over NVLink-C2C; the RTX recipe uses 5, plus its 29 fast paths). The RTX column is Garner's run on his box;
ds41 on this box measured the same or slightly more (741 / 506 / 1,467 tok/s, the section above). `--max-num-seqs 24`, so 8 of the
32 streams queue. Run `bench-dsv41-gb300-alone-20261106T104157Z` in `results/dgx/`.
