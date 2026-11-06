# Architecture

## 1. Why the experts are split

GLM-5.3-Flash's FP8 checkpoint is 306 GiB, 284 GiB of it routed experts: 42 MoE layers x 288 experts x 24 MiB, plus
the MTP layer's 288. The GB300 gives vLLM ~243 GiB. Stock vLLM can stream the overflow from Grace memory over
NVLink-C2C (stage 4 in RESULTS.md), but then every step waits on ~200-400 GB/s and single-stream decode runs at 75
tok/s. The RTX PRO 6000 adds 96 GB at ~1.6 TB/s: enough to hold a third of each layer's experts and compute them where
they sit, so only the tokens cross the PCIe link.

## 2. The rowmap: which experts go where

`routing/rowmap-mix-v1-h202.json` keeps 202 experts per layer on the GB300 and puts 86 on the RTX (84.7 GiB, all the
RTX can give to weights). Same count on every layer, as in Garner's rowmap, so each layer's RTX work is alike. The
choice is measured, not uniform: `routing/sitecustomize.py` counted every routed expert id in a copy of the server
while it answered 490 requests of chat, multilingual, reasoning, coding, agentic-coding and tool-call traffic
(`routing/build_mix.py`, `routing/drive.py`), split into decode-sized calls (up to 64 tokens) and prefill. Each layer's
experts are ranked by 0.6 x decode share + 0.4 x prefill share (Garner's weighting) and the 86 lowest go cold. They
receive 17-19 % of routed tokens, against 30 % for a split that ignores routing. The MTP layer stays whole on the
GB300: its five sequential draft passes cannot wait on the RTX.

## 3. Two processes and a channel

The Station's driver lets a process see the GB300 or the RTX, never both (with both visible, CUDA reports one
device). So the RTX runs a sidecar process and the two exchange each MoE layer's work through one file in
`/dev/shm`, registered as pinned memory in both (`peer/channel.py`): the layer's input rows (bf16), its top-k expert
ids and weights in, the cold experts' sum out, and a header with request and response sequence numbers.

On the GB300 the exchange is three Triton launches with no host synchronization, so it records into vLLM's CUDA
graphs: *send* copies the rows, ids and weights into the channel and ends with a system-scope release; *publish*
increments a device-side sequence number and releases it as the request; *wait* spins on the response with a
system-scope acquire. A wait that runs out of spins writes the request number to an error slot instead of returning
partial sums; a watchdog thread in vLLM sees it and stops the process.

The sidecar (`peer/sidecar.py`) loads every layer's cold experts into one stacked set of tensors on the RTX, then
serves requests in order: copy in, run the experts, copy out, publish the response. For the token counts vLLM
captures, each request is one CUDA graph keyed by (layer, token count), captured on first use and sharing one memory
pool; other counts run eagerly. Its loop busy-waits on the request number.

## 4. One MoE layer

`peer/sitecustomize.py` loads into vLLM through `PYTHONPATH`. vLLM loads every routed expert into pinned Grace memory
(UVA offload of `w13_weight` and `w2_weight`); after loading, the hook copies each target layer's 202 hot experts into
HBM, releases the host copy, checks one expert byte-for-byte against the checkpoint (so the kernel layout is the
checkpoint's), and replaces the layer's MoE with one custom op:

1. router: the gate, sigmoid scores with the correction bias, top 8, renormalized, scaled by 2.5;
2. send the rows, ids and weights to the RTX;
3. the hot experts on the GB300 and the shared expert, while the RTX computes the cold ones;
4. wait for the cold sum and add it.

Both GPUs run `peer/peer_moe.py`: vLLM's Triton FP8 block-scale MoE with an expert map (each GPU computes only its own
experts' routes) and GLM's clamped SwiGLU, silu(min(gate, 10)) x clamp(up, -10, 10), which the functional Triton
wrapper does not apply. The Triton tiles are tuned per token count on each GPU, timed as CUDA graphs
(`peer/tune_moe.py`, `peer/tuned/`).

## 5. Where the time goes

At one stream, 6 tokens per step (MTP with 5 drafts): the hot experts take ~193 us per layer (~1 GB at ~5.2 TB/s),
the RTX round trip ~185 us (144 us of kernels), so the two GPUs are balanced; the GB300's other work per layer
(attention, mHC, router) is ~170 us. 42 layers x ~395 us is ~16.6 ms per step. The next gains are in that other work
(Garner's DeepSeek recipe got its last 50 % from ~25 fused kernels there) and, at many streams, in the RTX's bandwidth
(LIMITATIONS.md, 1).
