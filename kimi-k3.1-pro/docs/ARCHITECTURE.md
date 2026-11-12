# Architecture

## 1. The budget

Kimi K3.1 Pro has 93 layers; 92 of them hold 896 routed experts each, and a token uses 16 per layer. An expert is
three matrices (3,072 x 3,584 twice, 3,584 x 3,072 once): 33 million weights, 17.5 MB as published (MXFP4).

| part | published | served |
|---|---|---|
| routed experts | 1,347 GiB at 4.25 bits | 674 GiB at 2.125 bits (8,773,632 bytes each) |
| everything else (attention, shared experts, projections, embeddings) | 107 GiB BF16 | ~55 GiB, FP8 by vLLM at load time |

The box holds experts in three places. Per layer, ranked by how often a pass of the published model over mixed
traffic used each expert (`serve/rank-native.json`, from `tools/kplace.py`):

| rank | where | experts | size | share of calls |
|---|---|---|---|---|
| 1-110 | RTX PRO 6000 (sidecar) | 110 | 83 GiB | 34% |
| 111-310 | GB300 HBM | 200 | 150 GiB | 27% |
| 311-848 | Grace memory, pinned | 538 | 404 GiB | 38% |
| 849-896 | dropped | 48 | - | 1.2% |

The router's bias for a dropped expert is set so it is never chosen; its calls go to the token's next choices.

## 2. The packed format (`serve/kpack.py`, `tools/krtn.py`)

A matrix is stored as 2-bit codes, four per byte, and one scale byte per 64 inputs of each output row: weight =
(2 code - 3) x 2^((scale - 200) / 8), i.e. the levels -3a, -a, +a, +3a. `krtn.py` picks, per group, the scale that
minimises the squared rounding error; no calibration data. An expert is one contiguous block of bytes, a layer one
safetensors file `[896, 8773632]`, so any process can memory-map it and take single experts.

## 3. Inside vLLM (`serve/hook/sitecustomize.py`)

vLLM loads a checkpoint without routed experts (`tools/kstrip.py`). The hook, found through `PYTHONPATH`:

1. replaces `FusedMoEFactory` while `KimiMoE` is built, so vLLM never allocates 1.3 TiB of expert tensors;
2. gives `KimiMoE` the forward of vLLM's own "routed experts as one call" branch: its down-projection, gate and
   `fused_grouped_topk`, then the packed experts, then its shared experts, latent norm and up-projection;
3. after loading, places the experts: HBM tier into one tensor per layer, Grace tier into anonymous memory registered
   with CUDA (PyTorch's pinned allocator would round each layer up to 8 GiB), and builds one address table per layer;
4. checks itself: the first MoE layer, as it will run, against the reference model's formulas on experts unpacked to
   float32 from the file.

Everything the layer launches records into vLLM's CUDA graphs.

## 4. The kernels (`serve/kmoe.py`)

One launch reads an expert wherever it lives: `table[expert]` is the distance from the layer's HBM tensor to the
expert's bytes, and pinned Grace memory is in the GPU's address space, so the same pointer arithmetic reaches both.
Entries count 16-byte units: Triton merges byte loads into wide ones only when it can prove the address aligned,
and "entry x 16" proves it where a loaded number does not (4x on this kernel).

* **Decode** (up to 16 tokens): one program per route and 16 output rows, plain multiply-adds on tiles unpacked in
  registers. Tensor cores lose here: with one row per expert, moving operands into their layout costs more than
  the multiply.
* **Prefill**: routes grouped by expert (vLLM's `moe_align_block_size`), 16 routes x 32 rows per program on tensor
  cores, so an expert's bytes are read once per 16 tokens.
* SiTU, the activation between the two multiplies, is one more kernel; a long prompt has 131,072 routes there.

## 5. The sidecar (`serve/ksidecar.py`, `serve/hook/kchannel.py`)

The Station's driver shows a process the GB300 or the RTX, never both. The RTX runs a second process holding its 110
experts per layer; the two exchange each layer's latent rows, top-k ids and weights, and the RTX's sum, through one
file in `/dev/shm` registered as pinned memory in both. On the GB300 the exchange is three kernel launches that
record into CUDA graphs (send, publish, wait); a watchdog stops vLLM if the sidecar stops answering. The design and
the channel are those of `../glm-5.3-flash-fp8/`, which took the idea from Garner McCloud's DeepSeek recipe.

The RTX holds the most used experts, not the tier below HBM: the GB300 is the busier GPU (it also reads the Grace
tier), and the RTX finishes its third of the calls inside the GB300's own time for the layer.

## 6. Where a token's time goes

One request, 2 draft tokens per step, about 2.1 tokens accepted per step, 72 ms per step. Without routed experts
(`KIMI_PACK_NOOP=1`, wrong output) vLLM's own work is 17.5 ms per single-token step. The rest is the experts, and
there the limit is the link: 38% of calls read 8.8 MB each from Grace memory at ~320 GiB/s, which overlaps with the
HBM tier's arithmetic but not with itself. That is also why longer drafts do not pay (each extra token brings its
own 16 experts per layer), and why a recency cache in HBM did not either (RESULTS.md, What did not help).
