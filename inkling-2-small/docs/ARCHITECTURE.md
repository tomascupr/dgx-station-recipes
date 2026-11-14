# Architecture

48 MoE layers, 384 experts, top-8, no shared experts, hidden 4608, intermediate
2304. Layers 0–39 are NVFP4 (packed U8 weights, E4M3 scales, per-expert amax,
one fp32 denominator per w13 and per w2). Layers 40–47 are MXFP8 (E4M3
weights, U8 UE8M0 scales, no amax). Experts are packed along axis 0 of one
tensor per projection, not one file per expert. w13 rows are interleaved
gate, up, gate, up. The activation is

```
4 * tanh(gate / 4) * sigmoid(gate) * 25 * tanh(up / 25) * route_weight
```

The router scores `sigmoid(logits) + bias`, takes a stable top-8 (lower id
wins a tie), and the weights are `softmax(logsigmoid(chosen logits)) * 8`.

Full-attention layer ids, from `text_config.local_layer_ids`, are 5, 11, 17,
23, 29, 35, 41, 47. The MTP file is five dense BF16 layers plus a Markov
head, not MoE. Attention, short convs and block residuals stay on the GB300
and stay inside the day-0 image. This recipe does not reimplement them.

## Placement

`placement/plan_experts.py` reads the safetensors index and the JSON header of
each shard (8-byte length, then a Range request). It does not read weights.

Default budgets are 199.425 GiB of experts on the GB300 and 87 GiB on the
RTX. That is every MXFP8 expert on the GB300 and 158 / 130 / 96 NVFP4 ids on
GB300 / RTX / Grace. A profile JSON (`routing/profile.py`) reorders the NVFP4
ids by hit count. It cannot move an MXFP8 expert off the GB300; the planner
exits if the MXFP8 experts themselves do not fit.

`placement/rows.py` returns the byte range of one expert inside a packed
tensor. The GB300 loader's `coldskip` is that expert id not being in `hot`.

## Peer

`peer/segment.py` is one request in flight in `/dev/shm/inkling2-peer`
(1,191,936 bytes at hidden 4608, 64 tokens, top-8). Hot experts call the
sm_103a GEMM in-process. Cold experts cross the segment to the sm_120a GEMM
in the sidecar. Spill experts stay on the CPU: the plan's Grace tier is a
host dequant into the fp64 reference, not a host-mapped device GEMM and not
a third GPU. None of the three run unless `INKLING2_ALLOW_UNVALIDATED=1`.

## Where the 14 Nov plan is off

Checked against the Hub revision above (config, `hf_quant_config.json`,
index, and Range headers for shard 1, shard 15 and `mtp.safetensors`), and
against vLLM PR 67325 at `d664cd1e`.

* **158 is not what 200.81 GiB divides to.** After reserving all MXFP8
  (100,902,371,328 bytes), one NVFP4 id across 40 layers is 716,636,480
  bytes. `round(200.81 GiB)` fits **160** ids. 158 is the count whose byte
  total is 199.425 GiB, which is inside the plan's 200–201 GiB band. The
  recipe defaults to 158, and `./inkling2 test` locks both numbers.
* **MXFP8 scales are stored per row, 32 along K**, shapes `(384, 4608, 144)`
  and `(384, 4608, 72)`. `hf_quant_config.json` says `weight_block_shape`
  `[32, 32]` for that group. A 32×32 block would be `(384, 144, 144)` and
  `(384, 144, 72)`. The dequant follows the stored shape. vLLM's own test
  dequant also repeats each scale 32 times along K.
* **w13 is interleaved**, not a concatenated `[gate; up]` slab. The plan's
  shapes are right; the row order is not stated.
* **`config.json` says `quant_algo: NVFP4` for the whole model.** The split
  is only in `hf_quant_config.json` (`quant_method: inkling2`): 80 NVFP4
  expert tensors and 16 MXFP8 expert tensors. Headers agree: amax and
  denominator exist on layers 0–39 only.
* **The day-0 class is `vllm.models.inkling2.moe.Inkling2MoE`.** It asserts
  `n_shared_experts == 0`. `vllm.models.inkling.nvidia.moe.InklingMoE` asserts
  the opposite and is not this checkpoint. There is no DeepSeek MegaMoE in
  this path, and `route()` does not take an external expert map, which is why
  the hook replaces `forward` instead of patching a weight loader the way
  DeepSeek does.
* **Gate and activation** are the formulas above, not a plain softmax and a
  plain SwiGLU. Weight decode scale is `amax / denominator`. The layer-0
  denominator is **2688.0** (four bytes read from shard 1). The activation
  side of the vLLM row scale uses **1536**, which is not in the plan.
* **1,854 tensors** is 1,770 in `model.safetensors.index.json` plus 84 in
  `mtp.safetensors`. `metadata.total_size` is 390,056,448,320. Both match the
  plan. Full-attention ids match. Per-expert byte arithmetic matches.

A header-only `./inkling2 plan --from-hf` against this revision returned
1,854 tensors, expert payload **376,090,779,968** bytes (the plan's figure,
including the 320 bytes of per-layer denominators) and other bytes
**17,389,089,120**. Those three match the plan. The CPU unit tests use
synthetic headers of the same shapes and do not download shards.
