# Limitations

What this recipe does not do, so you do not spend a day finding out.

## 1. Many streams stop at ~640 tok/s

From about 8 concurrent streams on, almost every cold expert is used by some token in every step, so the RTX reads
nearly all of its 85 GB of experts per step at ~1.6 TB/s: a floor of ~50 ms per step. That is the FP8 checkpoint
against the RTX's bandwidth, not overhead (RESULTS.md, stage 5). Two B300s with stock vLLM reach 3,481 tok/s at 32
streams. 4-bit cold experts would halve the RTX's bytes but change the numerics; not done.

## 2. A 6 GiB KV cache: ~680K tokens in all

With 202 hot experts per layer the GB300 holds ~221 GiB of weights, which leaves room for a 6 GiB KV cache and small
CUDA graphs (capture sizes up to 192 tokens). One request can use 512K tokens; many long contexts at once will queue.
Garner's capacity check (16 x 110K + a 500K needle, ~2.26M tokens resident) does not fit and is not run.

## 3. Not bit-identical to stock vLLM

Experts run on vLLM's Triton FP8 kernel on both GPUs instead of the stock TRT-LLM kernel, the hot and cold sums are
added in a different order, and activations are quantized to FP8 per 128 values as in stock. The split matches all
288 experts computed in one place to ~3e-3 relative; quality is checked by GSM8K-200, needles and the tool call
(RESULTS.md), not by bit equality. Temperature 0 is not reproducible across servers or batch shapes.

## 4. This box, this checkpoint, this image

* GB300 + RTX PRO 6000 on aarch64 only. The rowmap, the tuned Triton tiles (`peer/tuned/`) and the memory budget are
  for these two GPUs and this checkpoint revision; another revision needs the byte constants in `routing/rowmap.py`
  and the rowmap re-measured.
* The image is a vLLM nightly (`a2bcb565`, pinned by digest): the build the B300 reference ran, and the first with
  GLM-5.3's native chat rendering (vLLM #62424), which v0.33.0 lacks. The hook relies on vLLM's internals (the V2
  model runner, `Glm5NextMoE`, the fused-MoE helpers); another image needs `glm53 verify` and probably changes.
* The Station's driver gives a process either the GB300 or the RTX, never both, so the peer tier is two processes and
  two containers.

## 5. Text only, as measured

The vision encoder loads and runs, but image and video requests were never measured on this split, and vLLM's Rust
frontend serves text only (use `VLLM_USE_RUST_FRONTEND=0` for multimodal, untested here).

## 6. Token counts outside the captured sizes are slower

The sidecar replays a CUDA graph per (layer, token count) for the sizes vLLM captures (up to 192 tokens); other
counts (prefill chunks) run through Python on the RTX. A different MTP draft count changes the token counts per step:
with 4 drafts every step missed the graphs and ran at 60 % of the speed (37 against 60 steps/s). Keep `GLM53_MTP_TOKENS` at 5, or add the
new counts to `GLM53_CAPTURE_SIZES` and `peer/sidecar.py`'s `GRAPH_M`.

## 7. First start compiles

The first `up` compiles kernels and tunes FlashInfer (11 min for the GB300-alone server on a cold cache; the peer
tier's cold start was not timed); later starts reuse `~/.glm53/cache` and took ~3.5 min.
