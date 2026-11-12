# Results

One DGX Station: GB300 (251 GiB), RTX PRO 6000 (96 GiB), 72-core Grace with 494 GiB. Image
`vllm/vllm-openai:kimi-k31` (vLLM 0.1.dev24216+g266e88c05). Model `moonshotai/Kimi-K3.1-Pro` @ `d9e99b07`.
Configuration: `config/recipe.env`. Measured 2026-11-11/12. Results repeat exactly across restarts (greedy outputs
and logprobs are identical), so single runs are reported where a run is long.

## Speed

`./kimi verify` (its first three parts: `tools/vbench.py --runs 6 --max-tokens 250`, `tools/kchecks.py long`,
`tools/kchecks.py par`). vbench streams three prompts in turn (a data-structure explanation, a coding task, a
300-word summary), greedy, and reports tokens per second after the first token.

| shape | result |
|---|---|
| one request, 250 tokens | **median 29.9 tokens/s** over 9 runs (30.4 on another start); by prompt 28.7 / 30.7 / 22.5 |
| first token, ~105-token prompt | 0.7-0.8 s |
| 2 requests at once | 19-21 tokens/s each, 37-41 in total |
| 4 requests at once | ~12 tokens/s each |
| 12,099-token prompt | first token after 23.0 s (527 prompt tokens/s) |
| 36,099-token prompt | first token after 70.2 s (514 prompt tokens/s) |

Draft model (DSpark, `KIMI_DRAFT_TOKENS`), same benchmark: no draft 26.6, 1 token 29.7, **2 tokens 29.9-30.4**,
3 tokens 27.5, 5 tokens 18.3 (measured before two later speedups). First guesses are accepted 64% of the time,
second 37%, so a step yields 2.07 tokens.

Without the sidecar (`KIMI_PEER=0 KIMI_PRUNE=158`): 24.9 tokens/s, with 158 experts per layer dropped instead of 48.

## Quality

**Held-out text against the published model.** `tools/kgate.py /native NATIVE_PASS_DIR` sends the 96 held-out
sequences (1,024 tokens each, mixed chat, reasoning, coding and tool-call traffic) to the server with
`prompt_logprobs` and compares with the published model's own outputs on the same tokens (from `tools/kstream.py`,
which runs the published checkpoint one layer at a time). Over the 52,891 tokens inside assistant messages:

| | perplexity | same top-1 token as published |
|---|---|---|
| published model (kstream.py) | 2.954 | - |
| **this recipe** | **2.924** | **85.0%** |
| BF16 non-expert weights, 96 experts dropped per layer | 2.958 | 84.9% |

**Against stock vLLM on identical weights.** `tools/kab.py` writes the first three MoE layers' experts, rounded to
2 bits with power-of-two scales, twice: packed, and in the published MXFP4 layout (such weights are exact in both).
Stock vLLM on the one and the hook with all three tiers on the other (`tools/kcompare.py`): the first 11 greedy
tokens are identical, top-5 sets agree, logprobs differ by at most 0.14 (stock vLLM quantizes the experts' inputs
to 8 bits, the hook does not), and the paths part at a near tie.

**Decode against prefill.** `tools/kchecks.py decode`: 200 greedy tokens, then the same tokens scored in one
prefill pass. Median logprob difference 0.03 (mean 0.12); the prefill pass ranks the generated token first for 179
of 200. The same figures appear with the prefill kernel forced for every call (`KIMI_PACK_GROUPED=1`: 0.03 / 0.10 /
190) and without the draft model (0.02 / 0.10 / 190), so the difference is vLLM's own between its decode and
prefill paths (its FP8 KV cache, which this model always uses), not the hook's.

**At start**, every time: the first MoE layer as served against the reference formulas on float32 experts,
relative difference 2-4e-4 (the `KIMI_PACK selfcheck` line in `kimi logs`).

**GSM8K-200** (the 200 questions and scorer of `../glm-5.3-flash-fp8/verify/gsm8k.py`, thinking on, temperature 0.3, 4
at once): **196 of 200, 98.0%**; every answer ended normally, median 264 output tokens, 23 minutes in all.

**Tool call** (`../glm-5.3-flash-fp8/verify/tool.py`): one function call with arguments, parsed by vLLM's `kimi_k3`
parser: pass.

## Why 2 bits

Measured before any serving code, with `tools/kstream.py` (the published model, one layer at a time, with variants
of the experts) and `tools/metrics.py`, on the same held-out half; experts ranked on the other half.

| experts | size | perplexity | KL from published | same top-1 |
|---|---|---|---|---|
| published (4.25 bits) | 1,347 GiB | 2.95 | - | - |
| **all kept, 2 bits** | 713 GiB | 3.04 (all rows; published 3.04) | 0.106 | 88.1% |
| 60% kept, 4.25 bits | 808 GiB | 3.29 | 0.178 | 87.2% |
| 50% kept, 4.25 bits | 674 GiB | 3.49 | 0.258 | 84.3% |
| 30% kept, 4.25 bits | 404 GiB | 4.31 | 0.543 | 76.4% |
| 50% kept, 2 bits | 357 GiB | 3.70 | 0.380 | 79.5% |

At equal size, keeping every expert at 2 bits loses less than half of what dropping half of them loses. Output-aware
rounding (GPTQ, `study/kgptq.py`) improved single experts but not the model (KL 0.104 against 0.106), so the recipe
rounds without data. FP8 for the non-expert weights alone: KL 0.010.

## The road here

| | tokens/s | |
|---|---|---|
| own engine (`study/kserve.py`: reference modeling code, LRU expert cache, experts memory-mapped) | 1.5 | 2,048-token context |
| stock vLLM, published weights with 75% of experts removed so that it loads | 13.6 | output unusable |
| this recipe, first full run (BF16 non-expert, tensor-core kernel only) | 17.3 | |
| + multiply-add decode kernel, aligned address table | 20.3 | |
| + FP8 non-expert weights, 200 instead of 130 experts per layer in HBM | 25.1 | |
| + draft model, 2 tokens | 28.7 | |
| + the RTX holds the most used experts instead of the tier below HBM | 30.3 | |

vLLM's own work per token, measured with the routed experts switched off (`KIMI_PACK_NOOP=1`): 17.5 ms, i.e. 57
tokens/s is what this vLLM build would do if experts were free.

## What did not help

* **A recency cache in HBM** for Grace experts (copies made inside the CUDA graph, least recently used out). A
  replay of the routing trace (`study/tiersim5.py`) promised half the Grace traffic; served, it was 26.9 against
  29.9 tokens/s. A copy has to finish before the arithmetic starts, while a direct read overlaps with it.
* **Swapping** experts between HBM and Grace memory: writes share the link with reads, 158 GiB/s each way against
  325 one way (`study/kswap_proto.py`).
* **Lookup tables** instead of arithmetic for the 2-bit codes: within 25% of the arithmetic kernel.
* **Longer drafts**: see Speed.
* **vLLM's kernel autotune**: over an hour without finishing when experts are outside HBM.
