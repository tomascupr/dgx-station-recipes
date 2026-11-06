# DGX Station recipes

Serving recipes for an NVIDIA **DGX Station GB300**, each one a tested, one-command way to run a model on it and
get a private, OpenAI-compatible endpoint:

- **[`glm-5.3-flash-fp8/`](glm-5.3-flash-fp8/), GLM-5.3-Flash (native FP8): by Tomas Cupr, new in this
  repository.** It takes the design of the DeepSeek recipe below to a different model, with its own expert
  routing, sidecar and kernels.
- **[`deepseek-v4.1-flash/`](deepseek-v4.1-flash/), DeepSeek-V4.1-Flash-1017: by Garner McCloud**, unchanged from
  [his repository](https://github.com/garnermccloud/dgx-station-recipes), which this one builds on (its history
  starts with his two commits).

Every recipe carries its own CLI, docs, measured results and licence notices, so you can copy a recipe's directory
on its own. Both add one GPU, an **RTX PRO 6000 (96 GB)**: neither model fits the GB300's memory, so each MoE
layer's least-used experts live on the RTX, which computes them in a sidecar process. Without it, the GB300 reads
those experts from Grace memory on every step; with it, both models run **3-4x faster** on the same box.

Output tokens per second, with and without the RTX, the same shapes for both models (footnotes):

| model | setup | code, 1 stream ¹ | reasoning, 1 stream ² | 8 streams ² | 32 streams ² |
|---|---|---|---|---|---|
| GLM-5.3-Flash (native FP8) | GB300 alone ³ | 75 | 82 | 147 | 176 |
| | GB300 + RTX PRO 6000: [`glm-5.3-flash-fp8/`](glm-5.3-flash-fp8/), by Tomas Cupr | **217** | **242** | **617** | **633** |
| | **speedup** | **2.9x** | **3.0x** | **4.2x** | **3.6x** |
| DeepSeek-V4.1-Flash-1017 | GB300 alone ⁴ | 171 | 130 | 326 | 584 |
| | GB300 + RTX PRO 6000: [`deepseek-v4.1-flash/`](deepseek-v4.1-flash/), by Garner McCloud | **734** | **501** | **1,373** | **2,524** |
| | **speedup** | **4.3x** | **3.8x** | **4.2x** | **4.3x** |

1. The same fixed coding prompt (the coding task of Fraser Price's BRRR bench, ~2.6K tokens), greedy, 2,048 tokens
   out. DeepSeek runs it with thinking off; GLM-5.3-Flash cannot turn thinking off and runs it at reasoning effort
   low.
2. GSM8K questions at temperature 1.0 with thinking on (GLM: reasoning effort max), each stream asking the next
   question when one ends; aggregate over all streams.
3. Stock vLLM on the GB300 with the RTX unused: the experts that do not fit in HBM stay in Grace memory and are read
   over NVLink-C2C on every step (`glm-5.3-flash-fp8/baseline/serve-glm53-gb300-stock.sh`). Same box, image, model
   flags and benchmark as the recipe row below it.
4. vLLM's own DGX Station configuration for this model (recipes.vllm.ai, image v0.32.0) on the GB300 with the RTX
   unused, experts in Grace memory as in 3. It is tuned for the GB300 alone (3 speculative drafts instead of 5) and
   runs at most 24 requests at once (`glm-5.3-flash-fp8/baseline/serve-deepseek-gb300-stock.sh`). Measured for
   this repository on 2026-11-06 with Garner's benchmark, unchanged. The recipe row below it is Garner's release
   run; on the Station this baseline ran on, his recipe measured the same or slightly more.

Both serve 512K tokens per request; the KV cache holds 11.4M tokens for DeepSeek and ~680K for GLM. Each recipe's
`RESULTS.md` has every shape, the run it comes from and the command that reproduces it (`verify/bench_full.py`);
the DeepSeek GB300-alone run is in `glm-5.3-flash-fp8/RESULTS.md`, since `deepseek-v4.1-flash/` is kept as Garner
published it. Every run uses 60 s windows except GLM's, which come from a quicker run of the same suite (30 s).
DeepSeek is faster than GLM with the RTX for two reasons: its experts are 4-bit (MXFP4) and GLM's 8-bit (FP8), so
each GLM token reads about 1.8x the expert bytes (8 x 25.2 MB against 6 x 18.8 MB), and the DeepSeek recipe adds
29 fast-path components.

## Quick start

```bash
git clone https://github.com/tomascupr/dgx-station-recipes.git
cd dgx-station-recipes/glm-5.3-flash-fp8      # or deepseek-v4.1-flash, with ./ds41 for ./glm53
./glm53 install
./glm53 up
```

Then read that recipe's `README.md`.

## Licence

MIT (see `LICENSE`), copyright 2026 Garner McCloud; `glm-5.3-flash-fp8/` copyright 2026 Tomas Cupr, apart from the
files its `NOTICE` credits to others. Each recipe's `NOTICE` lists the third-party code and data it includes and
their licences. `glm-5.3-flash-fp8/` follows the design of Garner's `deepseek-v4.1-flash/`.
