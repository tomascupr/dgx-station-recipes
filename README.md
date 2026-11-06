# DGX Station recipes

Serving recipes for an NVIDIA **DGX Station GB300**, each one a tested, one-command way to run a model on it and
get a private, OpenAI-compatible endpoint. Every recipe carries its own CLI, docs, measured results and licence
notices, so you can copy a recipe's directory on its own. Both recipes need one extra GPU, an **RTX PRO 6000
(96 GB)**: neither model fits the GB300's memory, so each MoE layer's least-used experts live on the RTX, which
computes them in a sidecar process.

Output tokens per second, the same shapes for both recipes (footnotes):

| recipe | model | code, 1 stream ¹ | reasoning, 1 stream ² | 8 streams ² | 32 streams ² | vs. the GB300 alone ³ | context |
|---|---|---|---|---|---|---|---|
| [`deepseek-v4.1-flash/`](deepseek-v4.1-flash/) | DeepSeek-V4.1-Flash-1017 | 734 | 501 | 1,373 | 2,524 | not measured | 512K per request, 11.4M-token KV cache |
| [`glm-5.3-flash-fp8/`](glm-5.3-flash-fp8/) | GLM-5.3-Flash (native FP8) | 217 | 242 | 617 | 633 | **2.9x** at 1 stream, **3.6x** at 32 | 512K per request, ~680K-token KV cache |

1. The same fixed coding prompt (the coding task of Fraser Price's BRRR bench, ~2.6K tokens), greedy, 2,048 tokens
   out. DeepSeek runs it with thinking off; GLM-5.3-Flash cannot turn thinking off and runs it at reasoning effort
   low.
2. GSM8K questions at temperature 1.0 with thinking on (GLM: reasoning effort max), each stream asking the next
   question when one ends; aggregate over all streams.
3. The same model and benchmark on the GB300 alone, streaming the experts that do not fit from Grace memory:
   GLM-5.3-Flash does 75 tok/s on the code shape and 176 at 32 streams there. Not measured for DeepSeek.

Each recipe's `RESULTS.md` has every shape, the run it comes from and the command that reproduces it
(`verify/bench_full.py`). The DeepSeek numbers are Garner McCloud's release run (60 s windows); the GLM numbers
are from a quicker run of the same suite (30 s windows). The two models differ: DeepSeek's experts are 4-bit
(MXFP4) and GLM's 8-bit (FP8), so each GLM token reads about 1.8x the expert bytes (8 x 25.2 MB against 6 x
18.8 MB), and the DeepSeek recipe adds 29 fast-path components.

## Quick start

```bash
git clone https://github.com/tomascupr/dgx-station-recipes.git
cd dgx-station-recipes/glm-5.3-flash-fp8      # or deepseek-v4.1-flash, with ./ds41 for ./glm53
./glm53 install
./glm53 up
```

Then read that recipe's `README.md`.

## Licence

MIT (see `LICENSE`), copyright 2026 Garner McCloud. Each recipe's `NOTICE` lists the third-party code and data it
includes and their licences. `glm-5.3-flash-fp8/` follows the design of Garner's `deepseek-v4.1-flash/`.
