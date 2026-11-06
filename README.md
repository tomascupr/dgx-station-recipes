# DGX Station recipes

Serving recipes for an NVIDIA **DGX Station GB300**, each one a tested, one-command way to run a model on it and
get a private, OpenAI-compatible endpoint. Every recipe carries its own CLI, docs, measured results and licence
notices, so you can copy a recipe's directory on its own. Both recipes add one GPU, an **RTX PRO 6000 (96 GB)**:
neither model fits the GB300's memory, so each MoE layer's least-used experts live on the RTX, which computes them
in a sidecar process. Without it, the GB300 reads those experts from Grace memory on every step; with it,
GLM-5.3-Flash runs **3-4x faster** on the same box.

Output tokens per second, with and without the RTX, the same shapes for both models (footnotes):

| model | setup | code, 1 stream ¹ | reasoning, 1 stream ² | 8 streams ² | 32 streams ² |
|---|---|---|---|---|---|
| GLM-5.3-Flash (native FP8) | GB300 alone ³ | 75 | 82 | 147 | 176 |
| | GB300 + RTX PRO 6000, [`glm-5.3-flash-fp8/`](glm-5.3-flash-fp8/) | **217** | **242** | **617** | **633** |
| | **speedup from the RTX** | **2.9x** | **3.0x** | **4.2x** | **3.6x** |
| DeepSeek-V4.1-Flash-1017 | GB300 alone ³ | — | — | — | — |
| | GB300 + RTX PRO 6000, [`deepseek-v4.1-flash/`](deepseek-v4.1-flash/) | **734** | **501** | **1,373** | **2,524** |

1. The same fixed coding prompt (the coding task of Fraser Price's BRRR bench, ~2.6K tokens), greedy, 2,048 tokens
   out. DeepSeek runs it with thinking off; GLM-5.3-Flash cannot turn thinking off and runs it at reasoning effort
   low.
2. GSM8K questions at temperature 1.0 with thinking on (GLM: reasoning effort max), each stream asking the next
   question when one ends; aggregate over all streams.
3. Stock vLLM on the GB300 with the RTX unused: the experts that do not fit in HBM stay in Grace memory and are read
   over NVLink-C2C on every step (`glm-5.3-flash-fp8/baseline/serve-glm53-gb300-stock.sh`). Same image, model flags
   and benchmark as the row below it. Not measured for DeepSeek-V4.1-Flash, so it has no speedup row.

Both serve 512K tokens per request; the KV cache holds 11.4M tokens for DeepSeek and ~680K for GLM. Each recipe's
`RESULTS.md` has every shape, the run it comes from and the command that reproduces it (`verify/bench_full.py`).
The DeepSeek numbers are Garner McCloud's release run (60 s windows); the GLM numbers are from a quicker run of the
same suite (30 s windows). DeepSeek is faster with the RTX for two reasons: its experts are 4-bit (MXFP4) and
GLM's 8-bit (FP8), so each GLM token reads about 1.8x the expert bytes (8 x 25.2 MB against 6 x 18.8 MB), and the
DeepSeek recipe adds 29 fast-path components.

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
