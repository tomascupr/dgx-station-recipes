# Kimi K3.1 Pro on a DGX Station + RTX PRO 6000: 2-bit experts under stock vLLM

[moonshotai/Kimi-K3.1-Pro](https://huggingface.co/moonshotai/Kimi-K3.1-Pro) has 2.78 trillion parameters and a
1,454 GiB checkpoint. A DGX Station has 251 GiB on the GB300 and 494 GiB of Grace memory; like the other recipes
here, this one adds an RTX PRO 6000 (96 GiB) as a second GPU. It serves the whole model on that box behind an
OpenAI-compatible endpoint (without the RTX: 24.9 tokens/s and more experts dropped, see LIMITATIONS.md):

| | |
|---|---|
| speed, one request | **about 30 tokens/s** (22-31 by prompt) |
| quality against the published model, held-out text | perplexity **2.92 vs 2.95**, same top-1 token **85%** of the time |
| GSM8K-200 with thinking | **98.0%** |
| context | 131,072 tokens, read at ~520 tokens/s |
| features | thinking, tool calls, 4 concurrent requests; text only |

Every number, the run it comes from and the command that reproduces it: [RESULTS.md](RESULTS.md). What it does not
do: [LIMITATIONS.md](LIMITATIONS.md). How it works: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## How it fits

* **Routed experts at 2.125 bits.** They are 1,347 of the checkpoint's 1,454 GiB, published at 4.25 bits (MXFP4).
  `tools/krtn.py` rounds them to four levels with one scale per 64 inputs, without calibration data: 674 GiB.
  Keeping every expert at 2 bits costs far less quality than dropping experts to reach the same size (RESULTS.md,
  Why 2 bits).
* **Everything else in FP8**, by vLLM's own load-time quantization: 107 GiB becomes ~55.
* **Three places for experts**, per layer and by measured usage: the 110 most used on the RTX, the next 200 in the
  GB300's HBM, 538 in pinned Grace memory that the GB300 reads in place over NVLink-C2C. The 48 least used (1.2% of
  calls) are dropped; nothing else is.
* **Stock vLLM does the rest**: attention, KV cache, scheduling, the DSpark draft model, the API. A hook loaded
  through `PYTHONPATH` gives vLLM's Kimi model an expert module that multiplies straight from the packed bytes
  (Triton kernels in `serve/kmoe.py`), and a sidecar process computes the RTX's share through a shared-memory
  channel, as in [`../glm-5.3-flash-fp8/`](../glm-5.3-flash-fp8/). No vLLM source is changed.

## Quick start

```bash
./kimi install     # checks the box, pulls vllm/vllm-openai:kimi-k31, downloads the served files (781 GiB) and the draft
./kimi up          # ready after about 5 minutes on http://127.0.0.1:30005/v1, model kimi-k3.1-pro
./kimi verify
```

The served files are on the Hub: [tomcupr/Kimi-K3.1-Pro-DGX-Station-2bit](https://huggingface.co/tomcupr/Kimi-K3.1-Pro-DGX-Station-2bit).
To make them yourself from Moonshot's checkpoint (1.45 TiB more disk, about 40 minutes after the download):
`./kimi install --from-source`, then `./kimi prepare`. If no single disk has room for the packed experts, give two
directories (`config/kimi.env.example`).

## Layout

| | |
|---|---|
| `kimi`, `config/` | the command and its settings |
| `serve/` | what runs while serving: the vLLM hook, the kernels, the packed format, the sidecar, the usage ranking |
| `tools/` | packing, the expert-free checkpoint, and every check and benchmark RESULTS.md cites |
| `study/` | the measurements that led here (layer-at-a-time passes, quantizer comparisons, placement replays, kernel profiles) and the first engine; run them with `serve/` and `tools/` on `PYTHONPATH` |

Licences and credits: [NOTICE](NOTICE). The model's licence has conditions for large commercial use; read it.
