# GLM-5.3-Flash (native FP8) on a DGX Station GB300 (with an RTX PRO 6000 sidecar)

By Tomas Cupr, following the design of Garner McCloud's [`deepseek-v4.1-flash`](../deepseek-v4.1-flash/) recipe.

This recipe turns one NVIDIA DGX Station GB300 plus one RTX PRO 6000 into a private, OpenAI-compatible
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) server, in the model's native FP8 build, **3-4x faster
than the same model on the GB300 alone**: 217 against 75 tok/s on one stream, 640 against 173 at 16 streams (table
below). The checkpoint (306 GiB) is larger than the GB300's memory, so each MoE layer's experts are split, as in
Garner McCloud's [`deepseek-v4.1-flash`](../deepseek-v4.1-flash/) recipe whose design this follows: the 202 experts
real traffic uses most stay on the GB300, the other 86 live on the RTX, which computes them in a sidecar process.

## Quick start

```bash
cd dgx-station-recipes/glm-5.3-flash-fp8
./glm53 install     # checks the box, pulls the image, downloads the weights (306 GiB) or checks the ones you have
./glm53 up          # the first start compiles kernels (11 min for the GB300-alone server); later ~4 min
./glm53 status
```

Run as root (or a member of the `docker` group; root also lets `up` drop the page cache first). If the weights are
already on the box, put `GLM53_MODEL_DIR=/path/to/GLM-5.3-Flash` in `~/.glm53/config.env` before `install`.

```bash
curl -s http://127.0.0.1:30004/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Explain a KV cache in two sentences."}],
       "max_tokens":1024,"reasoning_effort":"low"}'
```

The server listens on `127.0.0.1` and has no API key: put a proxy with authentication in front of it before you
expose it. GLM-5.3-Flash always reasons first; `reasoning_effort` `low` / `high` / `max` (default `max`) sets how much,
either top-level or in `chat_template_kwargs`, and the reasoning arrives in the `reasoning` field.

## What you get

An endpoint at `http://127.0.0.1:30004/v1` (`/chat/completions`, `/completions`, `/models`), model `glm-5.3-flash`:
`zai-org/GLM-5.3-Flash` at `46a050be…` (320B total / 18B active, FP8 with 128x128 block scales, KDA + sparse-MLA
attention, MTP), 512K tokens per request, a 6 GiB KV cache (~680K tokens in all). Needles pass 3/3 at 182K tokens
and the tool call passes.

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
used 30 s windows and 3 runs per shape, the B300 runs 60 s and 10: the same shapes, fewer samples.

## Requirements

* **GB300** (DGX Station) and an **RTX PRO 6000 Blackwell** (96 GB): the 86 cold experts per layer, 84.7 GiB, live on
  the RTX. The model does not fit the GB300 alone; `install` refuses without the RTX.
* **Host memory: ~450 GiB.** vLLM stages every routed expert (284 GiB) in pinned Grace memory while it loads, then
  moves the hot ones into HBM.
* **Disk:** 306 GiB for the weights, ~11 GB for the image, a few GB of kernel caches in `~/.glm53/cache`.
* Linux **aarch64**, Docker with the NVIDIA Container Toolkit, and a **CDI spec** for the RTX
  (`nvidia-ctk cdi generate`): the RTX is passed by CDI name. The image (`vllm/vllm-openai:nightly-a2bcb565…`, pinned
  by digest) brings CUDA, vLLM and the kernels; the host needs nothing else.

## Commands

| command | what it does |
|---|---|
| `./glm53 install` | checks the box, pulls the image, downloads or checks the weights; safe to re-run |
| `./glm53 up` / `down` | start / stop both containers (`glm53-sidecar` on the RTX, `glm53` on the GB300) |
| `./glm53 status` | health, endpoint, containers, GPU memory, a sample curl |
| `./glm53 verify [--full]` | preflight and boot gate, tool call, needles at 200K, GSM8K-200, decode and coding against `verify/published.json` |
| `./glm53 bench [args]` | the metrics suite (`verify/bench_full.py`); results in `~/.glm53/bench/` |
| `./glm53 logs [glm53\|glm53-sidecar]` | follow a container's log |

## Going deeper

* `docs/ARCHITECTURE.md`: the split, the sidecar and the channel between the GPUs, and how the rowmap was chosen.
* `LIMITATIONS.md`: what this does not do (many-stream ceiling, KV size, determinism, other hardware).
* `RESULTS.md`: every measurement, the command behind it and what was tried and dropped.
* `NOTICE`: the model's licence, Garner's design, and the data the checks use.

## Licence

The recipe is MIT (`../LICENSE`); the weights are Z.AI's, MIT, downloaded from the official release and not
redistributed. See `NOTICE`.
