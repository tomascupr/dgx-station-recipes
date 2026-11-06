# GLM-5.3-Flash (native FP8) on a DGX Station GB300 (with an RTX PRO 6000 sidecar)

This recipe turns one NVIDIA DGX Station GB300 plus one RTX PRO 6000 into a private, OpenAI-compatible
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) server, in the model's native FP8 build: **~220-250
tok/s single stream** and **~640 tok/s at 8-32 concurrent streams**, against 75 and 176 for the same model on the
GB300 alone. The checkpoint (306 GiB) is larger than the GB300's memory, so each MoE layer's experts are split, as in
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

| | |
|---|---|
| Model | `zai-org/GLM-5.3-Flash` at `46a050be…`: 320B total / 18B active, 288 routed experts per layer, FP8 weights with 128x128 block scales, KDA + sparse-MLA attention, MTP |
| Endpoint | `http://127.0.0.1:30004/v1` (`/chat/completions`, `/completions`, `/models`), model `glm-5.3-flash` |
| Single stream | 217 tok/s on the fixed prompt, 250 on code, 242 on sampled reasoning (MTP, ~3.6-4.3 tokens per step) |
| Many streams | 556 / 617 / 640 / 633 tok/s aggregate at 4 / 8 / 16 / 32 streams |
| Prefill | ~15.6K tok/s; first token in 1.0 s at 15K tokens, 7.7 s at 119K |
| Context | 512K tokens per request; a 6 GiB KV cache (~680K tokens in all) |
| Quality | GSM8K-200 99.0 %, needles 3/3 at 182K tokens, tool call PASS |

All from `RESULTS.md`, which also has the same model on two B300s (stock vLLM, the reference) and on the GB300 alone.

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
