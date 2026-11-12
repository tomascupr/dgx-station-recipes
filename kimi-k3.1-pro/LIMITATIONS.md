# Limitations

1. **It is a quantized model.** Perplexity on held-out text matches the published model, but it picks a different
   top-1 token about one time in seven. Task checks so far: GSM8K-200 (98.0%) and a tool call; the published model
   could not be run here for a head-to-head.
2. **Speed depends on the prompt**: 22-31 tokens/s for one request. The draft model's guesses are accepted less often
   on some text (64% of first guesses on average).
3. **Long prompts are slow**: ~520 prompt tokens per second, so 36K tokens take 70 s to the first token. vLLM's
   prefix cache helps on later turns of the same conversation.
4. **Four concurrent requests**, each slower: about 20 tokens/s with two, ~12 with four.
5. **Text only.** The vision tower is not loaded (`--language-model-only`).
6. **Memory is tight.** While serving: GB300 232-238 of 251 GiB, RTX 86-90 of 96 GiB, Grace ~453 of 494 GiB, 404 of
   them pinned. Nothing else of size can run on the box.
7. **Without the RTX** (`KIMI_PEER=0 KIMI_PRUNE=158` in `config/recipe.env`): 24.9 tokens/s, and 158 experts per
   layer dropped instead of 48; quality in that setting was not measured.
8. **One vLLM build.** The hook replaces `FusedMoEFactory` inside `vllm.models.kimi_k3.nvidia.model` and
   `KimiMoE.forward`; it stops with one line if the model is not what it expects, and another vLLM build may need
   changes. vLLM's `latency_specialized` decode backend bypasses the hook and is not supported.
9. **vLLM's kernel autotune is off** (`--disable-flashinfer-autotune`): with experts outside HBM it ran for over an
   hour without finishing.
10. **Start takes ~5 minutes**, 2 of them copying 404 GiB into pinned memory.
