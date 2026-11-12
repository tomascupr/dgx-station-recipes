#!/usr/bin/env python3
"""Serve Kimi K3.1 Pro on a DGX Station from 2-bit packed experts (kgptq.py --save), text only.

    kserve.py MODEL_DIR --ref REF_DIR --experts DIR[,DIR2] [--gpu-gib 120] [--port 8011] [--layers N]
    kserve.py ... --prompt "text"          one completion on stdout, with timings, instead of the HTTP server
    kserve.py ... --selfcheck              cached decode against a full forward, packed against unpacked experts

What goes where:
  GPU          everything that is not a routed expert (attention, shared experts, router, embeddings: 107 GiB as
               published, BF16), the KV cache, and a cache of `--gpu-gib` worth of packed experts
  host memory  the packed experts, memory-mapped from their files: the kernel's page cache keeps as many as fit in RAM
               and reads the rest from disk when they are touched
An expert is 8.4 MiB packed. When a token needs one that is not in the GPU cache, the least recently used expert of
that layer gives up its slot and the new one is copied in (one copy); kpack.py's kernel multiplies straight from
the packed bytes, so nothing is ever unpacked.

The non-expert part runs through Moonshot's reference modeling code, which implements dense attention: it equals
K3.1's sparse attention for contexts up to 2,048 tokens, which is therefore this server's context limit.
"""
import argparse
import importlib
import json
import os
import struct
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kpack import layout, matvec  # noqa: E402
from kstream import PREFIX, Checkpoint, StreamMoE  # noqa: E402

MAX_CONTEXT = 2048
STAGE = 32                      # experts moved to the GPU per transfer
POOL = ThreadPoolExecutor(STAGE)


def mmap_experts(path):
    """The `experts` tensor of a layer file as a read-only memory map [E, BLOB], and the file's metadata."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
    t = head["experts"]
    return np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n + t["data_offsets"][0], shape=tuple(t["shape"])), head["__metadata__"]


class ExpertCache:
    """One layer's packed experts: all of them memory-mapped on the host, `slots` of them on the GPU (LRU)."""

    stage = stage_np = None

    def __init__(self, path, slots, device):
        self.host, md = mmap_experts(path)
        assert int(md["group"]) == 64
        self.lay = layout(int(md["inter"]), int(md["latent"]))
        E = self.host.shape[0]
        self.slots = min(slots, E)
        self.gpu = torch.empty((self.slots, self.lay["size"]), dtype=torch.uint8, device=device)
        self.slot_of = np.full(E, -1, dtype=np.int64)
        self.expert_of = np.full(self.slots, -1, dtype=np.int64)
        self.last = np.zeros(self.slots, dtype=np.int64)
        self.clock = self.hits = self.misses = 0
        self.lut = torch.zeros(E, dtype=torch.int32, device=device)
        if ExpertCache.stage is None:                  # shared by all layers: they load one after the other
            ExpertCache.stage = torch.empty((STAGE, self.lay["size"]), dtype=torch.uint8).pin_memory()
            ExpertCache.stage_np = ExpertCache.stage.numpy()

    def _stage(self, e, i):
        np.copyto(self.stage_np[i], self.host[e])

    def ensure(self, experts: np.ndarray) -> torch.Tensor:
        """Make `experts` (unique ids, at most `slots`) resident; returns the expert -> slot table (a GPU tensor)."""
        self.clock += 1
        have = self.slot_of[experts]
        self.last[have[have >= 0]] = self.clock
        need = experts[have < 0]
        self.hits += len(experts) - len(need)
        self.misses += len(need)
        if len(need):
            victims = np.argpartition(self.last, len(need) - 1)[:len(need)]      # least recently used slots
            old = self.expert_of[victims]
            self.slot_of[old[old >= 0]] = -1
            # host -> pinned staging buffer with one thread per expert (a memcpy is bound by one core), then one
            # transfer to the GPU and a scatter into the slots
            for a in range(0, len(need), STAGE):
                es = need[a:a + STAGE].tolist()
                list(POOL.map(self._stage, es, range(len(es))))
                idx = torch.from_numpy(victims[a:a + STAGE]).to(self.gpu.device)
                self.gpu[idx] = self.stage[:len(es)].to(self.gpu.device)     # blocking: the buffer is reused
            self.slot_of[need], self.expert_of[victims], self.last[victims] = victims, need, self.clock
            self.lut[torch.from_numpy(need).to(self.lut.device)] = torch.from_numpy(victims).to(self.lut.device, torch.int32)
        return self.lut


class PackedMoE(StreamMoE):
    """The MoE block with its routed experts computed from an ExpertCache."""

    cache: ExpertCache = None

    def forward(self, hidden_states):
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        idx, w = self.route(x)
        h = self.routed_expert_down_proj(x).float()
        c, lay = self.cache, self.cache.lay
        flat = idx.reshape(-1)
        tok = torch.arange(x.shape[0], device=x.device).repeat_interleave(self.top_k)
        wflat = w.reshape(-1, 1).float()
        y = torch.zeros_like(h)
        uniq = torch.unique(flat).cpu().numpy()
        for a in range(0, len(uniq), c.slots):            # a long prompt can need more experts than there are slots
            part = uniq[a:a + c.slots]
            lut = c.ensure(part)
            if len(uniq) > c.slots:
                sel = ((flat >= int(part[0])) & (flat <= int(part[-1]))).nonzero()[:, 0]
                t, e, ww = tok[sel], flat[sel], wflat[sel]
            else:
                t, e, ww = tok, flat, wflat
            slot = lut[e]
            z = h[t]
            mid = self.act(torch.cat((matvec(c.gpu, lay, "w1", z, slot), matvec(c.gpu, lay, "w3", z, slot)), dim=-1))
            y.index_add_(0, t, matvec(c.gpu, lay, "w2", mid.float(), slot) * ww)
        y = self.routed_expert_up_proj(self.routed_expert_norm(y.to(x.dtype)))
        return y.view(shape) + self.shared_experts(hidden_states)


def _set(model, name, tensor):
    mod = model
    *path, leaf = name.split(".")
    for p in path:
        mod = getattr(mod, p)
    mod._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=False)


@torch.no_grad()
def load(model_dir, ref_dir, expert_dirs, gpu_gib, n_layers=None, device="cuda", dtype=torch.bfloat16, stand_in=None):
    sys.path.insert(0, os.path.dirname(os.path.abspath(ref_dir)))
    pkg = os.path.basename(os.path.abspath(ref_dir))
    ref = importlib.import_module(pkg + ".modeling_kimi_linear")
    cfgmod = importlib.import_module(pkg + ".configuration_kimi_k3")
    config = cfgmod.KimiLinearConfig(**json.load(open(os.path.join(model_dir, "config.json")))["text_config"])
    config._attn_implementation = "eager"
    if n_layers:
        config.num_hidden_layers = n_layers           # a truncated model, for tests
    orig = ref.KimiSparseMoeBlock
    ref.KimiSparseMoeBlock = lambda c: PackedMoE(c, ref)
    try:
        with torch.device("meta"):                    # no memory, no initialisation: every parameter is set below
            model = ref.KimiLinearForCausalLM(config)
    finally:
        ref.KimiSparseMoeBlock = orig
    ckpt = Checkpoint(model_dir)
    names = [n for n, _ in model.named_parameters()]
    t0 = time.time()
    for a in range(0, len(names), 64):
        chunk = names[a:a + 64]
        got = ckpt.get(["language_model." + n for n in chunk], device)
        for n in chunk:
            _set(model, n, got["language_model." + n].to(dtype))
    assert not [n for n, b in model.named_buffers() if b.is_meta], "a buffer was left without data"
    moes = [(i, l.block_sparse_moe) for i, l in enumerate(model.model.layers) if hasattr(l, "block_sparse_moe")]
    blob = None
    for i, moe in moes:
        path = next((p for p in (os.path.join(d, f"layer-{i:02d}.safetensors") for d in expert_dirs)
                     if os.path.exists(p)), stand_in)       # stand_in: tests of the machinery on a few layers
        if blob is None:
            blob = mmap_experts(path)[0].shape[1]
            slots = int(gpu_gib * 2**30 / (len(moes) * blob))
        moe.cache = ExpertCache(path, slots, device)
    print(f"loaded {len(names)} tensors in {time.time() - t0:.0f}s; {len(moes)} MoE layers, {slots} expert slots per layer "
          f"on the GPU; GPU memory in use {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    return model.eval(), config


@torch.no_grad()
def generate(model, ids, max_new, stop, temperature=0.0, top_p=1.0, seed=None):
    """Yield new token ids one at a time. ids: list of prompt token ids."""
    assert len(ids) + max_new <= MAX_CONTEXT, f"prompt + output must stay within {MAX_CONTEXT} tokens"
    dev = model.lm_head.weight.device
    g = torch.Generator(device=dev)
    if seed is not None:
        g.manual_seed(seed)
    out = model(input_ids=torch.tensor([ids], device=dev), use_cache=True, generation_mode=True)
    past = out.past_key_values
    for _ in range(max_new):
        logits = out.logits[0, -1].float()
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            p = torch.softmax(logits / temperature, -1)
            if top_p < 1:
                sp, si = p.sort(descending=True)
                keep = sp.cumsum(0) - sp < top_p
                p = torch.zeros_like(p).scatter_(0, si[keep], sp[keep])
            nxt = int(torch.multinomial(p / p.sum(), 1, generator=g))
        if nxt in stop:
            return
        yield nxt
        out = model(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=past, use_cache=True,
                    generation_mode=True)


def cache_stats(model):
    cs = [l.block_sparse_moe.cache for l in model.model.layers if hasattr(l, "block_sparse_moe")]
    h, m = sum(c.hits for c in cs), sum(c.misses for c in cs)
    return h, m


@torch.no_grad()
def selfcheck(model, config, tokens):
    """(1) token-by-token cached decode equals one full forward; (2) packed experts equal the unpacked ones."""
    from kpack import from_blob, unpack
    dev = model.lm_head.weight.device
    ids = tokens[:1, :48].to(dev)
    full = model(input_ids=ids, use_cache=False).logits[0].float()
    out = model(input_ids=ids[:, :16], use_cache=True)
    past, steps = out.past_key_values, [out.logits[0].float()]
    for t in range(16, ids.shape[1]):
        out = model(input_ids=ids[:, t:t + 1], past_key_values=past, use_cache=True)
        steps.append(out.logits[0].float())
    inc = torch.cat(steps)
    rel = ((inc - full).norm() / full.norm()).item()
    agree = (inc.argmax(-1) == full.argmax(-1)).float().mean().item()
    # The reference code computes in bf16, where the same forward run twice (or on a prefix) already differs in the
    # last bit of many activations; that noise is the yardstick for the cached path.
    again = model(input_ids=ids, use_cache=False).logits[0].float()
    prefix = model(input_ids=ids[:, :16], use_cache=False).logits[0].float()
    noise = max(((again - full).norm() / full.norm()).item(), ((prefix - full[:16]).norm() / full[:16].norm()).item())
    print(f"selfcheck 1: cached decode vs full forward: relative difference {rel:.2e} (the full forward against itself: "
          f"{noise:.2e}), top-1 agreement {agree:.4f}")
    moe = next(l.block_sparse_moe for l in model.model.layers if hasattr(l, "block_sparse_moe"))
    x = torch.randn(1, 24, config.hidden_size, device=dev, dtype=torch.bfloat16)
    got = moe(x).float()
    xf = x.reshape(-1, x.shape[-1])
    idx, w = moe.route(xf)
    h = moe.routed_expert_down_proj(xf).float()
    y = torch.zeros_like(h)
    for e in torch.unique(idx).tolist():               # the same block with each used expert unpacked to float32
        blob = torch.from_numpy(np.array(moe.cache.host[e])).to(dev)
        w1, w2, w3 = (unpack(*from_blob(blob, moe.cache.lay, n)) for n in ("w1", "w2", "w3"))
        t, k = (idx == e).nonzero(as_tuple=True)
        z = h[t]
        y.index_add_(0, t, (moe.act(torch.cat((z @ w1.T, z @ w3.T), dim=-1)).float() @ w2.T) * w[t, k].float().unsqueeze(1))
    want = (moe.routed_expert_up_proj(moe.routed_expert_norm(y.to(x.dtype))).view(x.shape) + moe.shared_experts(x)).float()
    rel2 = ((got - want).norm() / want.norm()).item()
    print(f"selfcheck 2: MoE block from packed experts vs unpacked: relative difference {rel2:.2e}")
    assert rel < 3 * noise + 1e-3 and rel2 < 1e-3
    print("selfcheck: PASS")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir"); p.add_argument("--ref", required=True); p.add_argument("--experts", required=True)
    p.add_argument("--gpu-gib", type=float, default=120); p.add_argument("--layers", type=int)
    p.add_argument("--port", type=int, default=8011); p.add_argument("--prompt"); p.add_argument("--max-new", type=int, default=128)
    p.add_argument("--selfcheck", action="store_true"); p.add_argument("--tokens")
    p.add_argument("--stand-in", help="tests only: layer file to use for layers that have none")
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
    model, config = load(a.model_dir, a.ref, a.experts.split(","), a.gpu_gib, a.layers, stand_in=a.stand_in)
    stop = {tok.encode(s)[0] for s in ("<|end_of_msg|>",)}
    if a.selfcheck:
        selfcheck(model, config, torch.load(a.tokens))
    elif a.prompt:
        ids = tok.apply_chat_template([{"role": "user", "content": a.prompt}], add_generation_prompt=True, tokenize=True)
        ids = ids["input_ids"] if isinstance(ids, dict) else ids
        t0 = time.time(); n = 0; t1 = None; out = []
        for t in generate(model, list(ids), a.max_new, stop):
            t1 = t1 or time.time(); n += 1; out.append(t)
        dt = time.time() - t1 if t1 else 0
        h, m = cache_stats(model)
        print(tok.decode(out))
        print(f"\n{len(ids)} prompt tokens, first token after {t1 - t0:.2f}s; {n} tokens, {(n - 1) / max(dt, 1e-9):.2f} tok/s; "
              f"expert cache: {m / max(1, h + m):.1%} misses")
    else:
        from kserve_http import serve
        serve(model, tok, stop, a.port)
