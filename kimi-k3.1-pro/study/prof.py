"""Where a decode step's time goes, per layer.

    prof.py MODEL_DIR REF_DIR EXPERT_DIRS TOKENS.pt [LAYERS]     EXPERT_DIRS comma-separated; LAYERS < 93 truncates
"""
import sys, time, torch, numpy as np
M, REF, EXP, TOK = sys.argv[1:5]
ARG = sys.argv[5] if len(sys.argv) > 5 else "93"
sys.argv = ["kserve.py"]
import kserve
from kserve import load, PackedMoE, ExpertCache, generate
L = int(ARG)
model, config = load(M, REF, EXP.split(","), 120 if L >= 93 else 1.5 * L, None if L >= 93 else L,
                     stand_in=None if L >= 93 else EXP.split(",")[0] + "/layer-01.safetensors")
L = min(L, 93)
T = dict(ensure=0.0, stage=0.0, moe=0.0, layer=0.0, n=0)
import kpack
kserve.matvec = None

def timed(fn, key):
    def w(*a, **k):
        torch.cuda.synchronize(); t0 = time.perf_counter(); r = fn(*a, **k); torch.cuda.synchronize(); T[key] += time.perf_counter() - t0; return r
    return w
ExpertCache.ensure = timed(ExpertCache.ensure, "ensure")
_st = ExpertCache._stage
def _stage(self, e, i):
    t0 = time.perf_counter(); _st(self, e, i); T["stage"] += time.perf_counter() - t0
ExpertCache._stage = _stage
kserve.matvec = kpack.matvec
PackedMoE.forward = timed(PackedMoE.forward, "moe")
for l in model.model.layers: l.forward = timed(l.forward, "layer")
ids = torch.load(TOK)[0, :96].tolist()
gen = generate(model, ids, 150, set(), temperature=1.0, seed=0)
for i, t in enumerate(gen):
    if i == 20:
        for k in T: T[k] = 0.0
        t0 = time.perf_counter(); h0, m0 = kserve.cache_stats(model)
n = 150 - 21
dt = time.perf_counter() - t0
h1, m1 = kserve.cache_stats(model)
print(f"{L} layers: {dt / n * 1e3:.1f} ms/token; per layer: all {T['layer'] / n / L * 1e3:.2f} ms, of which MoE block {T['moe'] / n / L * 1e3:.2f} ms, of which expert loading {T['ensure'] / n / L * 1e3:.2f} ms (host memcpy thread-time {T['stage'] / n / L * 1e3:.2f} ms); misses {(m1 - m0) / (m1 - m0 + h1 - h0):.1%} = {(m1 - m0) / n / (L - 1):.1f} per layer per token")
