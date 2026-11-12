"""Routing skew of a native pass: what share of expert calls the most used experts of each layer take."""
import sys, torch
d = torch.load(sys.argv[1])
c = d["counts"][0].double()                     # [L, E]
w = d["wsum"][0].double() if "wsum" in d else None
L, E = c.shape
print("layers", L, "experts", E, "calls/layer", c.sum(1)[1].item() if L > 1 else c.sum().item())
def share(x, label):
    s = x.sort(1, descending=True).values.cumsum(1) / x.sum(1, keepdim=True).clamp_min(1)
    rows = [l for l in range(L) if x[l].sum() > 0]
    print(label, "layers with data:", len(rows))
    for f in (0.05, 0.10, 0.25, 0.50, 0.75):
        v = s[rows, round(f * E) - 1]
        print(f"  top {f:4.0%} of experts: mean {v.mean():.3f}  min {v.min():.3f}  max {v.max():.3f}")
    return s, rows
s, rows = share(c, "route counts:")
if w is not None: share(w, "gate weight:")
print("never-used experts per layer: mean", (c[rows] == 0).sum(1).double().mean().item(), "max", (c[rows] == 0).sum(1).max().item())
for l in rows[::10]:
    print(f"layer {l:2d} top10% {s[l, round(.1*E)-1]:.3f} top25% {s[l, round(.25*E)-1]:.3f} top50% {s[l, round(.5*E)-1]:.3f} unused {(c[l]==0).sum().item()}")
# global ranking: how many expert-slots (across all layers) cover X of calls
g = c[rows].flatten().sort(descending=True).values.cumsum(0) / c[rows].sum()
for f in (0.05, 0.10, 0.2, 0.25, 0.3, 0.4, 0.50):
    print(f"global top {f:4.0%}: {g[round(f * g.numel()) - 1]:.3f}")
