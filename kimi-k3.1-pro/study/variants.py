"""Compressed variants of Kimi K3.1 Pro's routed experts, for kstream.py.

A variant is a JSON object:
  {"name": "...", "stats": "stats.pt",        # route counts of an earlier native pass; ranks experts per layer
   "tiers": [[0.25, "native"], [0.65, "q2"], [0.10, "prune"]],
   "nonexpert": "fp8"}                         # optional: the non-expert weight matrices rounded to FP8
Tiers are fractions of each layer's experts, from the most to the least used. Treatments:
  native   the published MXFP4 weights
  q3, q2   re-quantized to 8 / 4 uniform levels with one scale per group of 32 inputs (3.25 / 2.25 bits per weight)
  t        ternary {-s, 0, +s} with one scale per group of 32 (1.58 + 0.25 = 1.83 bits per weight)
  prune    removed; the router picks its 16 experts among the rest
"""
import torch


LEVELS = {"q3": 8, "q2": 4, "t": 3}


def _requant(w: torch.Tensor, levels: int, group: int = 32) -> torch.Tensor:
    """Uniform symmetric quantizer per group of `group` inputs; the scale minimises squared error over a small grid."""
    x = w.float().reshape(-1, group)
    amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    best, best_err = None, None
    for shrink in (1.0, 0.8, 0.62, 0.45):
        if levels % 2:                         # mid-tread: 0, +-s, ... (ternary when levels == 3)
            half = (levels - 1) // 2
            s = amax * (shrink / half)
            q = (x / s).round_().clamp_(-half, half).mul_(s)
        else:                                  # mid-rise: +-s/2, +-3s/2, ...
            half = levels // 2
            s = amax * (shrink / (half - 0.5))
            q = (x / s).floor_().clamp_(-half, half - 1).add_(0.5).mul_(s)
        err = (q - x).pow_(2).sum(-1, keepdim=True)
        if best is None:
            best, best_err = q, err
        else:
            take = err < best_err
            best = torch.where(take, q, best)
            best_err = torch.where(take, err, best_err)
    return best.reshape(w.shape).to(w.dtype)


def requant(w: torch.Tensor, name: str, chunk: int = 32) -> torch.Tensor:
    """Re-quantize a stack of experts [E, out, in], a few experts at a time."""
    out = torch.empty_like(w)
    for a in range(0, w.shape[0], chunk):
        out[a:a + chunk] = _requant(w[a:a + chunk], LEVELS[name])
    return out


class Variant:
    def __init__(self, spec: dict | None, config):
        self.name = spec["name"] if spec else "native"
        self.plan = None
        self.nonexpert = (spec or {}).get("nonexpert")      # "fp8": everything but the routed experts in FP8
        if spec and "tiers" in spec:
            counts = torch.load(spec["stats"])["counts"][0]            # [L, E], the native stream
            E = counts.shape[1]
            order = counts.argsort(dim=1, descending=True, stable=True)   # most used first
            self.plan = []
            for layer in range(counts.shape[0]):
                treat, start = ["native"] * E, 0
                for frac, name in spec["tiers"]:
                    n = round(frac * E)
                    for e in order[layer, start:start + n].tolist():
                        treat[e] = name
                    start += n
                self.plan.append(treat)

    def treatments(self, layer: int, E: int) -> list:
        """Per expert of this layer: "native", a LEVELS name, or "prune"."""
        return self.plan[layer] if self.plan is not None else ["native"] * E


def make_variants(specs, config) -> list:
    return [Variant(None, config)] + [Variant(s, config) for s in (specs or [])]
