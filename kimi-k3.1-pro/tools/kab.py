#!/usr/bin/env python3
"""A/B material: the same expert weights for stock vLLM and for the packed-expert hook, on a 4-layer test model.

    kab.py MODEL_DIR NOEXP_DIR AB_DIR PACK_DIR [--layers 4]

The first layers' routed experts are rounded to 2 bits with power-of-two scales (krtn.rtn(pow2=True)); such weights
are exact MXFP4 values too. They are written twice: packed (PACK_DIR/layer-NN.safetensors, for the hook) and in the
published checkpoint's MXFP4 layout (AB_DIR, a truncated model stock vLLM loads). Both servers then compute the same
function, and their logprobs may differ only by floating-point order of operations.
"""
import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kpack import GROUP, to_blob  # noqa: E402
from krtn import rtn  # noqa: E402
from kstream import PREFIX, Checkpoint, decode_mxfp4  # noqa: E402
from kstrip import truncate  # noqa: E402

NIB = torch.tensor([13, 10, 2, 5], dtype=torch.uint8)       # level -3, -1, +1, +3 as E2M1 codes

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("model_dir"); p.add_argument("noexp"); p.add_argument("ab"); p.add_argument("pack")
p.add_argument("--layers", type=int, default=4)
a = p.parse_args()
dev = "cuda"
ckpt = Checkpoint(a.model_dir)
truncate(a.layers, a.noexp, a.ab)
os.makedirs(a.pack, exist_ok=True)
index = json.load(open(os.path.join(a.ab, "model.safetensors.index.json")))
with torch.no_grad():
    for i in range(1, a.layers):
        pre = f"{PREFIX}layers.{i}.block_sparse_moe.experts."
        E = 1 + max(int(n[len(pre):].split(".")[0]) for n in ckpt.by_layer[i] if n.startswith(pre))
        native, blobs, shape = {}, [], None
        for lo in range(0, E, 32):
            es = range(lo, min(E, lo + 32))
            t = ckpt.get([f"{pre}{e}.{w}.weight{s}" for e in es for w in ("w1", "w2", "w3")
                          for s in ("_packed", "_scale")], dev)
            parts = []
            for w in ("w1", "w2", "w3"):
                W = torch.stack([decode_mxfp4(t[f"{pre}{e}.{w}.weight_packed"], t[f"{pre}{e}.{w}.weight_scale"],
                                              torch.float32) for e in es])
                shape = W.shape[1:] if w == "w1" else shape
                codes, u, _, _ = rtn(W, pow2=True)
                parts.append((codes, u))
                c = torch.stack([(codes >> s) & 3 for s in (0, 2, 4, 6)], dim=-1).flatten(-2)     # [B, out, in]
                nib = NIB.to(dev)[c.long()]
                packed = (nib[..., 0::2] | (nib[..., 1::2] << 4)).cpu()
                scale = ((u.to(torch.int16) - 200) // 8 + 127).to(torch.uint8).repeat_interleave(GROUP // 32, -1).cpu()
                for b, e in enumerate(es):
                    native[f"{pre}{e}.{w}.weight_packed"] = packed[b].contiguous()
                    native[f"{pre}{e}.{w}.weight_scale"] = scale[b].contiguous()
                if lo == 0 and w == "w1":             # the two encodings decode to the same matrix
                    from kpack import unpack
                    back = decode_mxfp4(packed[0].to(dev), scale[0].to(dev), torch.float32)
                    assert torch.equal(back, unpack(codes[0], u[0])), "MXFP4 and packed encodings disagree"
            blobs.append(to_blob(parts).cpu())
        fn = f"experts-{i:02d}.safetensors"
        save_file(native, os.path.join(a.ab, fn), metadata={"format": "pt"})
        index["weight_map"].update({n: fn for n in native})
        save_file({"experts": torch.cat(blobs)}, os.path.join(a.pack, f"layer-{i:02d}.safetensors"),
                  metadata={"group": str(GROUP), "inter": str(shape[0]), "latent": str(shape[1]), "method": "rtn-pow2"})
        print(f"layer {i}: {E} experts written in both encodings", flush=True)
json.dump(index, open(os.path.join(a.ab, "model.safetensors.index.json"), "w"), indent=2)
print("AB DONE")
