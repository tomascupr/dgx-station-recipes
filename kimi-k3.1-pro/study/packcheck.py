"""Check a packed layer file against the native checkpoint: shape, and the weight error of what it unpacks to."""
import sys, torch
from safetensors import safe_open
from kpack import from_blob, layout, unpack
from kstream import Checkpoint, decode_mxfp4, PREFIX
f = safe_open(sys.argv[1], "pt", device="cuda")
ck, layer = Checkpoint(sys.argv[2]), int(sys.argv[3])
md = f.metadata()
lay = layout(int(md["inter"]), int(md["latent"]))
print("experts", tuple(f.get_slice("experts").get_shape()), "blob", lay["size"], md)
for e in (0, 500, 895):
    blob = f.get_slice("experts")[e:e + 1][0]
    for w in ("w1", "w2", "w3"):
        b = f"{PREFIX}layers.{layer}.block_sparse_moe.experts.{e}.{w}.weight"
        t = ck.get([b + "_packed", b + "_scale"], "cuda")
        nat = decode_mxfp4(t[b + "_packed"], t[b + "_scale"], torch.float32)
        q = unpack(*from_blob(blob, lay, w))
        print(e, w, tuple(q.shape), "weight error", round(((q - nat).norm() / nat.norm()).item(), 4))
