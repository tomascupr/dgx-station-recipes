# A copy of the tiny checkpoint whose routers actually discriminate between tokens (the random one sends every token
# to the same experts), so the selftest exercises the token-to-expert scatter.
import shutil, os, sys, torch
from safetensors.torch import load_file, save_file
src, dst = sys.argv[1], sys.argv[2]        # mkdiv.py TINY_DIR OUT_DIR
os.makedirs(dst, exist_ok=True)
shutil.copy(f"{src}/config.json", dst)
sd = load_file(f"{src}/model.safetensors")
for k in list(sd):
    if k.endswith("block_sparse_moe.gate.weight"):
        sd[k] = sd[k] * 40
    if k.endswith("e_score_correction_bias"):
        sd[k] = sd[k] * 0
save_file(sd, f"{dst}/model.safetensors")
