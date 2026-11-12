#!/usr/bin/env python3
"""Write Kimi K3.1 Pro without its routed experts: the 107 GiB that vLLM loads itself when the experts come from
packed files (hook/sitecustomize.py).

    kstrip.py MODEL_DIR OUT_DIR [--workers 8]          one safetensors file per layer, rest.safetensors for the others
    kstrip.py --truncate N OUT_DIR SHORT_DIR             a test model of the first N layers: links into OUT_DIR

The truncated model keeps the embeddings, final norm and head, so it loads and generates (nonsense) quickly; it is
for exercising the serving path, not for quality.
"""
import argparse
import json
import multiprocessing
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor

from safetensors import safe_open
from safetensors.torch import save_file

LAYER = re.compile(r"\.layers\.(\d+)\.")
EXPERT = re.compile(r"\.block_sparse_moe\.experts\.\d+\.")


def write(args):
    src, dst, out, names, weight_map = args
    by_file, tensors = {}, {}
    for n in names:
        by_file.setdefault(weight_map[n], []).append(n)
    for fn, ns in by_file.items():
        with safe_open(os.path.join(src, fn), "pt") as f:
            for n in ns:
                tensors[n] = f.get_tensor(n)
    save_file(tensors, os.path.join(dst, out), metadata={"format": "pt"})
    return out, names


def strip(model_dir, out_dir, workers):
    os.makedirs(out_dir, exist_ok=True)
    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    wm = index["weight_map"]
    groups = {}
    for n in wm:
        if EXPERT.search(n):
            continue
        m = LAYER.search(n) if n.startswith("language_model.") else None
        groups.setdefault(f"layer-{int(m.group(1)):02d}.safetensors" if m else "rest.safetensors", []).append(n)
    new_map = {}
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        for out, names in ex.map(write, [(model_dir, out_dir, o, ns, wm) for o, ns in sorted(groups.items(), key=lambda g: g[0] != "rest.safetensors")]):
            new_map.update({n: out for n in names})
            print(out, len(names), flush=True)
    json.dump({"metadata": index.get("metadata", {}), "weight_map": dict(sorted(new_map.items()))},
              open(os.path.join(out_dir, "model.safetensors.index.json"), "w"), indent=2)
    for fn in os.listdir(model_dir):
        s = os.path.join(model_dir, fn)
        if fn.endswith(".safetensors") or fn == "model.safetensors.index.json":
            continue
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(out_dir, fn), dirs_exist_ok=True)
        elif fn == "config.json":
            # the published quantization_config describes the routed experts (MXFP4), which are gone; without it
            # vLLM can quantize what is left at load time (--quantization)
            cfg = json.load(open(s))
            cfg["text_config"].pop("quantization_config", None)
            json.dump(cfg, open(os.path.join(out_dir, fn), "w"), indent=2)
        else:
            shutil.copy(s, os.path.join(out_dir, fn))
    print("done", flush=True)


def truncate(n, full, short):
    os.makedirs(short, exist_ok=True)
    rel = os.path.relpath(full, short)
    index = json.load(open(os.path.join(full, "model.safetensors.index.json")))
    keep = {k: v for k, v in index["weight_map"].items()
            if v == "rest.safetensors" or int(v[6:8]) < n}
    for fn in os.listdir(full):
        d = os.path.join(short, fn)
        if os.path.lexists(d):
            os.remove(d)
        if fn == "model.safetensors.index.json":
            json.dump({"metadata": index.get("metadata", {}), "weight_map": keep}, open(d, "w"), indent=2)
        elif fn == "config.json":
            cfg = json.load(open(os.path.join(full, fn)))
            t = cfg["text_config"]
            t["num_hidden_layers"] = n
            for sect, key in (("linear_attn_config", "full_attn_layers"), ("linear_attn_config", "kda_layers"),
                              ("dsa_config", "dsa_layers")):
                t[sect][key] = [i for i in t[sect][key] if i <= n]         # these lists count layers from 1
            json.dump(cfg, open(d, "w"), indent=2)
        elif not fn.endswith(".safetensors") or fn in set(keep.values()):
            os.symlink(os.path.join(rel, fn), d)
    print(f"{short}: {n} layers", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a"); p.add_argument("b"); p.add_argument("--truncate", type=int)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if a.truncate:
        truncate(a.truncate, a.a, a.b)
    else:
        strip(a.a, a.b, a.workers)
