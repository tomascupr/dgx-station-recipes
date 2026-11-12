#!/usr/bin/env python3
"""Write a copy of Kimi K3.1 Pro that keeps only each MoE layer's KEEP most used routed experts.

    prune_ckpt.py MODEL_DIR STATS.pt KEEP OUT_DIR [--workers 8]

STATS.pt is kstream.py's stats.pt of a pass of the published model: counts[0][layer][expert] says how many tokens
were routed to each expert. Per layer the KEEP experts with the highest counts are kept, in their original order,
and renumbered 0..KEEP-1; the router's rows (gate.weight) and e_score_correction_bias entries of the others are
removed, and text_config.num_experts becomes KEEP. Everything else is copied unchanged, so the result loads wherever
the published MXFP4 checkpoint loads (it has the layout of vllm-project/Kimi-K3.1-Pro-pruned75, with a different
choice of experts).
"""
import argparse
import json
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor

import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXP = re.compile(r"^(.*\.layers\.(\d+)\.block_sparse_moe\.experts\.)(\d+)(\..*)$")
GATE = re.compile(r"^.*\.layers\.(\d+)\.block_sparse_moe\.gate\.(weight|e_score_correction_bias)$")


def shard(args):
    src, dst, fn, keep = args
    out, names = {}, []
    with safe_open(os.path.join(src, fn), "pt") as f:
        meta = f.metadata()
        for k in f.keys():
            m = EXP.match(k)
            if m:
                new = keep[int(m.group(2))].get(int(m.group(3)))
                if new is not None:
                    out[f"{m.group(1)}{new}{m.group(4)}"] = f.get_tensor(k)
                continue
            g = GATE.match(k)
            t = f.get_tensor(k)
            if g:
                t = t[torch.tensor(sorted(keep[int(g.group(1))]))].contiguous()
            out[k] = t
    if out:
        save_file(out, os.path.join(dst, fn), metadata=meta)
        names = list(out)
    return fn, names


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir"); p.add_argument("stats"); p.add_argument("keep", type=int); p.add_argument("out_dir")
    p.add_argument("--workers", type=int, default=8); p.add_argument("--only", help="tests: one shard file name")
    a = p.parse_args()
    counts = torch.load(a.stats)["counts"][0]
    keep = {}                                    # layer -> {old expert id: new id}
    for layer in range(counts.shape[0]):
        if counts[layer].sum() > 0:
            ids = sorted(counts[layer].argsort(descending=True, stable=True)[:a.keep].tolist())
            keep[layer] = {e: i for i, e in enumerate(ids)}
    share = sum(counts[l][list(keep[l])].sum() for l in keep) / counts.sum()
    print(f"keeping {a.keep} of {counts.shape[1]} experts in {len(keep)} layers: {share:.1%} of the measured calls", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    index = json.load(open(os.path.join(a.model_dir, "model.safetensors.index.json")))
    files = sorted(set(index["weight_map"].values()))
    if a.only:
        files = [a.only]
    weight_map = {}
    import multiprocessing
    # spawn: a forked child of a process that has used torch's thread pool can hang on its first tensor operation
    with ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        for fn, names in ex.map(shard, [(a.model_dir, a.out_dir, fn, keep) for fn in files]):
            weight_map.update({n: fn for n in names})
            print(fn, len(names), flush=True)
    json.dump({"metadata": index.get("metadata", {}), "weight_map": dict(sorted(weight_map.items()))},
              open(os.path.join(a.out_dir, "model.safetensors.index.json"), "w"), indent=2)
    for fn in os.listdir(a.model_dir):
        s = os.path.join(a.model_dir, fn)
        if fn.endswith(".safetensors") or fn == "model.safetensors.index.json":
            continue
        if fn == "config.json":
            cfg = json.load(open(s))
            cfg["text_config"]["num_experts"] = a.keep
            json.dump(cfg, open(os.path.join(a.out_dir, fn), "w"), indent=2)
        elif os.path.isdir(s):
            shutil.copytree(s, os.path.join(a.out_dir, fn), dirs_exist_ok=True)
        else:
            shutil.copy(s, os.path.join(a.out_dir, fn))
    json.dump({str(l): sorted(keep[l]) for l in keep}, open(os.path.join(a.out_dir, "kept_experts.json"), "w"))
    print("done", flush=True)
