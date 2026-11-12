#!/usr/bin/env python3
"""Score a served Kimi K3.1 Pro against the published model on the held-out calibration rows.

    kgate.py MODEL_DIR NATIVE_OUT_DIR [--url http://127.0.0.1:30005] [--first 96] [--rows N] [--out FILE]

NATIVE_OUT_DIR holds kstream.py's final.pt of the published model (its last hidden states for every token of every
row). For each row from --first on, the server scores the same tokens (one /v1/completions call with
prompt_logprobs: the logprob of every prompt token given the ones before it, and its top-1 token). Reported over the
tokens of assistant messages, as metrics.py does: the published model's perplexity (must reproduce its metrics.json),
the served model's perplexity, and how often both pick the same top-1 token.
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import urllib.request

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kstream import Checkpoint  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("model_dir"); p.add_argument("native")
p.add_argument("--url", default="http://127.0.0.1:30005"); p.add_argument("--first", type=int, default=96)
p.add_argument("--rows", type=int); p.add_argument("--out")
a = p.parse_args()

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics.py")).read()
ns: dict = {"torch": torch}
exec(src[src.index("def assistant_mask"):src.index("asst = assistant_mask")], ns)      # metrics.py's mask, verbatim
d = torch.load(os.path.join(a.native, "final.pt"))
final, tokens = d["final"][0, a.first:], d["tokens"][a.first:]
if a.rows:
    final, tokens = final[:a.rows], tokens[:a.rows]
mask = ns["assistant_mask"](tokens, AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True))[:, 1:]
head = Checkpoint(a.model_dir).get(["language_model.lm_head.weight"], "cuda")["language_model.lm_head.weight"]

tot = dict(n=0, nll_native=0.0, nll_served=0.0, agree=0, gap=0.0)
with torch.no_grad():
    for r in range(tokens.shape[0]):
        ids = tokens[r].tolist()
        lp = F.log_softmax((final[r, :-1].cuda() @ head.T).float(), dim=-1)
        tgt = tokens[r, 1:].cuda()
        nll_n = -lp.gather(1, tgt[:, None])[:, 0].cpu()
        top_n = lp.argmax(-1).cpu()
        body = json.dumps({"model": "kimi-k3.1-pro", "prompt": ids, "max_tokens": 1, "temperature": 0,
                           "prompt_logprobs": 1}).encode()
        req = urllib.request.Request(a.url + "/v1/completions", body, {"Content-Type": "application/json"})
        pl = json.load(urllib.request.urlopen(req, timeout=3600))["choices"][0]["prompt_logprobs"]
        assert len(pl) == len(ids) and pl[0] is None, "the server did not score the prompt token by token"
        for t in range(1, len(ids)):
            if not mask[r, t - 1]:
                continue
            entry = pl[t]
            actual = entry[str(ids[t])]["logprob"]
            top = min(entry.items(), key=lambda kv: kv[1]["rank"])[0]
            tot["n"] += 1
            tot["nll_native"] += float(nll_n[t - 1])
            tot["nll_served"] += -actual
            tot["agree"] += int(int(top) == int(top_n[t - 1]))
            tot["gap"] += abs(-actual - float(nll_n[t - 1]))
        if (r + 1) % 8 == 0 or r + 1 == tokens.shape[0]:
            n = max(tot["n"], 1)
            print(f"rows {a.first}..{a.first + r}: {n} assistant tokens, perplexity published "
                  f"{math.exp(tot['nll_native'] / n):.4f} served {math.exp(tot['nll_served'] / n):.4f}, "
                  f"top-1 agreement {tot['agree'] / n:.4f}, mean |logprob gap| {tot['gap'] / n:.4f}", flush=True)
n = tot["n"]
res = dict(first_row=a.first, rows=tokens.shape[0], tokens_asst=n, ppl_native_asst=round(math.exp(tot["nll_native"] / n), 4),
           ppl_served_asst=round(math.exp(tot["nll_served"] / n), 4), top1_asst=round(tot["agree"] / n, 4),
           mean_abs_logprob_gap=round(tot["gap"] / n, 4))
print(json.dumps(res))
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
