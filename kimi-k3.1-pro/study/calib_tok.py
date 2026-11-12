#!/usr/bin/env python3
"""Tokenize calib_gen.py's conversations with Kimi's own tokenizer and chat format, into TOKENS.pt for kstream.py.

    calib_tok.py MODEL_DIR CALIB.jsonl TOKENS.pt [--len 1024] [--n 192]

Each conversation is rendered with its answer (thinking, response, tool calls) by the model's apply_chat_template.
Conversations of at least --len tokens are cut to --len; shorter ones are packed whole, in file order, into rows of
--len (a row is then several complete conversations, each starting with [BOS]... as rendered, and the tail of the
last is cut), so every row has the same length and none is padded.
"""
import argparse
import json

import torch
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("model_dir")
p.add_argument("calib")
p.add_argument("out")
p.add_argument("--len", type=int, default=1024)
p.add_argument("--n", type=int, default=192)
a = p.parse_args()
tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
rows, short, kinds, skipped = [], [], {}, 0
for line in open(a.calib):
    r = json.loads(line)
    if "error" in r:
        continue
    ans = {k: v for k, v in r["answer"].items() if v}
    ans.setdefault("content", "")
    try:
        ids = tok.apply_chat_template(r["messages"] + [ans], tools=r.get("tools"), tokenize=True,
                                      add_generation_prompt=False)
    except Exception as exc:
        skipped += 1
        if skipped <= 3:
            print("skip:", r["kind"], type(exc).__name__, str(exc)[:160])
        continue
    ids = ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "input_ids") else ids
    ids = list(ids[0]) if ids and isinstance(ids[0], (list, tuple)) else list(ids)
    if len(ids) >= a.len:
        rows.append(ids[: a.len])
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    else:
        short += ids
        kinds[r["kind"] + " (packed)"] = kinds.get(r["kind"] + " (packed)", 0) + 1
packed = [short[i:i + a.len] for i in range(0, len(short) - a.len + 1, a.len)]
# interleave long and packed rows so a prefix of the file is still a mix
out = [x for pair in zip(rows, packed) for x in pair] + rows[len(packed):] + packed[len(rows):]
out = out[: a.n]
t = torch.tensor(out, dtype=torch.int64)
torch.save(t, a.out)
print(json.dumps(dict(rows=len(out), long_rows=len(rows), packed_rows=len(packed), len=a.len, skipped=skipped,
                      conversations=kinds)))
print(tok.decode(out[0][:120]))
