#!/usr/bin/env python3
"""Score kstream.py's streams at the logits: each compressed stream against stream 0 (the published model).

    metrics.py MODEL_DIR OUT_DIR [--score-from ROW]

Per stream: perplexity on the calibration tokens (next-token, positions 1..T-1), and against stream 0 the mean KL
divergence KL(native || stream) in nats per token, and how often the top-1 token is the same. Stream 0's perplexity
is also the sanity check of the whole pass: a wrong forward does not predict text.

Every number is given twice: over all tokens, and (suffix _asst) over the tokens of assistant messages only. The
model predicts the chat format's own scaffolding and the system and user text much worse than its own kind of
output (5% of the tokens, mostly format tokens, carry 31% of the native loss), so the assistant numbers are the
ones to read.
"""
import json
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from kstream import PREFIX, Checkpoint  # noqa: E402

model_dir, out_dir = sys.argv[1], sys.argv[2]
first = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[3] == "--score-from" else 0   # skip calibration rows
d = torch.load(f"{out_dir}/final.pt")
final, tokens, names = d["final"][:, first:], d["tokens"][first:], d["names"]
S, N, T, H = final.shape
head = Checkpoint(model_dir).get(["language_model.lm_head.weight"], "cuda")["language_model.lm_head.weight"]


def assistant_mask(tokens, tok):
    """True where the token to predict lies inside an assistant message (after its header, up to its end)."""
    OPEN, SEP, END = (tok.encode(s)[0] for s in ("<|open|>", "<|sep|>", "<|end_of_msg|>"))
    MESSAGE = tok.encode("message")[0]
    mask = torch.zeros(tokens.shape, dtype=torch.bool)
    for r, row in enumerate(tokens.tolist()):
        inside, i = False, 0                       # a row may start mid-message: unknown role, not counted
        while i < len(row):
            if row[i] == OPEN and i + 1 < len(row) and row[i + 1] == MESSAGE:
                j = i
                while j < len(row) and row[j] != SEP:
                    j += 1
                inside = 'role="assistant"' in tok.decode(row[i:j])
                i = j + 1
                continue
            if row[i] == END:
                inside = False
            mask[r, i] = inside
            i += 1
    return mask


asst = assistant_mask(tokens, AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True))
KEYS = ("nll", "kl", "top1")
res = [dict(name=n, **{k + sfx: 0.0 for k in KEYS for sfx in ("", "_asst")}) for n in names]
cnt = cnt_asst = 0
with torch.no_grad():
    for a in range(0, N, 4):
        tgt = tokens[a:a + 4, 1:].reshape(-1).cuda()
        m = asst[a:a + 4, 1:].reshape(-1).cuda()
        lp0 = None
        for s in range(S):
            x = final[s, a:a + 4, :-1].reshape(-1, H).cuda()
            lp = F.log_softmax((x @ head.T).float(), dim=-1)
            per = dict(nll=-lp.gather(1, tgt[:, None])[:, 0])
            if s == 0:
                lp0 = lp
            else:
                per["kl"] = (lp0.exp() * (lp0 - lp)).sum(-1)
                per["top1"] = (lp.argmax(-1) == lp0.argmax(-1)).float()
            for k, v in per.items():
                res[s][k] += v.sum().item()
                res[s][k + "_asst"] += v[m].sum().item()
        cnt += tgt.numel()
        cnt_asst += int(m.sum())
for i, r in enumerate(res):
    for sfx, n in (("", cnt), ("_asst", cnt_asst)):
        r["ppl" + sfx] = round(torch.tensor(r.pop("nll" + sfx) / n).exp().item(), 4)
        r["kl" + sfx] = round(r["kl" + sfx] / n, 5)
        r["top1" + sfx] = round(r["top1" + sfx] / n, 4)
        if i == 0:
            r.pop("kl" + sfx), r.pop("top1" + sfx)
out = dict(first_row=first, tokens=cnt, tokens_asst=cnt_asst, streams=res)
print(json.dumps(out, indent=1))
json.dump(out, open(f"{out_dir}/metrics.json", "w"), indent=1)
