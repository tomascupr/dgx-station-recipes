#!/usr/bin/env python3
"""kcompare.py get URL OUT.json [N]      vLLM: tokenize a chat prompt, greedy-continue N tokens with top-5 logprobs
kcompare.py diff A.json B                 compare with another run's JSON or with kserve.py --ids output (STEPS line)"""
import json
import sys
import urllib.request


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


if sys.argv[1] == "get":
    url, out, n = sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 16
    ids = post(url + "/tokenize", {"model": "kimi-k3.1-pro", "messages": [
        {"role": "user", "content": "Explain how a B-tree keeps itself balanced on insert."}]})["tokens"]
    r = post(url + "/v1/completions", {"model": "kimi-k3.1-pro", "prompt": ids, "max_tokens": n, "temperature": 0,
                                       "logprobs": 5, "return_tokens_as_token_ids": True})
    lp = r["choices"][0]["logprobs"]
    steps = []
    for top in lp["top_logprobs"]:
        items = sorted(top.items(), key=lambda kv: -kv[1])
        steps.append({"ids": [int(k.split(":")[1]) for k, _ in items], "logprobs": [round(v, 4) for _, v in items]})
    json.dump({"prompt": ids, "steps": steps}, open(out, "w"))
    json.dump(ids, open(out + ".ids", "w"))
    print(len(ids), "prompt tokens;", [s["ids"][0] for s in steps])
else:
    v = json.load(open(sys.argv[2]))["steps"]
    if sys.argv[3].endswith(".json"):                   # another vLLM run
        r = json.load(open(sys.argv[3]))["steps"]
    else:
        r = json.loads(next(l for l in open(sys.argv[3]) if l.startswith("STEPS "))[6:])
    same = 0
    for i, (a, b) in enumerate(zip(v, r)):
        if a["ids"][0] != b["ids"][0]:
            print(f"step {i}: A chose {a['ids'][0]} ({a['logprobs'][0]}), B {b['ids'][0]} ({b['logprobs'][0]}); "
                  f"paths diverge here")
            break
        same += 1
        common = [(a["logprobs"][a["ids"].index(t)], b["logprobs"][b["ids"].index(t)]) for t in a["ids"] if t in b["ids"]]
        d = max(abs(x - y) for x, y in common)
        print(f"step {i}: token {a['ids'][0]}  logprob A {a['logprobs'][0]:8.4f}  B {b['logprobs'][0]:8.4f}  "
              f"top-5 overlap {len(common)}/5, largest logprob gap {d:.3f}")
    print(f"{same} of {min(len(v), len(r))} greedy tokens identical")
