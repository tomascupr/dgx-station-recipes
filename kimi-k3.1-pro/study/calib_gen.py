#!/usr/bin/env python3
"""Calibration text for kstream.py: send requests of the routing mix (build_mix.py) to an OpenAI-compatible server
and keep each conversation with the answer it got, as JSON lines.

    calib_gen.py MIX.jsonl OUT.jsonl --base-url URL --model NAME [--key-file F] [--n 480] [--conc 16]
                 [--max-prompt-chars 12000] [--max-tokens 1500]

The answers come from another model than the one under study, so the study model reads text it did not write
(teacher forcing on plausible assistant text). Stdlib only.
"""
import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("mix")
p.add_argument("out")
p.add_argument("--base-url", required=True)
p.add_argument("--model", required=True)
p.add_argument("--key-file", default="")
p.add_argument("--n", type=int, default=480)
p.add_argument("--conc", type=int, default=16)
p.add_argument("--max-prompt-chars", type=int, default=12000)
p.add_argument("--max-tokens", type=int, default=1500)
a = p.parse_args()
key = Path(a.key_file).expanduser().read_text().strip() if a.key_file else "none"
items = [json.loads(line) for line in open(a.mix).read().splitlines()[1:]]
items = [i for i in items if len(json.dumps(i["body"]["messages"])) <= a.max_prompt_chars][: a.n]


def one(item):
    body = {k: v for k, v in item["body"].items() if k in ("messages", "tools")}
    body.update(model=a.model, max_tokens=a.max_tokens)
    req = urllib.request.Request(a.base_url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        m = json.load(urllib.request.urlopen(req, timeout=1800))["choices"][0]["message"]
    except Exception as exc:
        return dict(kind=item["kind"], error=f"{type(exc).__name__}: {exc}"[:200])
    return dict(kind=item["kind"], messages=body["messages"], tools=body.get("tools"),
                answer=dict(role="assistant", content=m.get("content") or "",
                            reasoning_content=m.get("reasoning") or m.get("reasoning_content") or "",
                            tool_calls=m.get("tool_calls") or None))


with ThreadPoolExecutor(a.conc) as ex, open(a.out, "w") as out:
    recs = list(ex.map(one, items))
    for r in recs:
        out.write(json.dumps(r) + "\n")
kinds = {}
for r in recs:
    kinds[r["kind"]] = kinds.get(r["kind"], 0) + ("error" not in r)
print(json.dumps(dict(requests=len(recs), errors=sum("error" in r for r in recs), ok_by_kind=kinds)))
