#!/usr/bin/env python3
"""kchecks.py decode|long|par [--url U]   checks of a running Kimi server beyond single short prompts.

decode  generate 200 greedy tokens (the decode path: a few tokens per step, draft verification), then score the same
        prompt + output in one prefill pass (prompt_logprobs) and compare each generated token's logprob
long    prompts of ~10k and ~30k tokens: time to the first token, prompt tokens per second
par     1, 2 and 4 requests at once: tokens per second per request and in total
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("what", choices=("decode", "long", "par")); p.add_argument("--url", default="http://127.0.0.1:30005")
a = p.parse_args()
MODEL = "kimi-k3.1-pro"


def post(path, body, timeout=3600):
    req = urllib.request.Request(a.url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def stream(messages, max_tokens):
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(a.url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time(); first = None; usage = None
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for line in resp:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            usage = d.get("usage") or usage
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                if delta.get("reasoning_content") or delta.get("reasoning") or delta.get("content"):
                    first = first or time.time()
    return usage, first - t0, time.time() - first


if a.what == "decode":
    ids = post("/tokenize", {"model": MODEL, "messages": [
        {"role": "user", "content": "Explain how a B-tree keeps itself balanced on insert, with a worked example."}]})["tokens"]
    r = post("/v1/completions", {"model": MODEL, "prompt": ids, "max_tokens": 200, "temperature": 0, "logprobs": 1,
                                 "return_tokens_as_token_ids": True})["choices"][0]["logprobs"]
    gen = [int(t.split(":")[1]) for t in r["tokens"]]
    lp_dec = r["token_logprobs"]
    pl = post("/v1/completions", {"model": MODEL, "prompt": ids + gen, "max_tokens": 1, "temperature": 0,
                                  "prompt_logprobs": 1})["choices"][0]["prompt_logprobs"]
    gaps, top_same = [], 0
    for i, (tok, lp) in enumerate(zip(gen, lp_dec)):
        entry = pl[len(ids) + i]
        gaps.append(abs(entry[str(tok)]["logprob"] - lp))
        top_same += int(min(entry.items(), key=lambda kv: kv[1]["rank"])[0] == str(tok))
    gaps.sort()
    print(f"{len(gen)} greedy tokens after a {len(ids)}-token prompt: decode-path logprob vs one prefill pass over the "
          f"same tokens: mean |gap| {statistics.mean(gaps):.4f}, median {gaps[len(gaps) // 2]:.4f}, 95th percentile "
          f"{gaps[int(len(gaps) * 0.95)]:.4f}, max {gaps[-1]:.4f}; the prefill pass ranks the generated token first "
          f"for {top_same} of {len(gen)}")
elif a.what == "long":
    para = ("The committee met again in the spring to review the harbour accounts, the state of the sea wall and the "
            "tenders for dredging the inner basin; the minutes record each objection in turn. ")
    for n in (300, 900):
        text = "".join(f"[{i}] {para}" for i in range(n))
        usage, ttft, rest = stream([{"role": "user", "content": text + "\n\nIn one sentence, what is this text about?"}], 40)
        print(f"{usage['prompt_tokens']} prompt tokens: first token after {ttft:.1f} s "
              f"({usage['prompt_tokens'] / ttft:.0f} prompt tokens/s), then {usage['completion_tokens']} tokens in {rest:.1f} s",
              flush=True)
else:
    prompts = ["Explain how a B-tree keeps itself balanced on insert, with a worked example.",
               "Write a Python function that parses ISO 8601 durations and returns seconds. Include tests.",
               "Summarise the causes of the 1997 Asian financial crisis in about 300 words.",
               "Describe how a TCP connection is opened and closed, and what each flag is for."]
    for n in (1, 2, 4):
        out = [None] * n

        def run(i):
            usage, ttft, rest = stream([{"role": "user", "content": prompts[i]}], 250)
            out[i] = (usage["completion_tokens"] - 1) / rest
        t0 = time.time()
        ts = [threading.Thread(target=run, args=(i,)) for i in range(n)]
        [t.start() for t in ts]; [t.join() for t in ts]
        print(f"{n} at once: {', '.join(f'{r:.1f}' for r in out)} tok/s each, {sum(out):.1f} tok/s in total", flush=True)
