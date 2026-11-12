#!/usr/bin/env python3
"""vbench.py [--url U] [--model M] [--max-tokens N] [--runs R]   single-stream decode speed of an OpenAI endpoint.

Streams R chat completions one after another and prints, per run, the time to the first token and the tokens per
second after it (completion tokens from the server's usage record), then the median."""
import argparse
import json
import statistics
import time
import urllib.request

PROMPTS = ["Explain how a B-tree keeps itself balanced on insert, with a worked example.",
           "Write a Python function that parses ISO 8601 durations and returns seconds. Include tests.",
           "Summarise the causes of the 1997 Asian financial crisis in about 300 words."]

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://127.0.0.1:30005"); p.add_argument("--model", default="kimi-k3.1-pro")
p.add_argument("--max-tokens", type=int, default=300); p.add_argument("--runs", type=int, default=3)
p.add_argument("--show", action="store_true")
a = p.parse_args()
rates = []
for r in range(a.runs):
    body = json.dumps({"model": a.model, "messages": [{"role": "user", "content": PROMPTS[r % len(PROMPTS)]}],
                       "max_tokens": a.max_tokens, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(a.url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time(); first = None; usage = None; text = []
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for line in resp:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                piece = (delta.get("reasoning_content") or delta.get("reasoning") or "") + (delta.get("content") or "")
                if piece:
                    first = first or time.time()
                    text.append(piece)
    t1 = time.time()
    n = usage["completion_tokens"]
    rate = (n - 1) / (t1 - first)
    rates.append(rate)
    print(f"run {r}: prompt {usage['prompt_tokens']} tokens, first token {first - t0:.2f}s, {n} tokens, {rate:.2f} tok/s",
          flush=True)
    if a.show:
        print("".join(text)[:600].replace("\n", " "), flush=True)
print(f"median {statistics.median(rates):.2f} tok/s")
