#!/usr/bin/env python3
"""Save reference outputs: a fixed prompt set at temperature 0, full reasoning, content and tool calls.

    reference.py OUT.jsonl [--base-url URL] [--model NAME]

Prompts come from verify/data only: the first 20 GSM8K-200 questions at effort max and the first 10 at effort low,
the fixed prompt and the coding module at effort low (2,048 tokens), and the verify tool call. One JSON line per
request: the request body and the server's message, usage and finish reason. Output at temperature 0 is not
bit-reproducible across servers (LIMITATIONS of the DeepSeek recipe; batch shape and kernels differ), so compare
answers and tool calls, not bytes. Stdlib only.
"""
import argparse
import json
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "verify/data"
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def prompts() -> list:
    gsm = [json.loads(l)["question"] for l in open(DATA / "gsm8k-200.jsonl")]
    tail = "\n\nSolve the problem. End your reply with a final line of the form '#### <number>'."
    out = [dict(id=f"gsm8k-{i}-max", max_tokens=8192, chat_template_kwargs={"reasoning_effort": "max"},
                messages=[{"role": "user", "content": q + tail}]) for i, q in enumerate(gsm[:20])]
    out += [dict(id=f"gsm8k-{i}-low", max_tokens=8192, chat_template_kwargs={"reasoning_effort": "low"},
                 messages=[{"role": "user", "content": q + tail}]) for i, q in enumerate(gsm[:10])]
    code = (DATA / "coding_source.py").read_text()
    out.append(dict(id="fixed-low", max_tokens=2048, chat_template_kwargs={"reasoning_effort": "low"},
                    messages=[{"role": "user", "content": (DATA / "fixed_prompt.txt").read_text()}]))
    out.append(dict(id="coding-low", max_tokens=2048, chat_template_kwargs={"reasoning_effort": "low"},
                    messages=[{"role": "user", "content": "You are reviewing the following Python module.\n\n```python\n"
                               f"{code}\n```\n\nWrite a complete pytest test suite for this module: cover every public "
                               "function, edge cases and error paths. Output only the code."}]))
    out.append(dict(id="tool-call", max_tokens=8192, tools=[WEATHER], tool_choice="auto",
                    messages=[{"role": "user", "content": "What's the weather in Denver right now? Use the tool."}]))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("--base-url", default="http://127.0.0.1:30004")
    p.add_argument("--model", default="glm-5.3-flash")
    a = p.parse_args()
    with open(a.out, "w") as f:
        for item in prompts():
            rid = item.pop("id")
            body = dict(item, model=a.model, temperature=0)
            req = urllib.request.Request(a.base_url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            d = json.load(urllib.request.urlopen(req, timeout=1800))
            ch = d["choices"][0]
            f.write(json.dumps(dict(id=rid, request=body, message=ch["message"], finish=ch.get("finish_reason"),
                                    usage=d["usage"]), ensure_ascii=False) + "\n")
            f.flush()
            print(rid, ch.get("finish_reason"), d["usage"]["completion_tokens"], flush=True)


if __name__ == "__main__":
    main()
