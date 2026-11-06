#!/usr/bin/env python3
"""Build the routing-calibration request mix: one JSON chat request per line, for drive.py.

    build_mix.py OUT.jsonl [--seed 1]

Six kinds of traffic, each a list of OpenAI chat bodies (no "model"; drive.py adds it):
  chat          UltraChat-200k test_gen first turns                         (HuggingFaceH4/ultrachat_200k, MIT)
  multilingual  Aya test prompts not in English (Chinese, Arabic, Portuguese, Turkish, Yoruba, Telugu), 250 of them
                                                                             (CohereLabs/aya_dataset, Apache-2.0)
  reasoning     GSM8K-200 (verify/data) and MATH-500                         (HuggingFaceH4/MATH-500)
  coding        HumanEval and MBPP-sanitized tasks                           (openai/openai_humaneval MIT, mbpp CC-BY-4.0)
  agentic       SWE-bench Lite issues with a coding-agent system prompt and file/shell tools, plus code-in-context
                tasks over vLLM and CPython sources (4K-60K-token prompts)   (princeton-nlp/SWE-bench_Lite)
  tools         BFCL v3 simple / multiple / parallel / parallel_multiple     (gorilla-llm/BFCL, Apache-2.0)
Reasoning effort is drawn per request (low 30 %, high 30 %, max 40 %); sampling is the model's default
(T=1.0, top_p 0.95). Rows come from the Hub's datasets-server, which serves each dataset's current revision; the
first line's "_meta" records those revisions at fetch time, and the output file's sha256 pins the exact mix.
Nothing here is redistributed, only used to drive the server.
"""
import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS = "https://datasets-server.huggingface.co/rows"
VLLM_COMMIT = "a2bcb5657a881a002e79d0adab3c6b5e39436121"
CPYTHON_TAG = "v3.12.7"


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "glm53-routing-mix"})
    for attempt in range(7):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 6:
                raise
            time.sleep(2 ** attempt)  # the datasets-server rate-limits bursts; back off and retry


def rows(dataset: str, config: str, split: str, n: int, offset: int = 0) -> list:
    out = []
    while len(out) < n:
        q = urllib.parse.urlencode(dict(dataset=dataset, config=config, split=split, offset=offset + len(out),
                                        length=min(100, n - len(out))))
        got = json.loads(get(f"{ROWS}?{q}"))["rows"]
        if not got:
            break
        out += [r["row"] for r in got]
    return out


def sha(dataset: str) -> str:
    return json.loads(get(f"https://huggingface.co/api/datasets/{dataset}"))["sha"]


def user(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


AGENT_SYSTEM = ("You are a coding agent working in a checked-out git repository. Use the tools to inspect the code, "
                "find the cause of the issue and fix it with minimal edits. Think before each tool call.")
AGENT_TOOLS = [{"type": "function", "function": f} for f in (
    {"name": "read_file", "description": "Read a file from the repository",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                     "start_line": {"type": "integer"}, "end_line": {"type": "integer"}},
                    "required": ["path"]}},
    {"name": "grep", "description": "Search the repository with a regular expression",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                    "required": ["pattern"]}},
    {"name": "bash", "description": "Run a shell command in the repository root",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "edit_file", "description": "Replace an exact string in a file",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"},
                                                     "new": {"type": "string"}}, "required": ["path", "old", "new"]}},
)]
CODE_TASKS = ("Explain how this code works, then list the three riskiest edge cases.",
              "Write a pytest test suite for the main public functions in this code.",
              "Find a bug or a performance problem in this code and propose a patch as a unified diff.",
              "Refactor the longest function here for readability; show the new version only.",
              "Summarize the module's responsibilities and its main data flow in a short design note.")
VLLM_FILES = ("vllm/v1/core/sched/scheduler.py", "vllm/v1/worker/gpu_model_runner.py",
              "vllm/model_executor/layers/fused_moe/layer.py", "vllm/v1/engine/core.py",
              "vllm/entrypoints/openai/chat_completion/serving.py", "vllm/v1/core/kv_cache_manager.py",
              "vllm/models/glm5next/common/model.py", "vllm/config/scheduler.py")
CPYTHON_FILES = ("Lib/json/decoder.py", "Lib/asyncio/tasks.py", "Lib/functools.py", "Lib/pathlib.py",
                 "Lib/argparse.py", "Lib/collections/__init__.py", "Lib/http/client.py", "Lib/textwrap.py")


def bfcl_tools(functions: list) -> list:
    """BFCL function specs to OpenAI tools (BFCL uses type 'dict' and Python-ish type names)."""
    fix = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}

    def walk(x):
        if isinstance(x, dict):
            return {k: (fix.get(v, v) if k == "type" and isinstance(v, str) else walk(v)) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x
    return [{"type": "function", "function": walk({k: f[k] for k in ("name", "description", "parameters") if k in f})}
            for f in functions]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    rng = random.Random(a.seed)
    kinds: dict = {}
    meta = {"vllm_sources": VLLM_COMMIT, "cpython_sources": CPYTHON_TAG, "datasets": {}}

    for d in ("HuggingFaceH4/ultrachat_200k", "CohereLabs/aya_dataset", "HuggingFaceH4/MATH-500",
              "openai/openai_humaneval", "google-research-datasets/mbpp", "princeton-nlp/SWE-bench_Lite",
              "gorilla-llm/Berkeley-Function-Calling-Leaderboard"):
        meta["datasets"][d] = sha(d)

    kinds["chat"] = [user(r["prompt"]) for r in rows("HuggingFaceH4/ultrachat_200k", "default", "test_gen", 400)]
    aya = rows("CohereLabs/aya_dataset", "default", "test", 1750)
    kinds["multilingual"] = [user(r["inputs"]) for r in rng.sample([r for r in aya if r["language"] != "English"], 250)]
    gsm = [json.loads(l)["question"] for l in open(HERE.parent / "verify/data/gsm8k-200.jsonl")]
    math = [r["problem"] for r in rows("HuggingFaceH4/MATH-500", "default", "test", 300)]
    kinds["reasoning"] = [user(q + "\n\nPut the final answer in \\boxed{}.") for q in gsm + math]
    he = [r["prompt"] for r in rows("openai/openai_humaneval", "openai_humaneval", "test", 164)]
    mbpp = [r["prompt"] for r in rows("google-research-datasets/mbpp", "sanitized", "test", 257)]
    kinds["coding"] = ([user("Complete this Python function. Return the whole function in one code block.\n\n" + s)
                        for s in he] +
                       [user(s + "\nWrite the Python function, with a short docstring.") for s in mbpp])

    swe = rows("princeton-nlp/SWE-bench_Lite", "default", "test", 300)
    agentic = [{"messages": [{"role": "system", "content": AGENT_SYSTEM},
                             {"role": "user", "content": f"Repository: {r['repo']}\n\nIssue:\n{r['problem_statement']}"}],
                "tools": AGENT_TOOLS, "tool_choice": "auto"} for r in swe]
    sources = [(f"{repo.split('/')[1]}/{f}", get(f"https://raw.githubusercontent.com/{repo}/{ref}/{f}").decode())
               for repo, ref, files in (("vllm-project/vllm", VLLM_COMMIT, VLLM_FILES),
                                        ("python/cpython", CPYTHON_TAG, CPYTHON_FILES)) for f in files]
    for i in range(120):
        k = rng.choice((1, 1, 2, 3, 5))  # files per prompt: mostly one, sometimes a long multi-file context
        picked = rng.sample(sources, k)
        ctx = "\n\n".join(f"### {name}\n```python\n{text}\n```" for name, text in picked)[:240_000]  # ~60K tokens
        agentic.append(user(f"{ctx}\n\n{rng.choice(CODE_TASKS)}"))
    kinds["agentic"] = agentic

    tools = []
    for cat in ("simple", "multiple", "parallel", "parallel_multiple"):
        lines = get("https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/"
                    f"{meta['datasets']['gorilla-llm/Berkeley-Function-Calling-Leaderboard']}/BFCL_v3_{cat}.json")
        for line in lines.decode().splitlines()[:120]:
            r = json.loads(line)
            q = r["question"][0] if isinstance(r["question"][0], list) else r["question"]
            tools.append({"messages": q, "tools": bfcl_tools(r["function"]), "tool_choice": "auto"})
    kinds["tools"] = tools

    out = []
    for kind, items in kinds.items():
        for item in items:
            effort = rng.choices(("low", "high", "max"), (0.3, 0.3, 0.4))[0]
            body = dict(item, max_tokens=4096, chat_template_kwargs={"reasoning_effort": effort})
            out.append({"kind": kind, "body": body})
    rng.shuffle(out)
    with open(a.out, "w") as f:
        f.write(json.dumps({"_meta": meta, "counts": {k: len(v) for k, v in kinds.items()}}) + "\n")
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({k: len(v) for k, v in kinds.items()}), len(out), "requests ->", a.out, file=sys.stderr)


if __name__ == "__main__":
    main()
