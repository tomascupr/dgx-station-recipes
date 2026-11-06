#!/usr/bin/env python3
"""Send build_mix.py's requests to the server at a fixed concurrency and log what each one cost.

    drive.py MIX.jsonl LOG.jsonl [--base-url URL] [--model NAME] [--key-file FILE] [--conc 32] [--limit N]

Requests go in file order (the mix is already shuffled). One line per finished request in LOG: kind, effort,
prompt/completion tokens, finish reason, seconds, or the error. Ends with a summary line on stdout. Stdlib only.
"""
import argparse
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mix")
    p.add_argument("log")
    p.add_argument("--base-url", default="http://127.0.0.1:8500")
    p.add_argument("--model", default="glm-5.3-flash")
    p.add_argument("--key-file", default="")
    p.add_argument("--conc", type=int, default=32)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    key = Path(a.key_file).expanduser().read_text().strip() if a.key_file else "none"
    lines = open(a.mix).read().splitlines()[1:]
    items = [json.loads(l) for l in (lines[: a.limit] if a.limit else lines)]
    lock = threading.Lock()

    with open(a.log, "a") as out:
        def one(item: dict) -> dict:
            body = dict(item["body"], model=a.model)
            rec = dict(kind=item["kind"], effort=body["chat_template_kwargs"]["reasoning_effort"])
            t0 = time.time()
            try:
                req = urllib.request.Request(a.base_url.rstrip("/") + "/v1/chat/completions",
                                             data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json",
                                                      "Authorization": f"Bearer {key}"})
                d = json.load(urllib.request.urlopen(req, timeout=3600))
                ch = d["choices"][0]
                rec.update(prompt_tokens=d["usage"]["prompt_tokens"], completion_tokens=d["usage"]["completion_tokens"],
                           finish=ch.get("finish_reason"), tool_calls=len(ch["message"].get("tool_calls") or []))
            except Exception as exc:  # recorded per request and counted in the summary
                rec.update(error=f"{type(exc).__name__}: {exc}"[:300])
            rec["s"] = round(time.time() - t0, 1)
            with lock:
                out.write(json.dumps(rec) + "\n")
                out.flush()
            return rec

        t0 = time.time()
        with ThreadPoolExecutor(a.conc) as ex:
            recs = list(ex.map(one, items))
    print(json.dumps(dict(requests=len(recs), errors=sum("error" in r for r in recs),
                          prompt_tokens=sum(r.get("prompt_tokens", 0) for r in recs),
                          completion_tokens=sum(r.get("completion_tokens", 0) for r in recs),
                          wall_s=round(time.time() - t0), conc=a.conc)))


if __name__ == "__main__":
    main()
