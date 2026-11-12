import json, sys, time, urllib.request
qs = ["Explain in three sentences why the sky is blue.",
      "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
      "A train leaves at 14:10 and arrives at 17:45. How long is the journey? Show the calculation.",
      "Translate to Czech: 'The model runs on a single workstation.'",
      "Write a Python function that checks whether a string is a palindrome, ignoring case and punctuation."]
for q in qs[: int(sys.argv[1]) if len(sys.argv) > 1 else len(qs)]:
    t0 = time.time()
    r = urllib.request.Request("http://127.0.0.1:8011/v1/chat/completions", json.dumps({"messages": [{"role": "user", "content": q}], "max_tokens": 400, "temperature": 0}).encode(), {"Content-Type": "application/json"})
    j = json.load(urllib.request.urlopen(r, timeout=3600))
    m = j["choices"][0]["message"]
    print(f"\n### {q}\n[{j['usage']['completion_tokens']} tokens in {time.time() - t0:.0f}s, finish {j['choices'][0]['finish_reason']}]\nTHINK: {m.get('reasoning_content', '')[:300]}\nANSWER: {m['content'][:900]}", flush=True)
