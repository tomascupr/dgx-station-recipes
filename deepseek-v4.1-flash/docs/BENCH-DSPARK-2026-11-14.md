# DSpark draft-length bench, 14 Nov 2026

Real-prompt measurements of DeepSeek-V4.1-Flash (dsv41) speculative decoding on 14 Nov 2026. The morning replay (section 2) ran on the old schedule. A later warm k-sweep (section 3) measured the edit in `serve/serve.sh` and kept it: draft length 5 through batch 8, `[[1,8,5],[9,32,3]]`, which boots as `[[1,1,31],[2,8,5],[9,32,3]]`.

Labels: **[M]** measured that day. The morning estimate in the first note is superseded by section 3 and is not repeated as a result.

## 1. The schedule change

`serve/serve.sh` passes one `--speculative-config`. The only change is the batch schedule:

```
- [[1,4,5],[5,8,3],[9,32,3]]
+ [[1,8,5],[9,32,3]]
```

The rest of the flag is unchanged: method `dspark`, `num_speculative_tokens` 5, `draft_sample_method` `probabilistic`, `rejection_sample_method` `block`.

The drafter is the checkpoint's own MTP layers, run as DSpark. The `suffix` fastpath (`fastpath/fastpath_w9_suffix.py`) rewrites this flag at boot: batch 1 gets K=31 (suffix-copy drafting from the request's own history) and dynamic K of 5 or 7. Batch sizes of 2 and up stay capped at `NORMAL_K=5`, so the usable knob there is k=1..5. With the new flag the fastpath turns the schedule into `[[1,1,31],[2,8,5],[9,32,3]]`. The sweep's boot log confirmed that widened schedule. Before the edit the same rewrite left batches 5–8 at k=3. Only that range changes. Batches 9–32 stay at k=3. There is no data above c=8, so that range stays at k=3. Batch-1 behaviour is a recipe constant, not this flag. That is why the morning replay's mean draft length at one stream is about 7.1, above the flag's k=5.

The c=8 decode shape is 8×6 = 48 rows. 48 is already in this script's `--cudagraph-capture-sizes`.

## 2. Real-prompt replay [M]

The server was the recipe on `vllm/vllm-openai:v0.31.0` with the recipe overlays, started 12 Nov 18:26Z, still on `[[1,4,5],[5,8,3],[9,32,3]]`.

- 1,000 real captured dsv41 requests. Distinct requests, original tools and settings kept. The synthetic load-test traffic was left out.
- Short prompts: 1–8k tokens (pool 1,403, median 4.1k). Long prompts: 20–60k tokens (pool 133, median 35k).
- `max_tokens` 512, streaming. Each cell ran 55 s closed-loop. The long c=4 and c=8 cells ran 40 s and reused the long pool.
- Acceptance is the `/metrics` delta for that cell. The server's request delta matched the client's in every cell, so the cells had no foreign traffic.
- Health was polled every 5 s: 0 non-200 responses, 0 errors, 0 5xx.

Per-stream decode and end-to-end figures are medians, in tokens/s. TTFT is in seconds.

| prompts | c | reqs | per-stream decode p50 (tok/s) | per-stream e2e p50 | aggregate (tok/s) | TTFT p50 / p90 (s) | mean k | accept rate | tokens/step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short 4.1k | 1 | 92 | 632 | 514 | 500 | 0.083 / 0.088 | 7.13 | 47.3% | 4.37 |
| short | 2 | 119 | 416 | 352 | 679 | 0.095 / 0.112 | 5.17 | 54.4% | 3.81 |
| short | 4 | 161 | 302 | 248 | 950 | 0.104 / 0.186 | 5.00 | 56.1% | 3.81 |
| short | 8 | 227 | 191 | 170 | 1,301 | 0.112 / 0.209 | 3.02 | 71.2% | 3.15 |
| long 35k | 1 | 71 | 624 | 443 | 446 | 0.140 / 0.254 | 7.17 | 45.4% | 4.25 |
| long | 2 | 62 | 449 | 329 | 645 | 0.160 / 0.177 | 5.26 | 52.6% | 3.76 |
| long* | 4 | 104 | 294 | 234 | 915 | 0.170 / 0.256 | 5.01 | 55.3% | 3.77 |
| long* | 8 | 140 | 190 | 155 | 1,221 | 0.205 / 0.328 | 3.03 | 69.4% | 3.10 |

\* The long c=4 and c=8 rows reused the long pool, so prefix-cache hits are likely and their TTFT is optimistic. Long TTFT on every cell probably includes prefix or offload cache hits: the usage field does not report cached tokens.

Per-position acceptance for batch ≥ 2 [M]: unconditional 0.826, 0.665, 0.531, 0.426, 0.356 (short, c=4) and 0.834, 0.664, 0.521, 0.415, 0.336 (long, c=4).

### Synthetic prose

The 14 Nov note does not include a synthetic load-test table. The only synthetic figures in it are from the previous night's prose load: about **379 tok/s at c=1**, and **2.65 tokens/step**, against 632 tok/s and 4.37 tokens/step on the real short prompts [M]. Real prompts are faster because the drafter copies repeated context.

### Output equivalence

Output equivalence is unprovable here, and it was not demonstrated. Four prompts at temperature 0, max 128 tokens, hashed over content, reasoning, and tool name and arguments, matched **0/4** even solo against solo, and **0/4** solo against c=4 [M]. The server is nondeterministic at T=0 under this recipe, or the T=0 override is not honoured. Exact string match cannot verify a draft-length change. Speculative decoding with a probabilistic draft and block rejection is distribution-preserving by construction. Verify a k change by acceptance plus a task-success A/B.

## 3. Warm k-sweep [M]

Later the same day the serving recipe was restarted onto each candidate and replayed warm. The winner is the schedule now in `serve/serve.sh`: k=5 for c=2..8, flag `[[1,8,5],[9,32,3]]`, booted as `[[1,1,31],[2,8,5],[9,32,3]]`. The baseline is the previous flag `[[1,4,5],[5,8,3],[9,32,3]]` (k=3 at c=5–8). The rejected arm is k=4 at c=3–8, flag `[[1,2,5],[3,8,4]]`.

The prompt pools are the morning ones: 1,403 short prompts (1–8k tokens, median 4.1k) and 133 long prompts (20–60k, median 35k). Each short cell used the same fixed slice of 250 prompts. Long cells cycled the same pool from the same fixed offset. `max_tokens` 512, streaming, original tools and parameters kept. Short cells ran 50 s and long cells 30 s, closed-loop. Acceptance is the `/metrics` delta. The server's request count matched the client's in every measured cell, except one extra smoke-test request in baseline repetition 1, short, c=2.

The first pass after each restart has an empty prefix and offload cache, and TTFT sat at about 0.18 s even at c=1. Those cold passes are not in the table. Each candidate got a warm-up pass; the numbers below are the warm repetition. The baseline process had already been up 38 h, so both of its repetitions were warm.

Noise floor, baseline repetition 1 against repetition 2 on the same prompts: per-stream decode median ±1–3% at c=6–8 and up to ±5% at c=2–4; aggregate ±0.5–1.1% at c=6–8 (up to 1.2% at c≤4); TTFT p50 ±5–10 ms.

Per-stream is the median decode rate, in tokens/s. Where the baseline has two warm repetitions, both are shown.

| variant (k at c=2..8) | prompts | c | per-stream | aggregate | TTFT p50 (s) | accept | tokens/step |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline `[[1,4,5],[5,8,3]]` rep 1 / rep 2 | short | 1 | 646 / – | 516 / – | 0.083 | 48.5% | 4.57 |
| | short | 2 | 425 / 406 | 672 / 665 | 0.094 | 53.8 / 52.5% | 3.79 / 3.71 |
| | short | 4 | 279 / 294 | 941 / 950 | 0.104 | 56.7 / 56.9% | 3.83 / 3.85 |
| | short | 6 | 218 / 221 | 1101 / 1106 | 0.115 / 0.109 | 70.8 / 70.7% | 3.17 / 3.15 |
| | short | 8 | 186 / 184 | 1295 / 1280 | 0.113 / 0.114 | 70.7 / 69.6% | 3.13 / 3.10 |
| | long | 1 | 660 / – | 473 / – | 0.141 | 46.6% | 4.38 |
| | long | 2 | 442 / 429 | 664 / 656 | 0.152 | 53.3 / 54.1% | 3.89 / 3.81 |
| | long | 4 | 299 / 298 | 885 / 906 | 0.172 / 0.167 | 52.9 / 54.7% | 3.65 / 3.73 |
| | long | 6 | 225 / 218 | 1072 / 1077 | 0.180 | 66.4 / 67.1% | 3.05 / 3.07 |
| | long | 8 | 182 / 188 | 1219 / 1226 | 0.189 / 0.199 | 68.3 / 69.2% | 3.06 / 3.10 |
| **k5 `[[1,8,5]]`** | short | 1 | 662 | 502 | 0.083 | 48.0% | 4.50 |
| | short | 2 | 402 | 649 | 0.093 | 51.6% | 3.66 |
| | short | 4 | 295 | 932 | 0.105 | 55.9% | 3.80 |
| | short | 6 | **229** | **1144** | 0.120 | 57.0% | 3.85 |
| | short | 8 | **197** | **1303** | 0.131 | 55.7% | 3.79 |
| | long | 1 | 628 | 457 | 0.141 | 44.9% | 4.15 |
| | long | 2 | 451 | 677 | 0.148 | 55.8% | 3.97 |
| | long | 4 | 300 | 906 | 0.171 | 54.4% | 3.72 |
| | long | 6 | **242** | **1112** | 0.190 | 55.9% | 3.80 |
| | long | 8 | **203** | **1247** | 0.197 | 54.5% | 3.73 |
| k4 `[[1,2,5],[3,8,4]]` | short | 1 | 669 | 509 | 0.083 | 47.8% | 4.56 |
| | short | 2 | 419 | 676 | 0.092 | 54.4% | 3.80 |
| | short | 4 | 278 | 932 | 0.101 | 63.1% | 3.53 |
| | short | 6 | 225 | 1141 | 0.113 | 63.7% | 3.56 |
| | short | 8 | 189 | 1287 | 0.120 | 62.8% | 3.51 |
| | long | 1 | 649 | 475 | 0.141 | 46.1% | 4.39 |
| | long | 2 | 445 | 663 | 0.148 | 54.4% | 3.92 |
| | long | 4 | 289 | 900 | 0.166 | 59.5% | 3.41 |
| | long | 6 | 227 | 1099 | 0.183 | 60.1% | 3.41 |
| | long | 8 | 197 | 1212 | 0.218 | 58.4% | 3.34 |

**k=5 against the baseline, warm.** Per-stream decode at c=6–8 is **+3–7% on short prompts and +8–11% on long ones**, above the 1–3% noise floor at those concurrencies:

| prompts | c=6 | c=8 |
|---|---|---|
| short | 229 vs 218–221 | 197 vs 184–186 |
| long | 242 vs 218–225 | 203 vs 182–188 |

Aggregate is **+1.2–3.7%**, at or a little above the aggregate noise. TTFT p50 rises by **up to 17 ms** (worst case: short c=8, 0.131 s vs 0.114 s). There is **no regression at c=1–4**. c=1 is outside this flag (dynamic K of 5 or 7, plus the 31-token copy) and did not regress in any variant. c=2–4 is k=5 in both the baseline and the winner.

**k=4 was rejected.** It accepts more often per drafted token and emits fewer tokens per step (3.4–3.6 vs 3.7–3.85). Per stream it is **2–6% slower than k=5** at c=4–8, and its aggregate is from equal to 2.8% lower.

**Pure k=3 at c=2–8 is not bootable.** The flag `[[1,1,5],[2,8,3],[9,32,3]]` dies in the recipe's startup check: `quant_producer fused widths [8]; the recipe captures (6, 8)`. Some batch range has to be k=5 so the 1×6 decode graph is captured. A pure k=4 for c=2–8 fails the same check, which is why k=4 was tested as k=5 at c=2 and k=4 at c=3–8. The k=3 behaviour at c=5–8 is the baseline above. **c≥9 stays at k=3.** Nothing above c=8 was measured.

After the last restart the winner was healthy at 10:57:17 CET: health 200, one completion finished with `stop`, and the vLLM log had no `ERROR` or `Traceback` line.

## 4. Restarts measured during the sweep

Each successful down/up took about 5.5–6 min. None of them triggered a FlashInfer re-autotune. The pure k=3 attempt failed at graph capture and left the server down, so the first k=5 boot includes that gap.

| step | down start (CET) | healthy (CET) | downtime |
|---|---|---|---|
| k=3 attempt (failed at graph capture) | 09:59:06 | – | – |
| k=5 | 10:05:41 | 10:11:09 | 09:59:06 → 10:11:09, 12.0 min including the failed k=3 attempt |
| k=4 | 10:28:43 | 10:34:32 | 5.8 min |
| final k=5 (winner, left up) | 10:51:18 | 10:57:17 | 6.0 min |

The server was unavailable for about 24 min, in three outages.

Rollback is the previous flag. From `deepseek-v4.1-flash/`, restore `[[1,4,5],[5,8,3],[9,32,3]]` in `serve/serve.sh` and run:

```bash
sudo -H ./ds41 down && sudo -H ./ds41 up
```
