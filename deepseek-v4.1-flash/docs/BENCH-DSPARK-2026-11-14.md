# DSpark draft-length bench, 14 Nov 2026

Real-prompt replay of DeepSeek-V4.1-Flash (dsv41) speculative decoding, measured 14 Nov 2026, 09:01–09:13 CET, on the schedule this tree used to boot with. The edit in `serve/serve.sh` raises draft length from 3 to 5 for batches 5–8. That gain is a prediction. It has not been measured. Confirm it with the replay in section 4 after the restart.

Labels: **[M]** measured that morning, **[E]** an estimate from those measurements. No figure below was filled in after the fact.

## 1. The schedule change

`serve/serve.sh` passes one `--speculative-config`. The only change is the batch schedule:

```
- [[1,4,5],[5,8,3],[9,32,3]]
+ [[1,8,5],[9,32,3]]
```

The rest of the flag is unchanged: method `dspark`, `num_speculative_tokens` 5, `draft_sample_method` `probabilistic`, `rejection_sample_method` `block`.

The drafter is the checkpoint's own MTP layers, run as DSpark. The `suffix` fastpath (`fastpath/fastpath_w9_suffix.py`) rewrites this flag at boot: batch 1 gets K=31 (suffix-copy drafting from the request's own history) and dynamic K of 5 or 7. Batch sizes of 2 and up stay capped at `NORMAL_K=5`, so the usable knob there is k=1..5. With the new flag the fastpath turns the schedule into `[[1,1,31],[2,8,5],[9,32,3]]`. Before the edit the same rewrite left batches 5–8 at k=3. Only that range changes. Batches 9–32 stay at k=3. Batch-1 behaviour is a recipe constant, not this flag. That is why the replay's mean draft length at one stream is about 7.1, above the flag's k=5.

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

## 3. The k=5 change at batches 5–8 is predicted, and has not been measured

Step time per cell, derived as tokens/step divided by per-stream decode [M]: batch ≈2 at k=5 is 9.2 ms; batch ≈4 at k=5 is 12.6 ms; batch ≈8 at k=3 is 16.5 ms (long: 16.3 ms).

The fit is `t(B,k) = 5.75 + 0.61·B + 0.184·B·(k+1)` ms [E]. That is three points and three unknowns, an exact fit with no residual, so it is an estimate. Expected tokens/step from the c=4 acceptance curve [M]: k=1 → 1.83, k=2 → 2.49, k=3 → 3.02, k=4 → 3.45, k=5 → 3.81.

| batch | current k | predicted per-stream tok/s by k (k=2 / 3 / 4 / 5) | best | gain |
|---|---:|---|---:|---|
| B=1 | 5/7/31 (fastpath DynK + copy) | not a schedule knob; 632 measured | keep | – |
| B=2–4 | 5 | B4: – / 272 / 291 / **302 (measured)** | 5 (cap) | 0 (k>5 blocked by NORMAL_K=5) |
| B=5–8 | 3 | B8: 166 / 183 (meas. 191) / 192 / **196** | **5** | **+4% to +8% per stream [E]**; c=8 aggregate ~1,300 → ~1,350–1,400 [E] |
| B=9–32 | 3 | no data (c>8 not tested) | keep | – |

On the c=8 cell's own, higher, acceptance curve, k=5 gives about 206 tok/s against the measured 191 (+8%) [E].

**Predicted, not measured.** The expected gain is small: **+4–8% per stream at 5–8 concurrent**. Among the cells above, only c=8 changes. The same replay has to be run again after the restart before this edit is treated as a win. Keep it only if that re-run shows it (section 4).

## 4. Restart, rollback, and the post-change replay

This commit does not restart anything. Applying it on a live server needs someone with docker rights (`sudo` needs a password). Confirm the operator DS41 config — served model name and model directory — before running, and use the same `DS41_HOME` and `config.env` as the 12 Nov restore. From `deepseek-v4.1-flash/`:

```bash
sudo -H ./ds41 down && sudo -H ./ds41 up && sudo -H ./ds41 status
```

Rollback on the serving checkout the 14 Nov note prepared: `git switch -` (back to that checkout's previous commit, `7f64c29`), then:

```bash
sudo -H ./ds41 down && sudo -H ./ds41 up
```

`7f64c29` is that serving checkout's commit. It is not a commit in this repository. Here, rollback means restoring the schedule to `[[1,4,5],[5,8,3],[9,32,3]]` and running the same `sudo -H ./ds41 down && sudo -H ./ds41 up`.

Expected downtime is about 4.5 min when the caches are warm. Measured on 12 Nov: container start 18:26:16Z, verified 18:30:40Z. The new decode shape (6 rows for batches 5–8) may trigger a fresh graph capture or a FlashInfer autotune. A first start on a new flag set can take 75–90 min [E]. Plan a 90-minute window.

Post-change check: rerun `replay_live.py --buckets short,long --conc 4,8` (about 4 min). That script is the one that produced the table above; it is not in this repository. Keep the change if per-stream decode at c=8 rises by at least 3% and there are no errors. Otherwise roll back.
