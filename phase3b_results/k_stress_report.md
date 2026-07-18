# K-stress addendum: FP8-KV under capacity pressure

| conc | goodput fp16 | goodput fp8 (ratio) | batch fp16 mean/max | batch fp8 mean/max | kv-usage max fp16/fp8 | preemptions fp16/fp8 | queue-p50 fp16/fp8 | ttft-p95 s fp16/fp8 |
|---|---|---|---|---|---|---|---|---|
| 8 | 221 | 223 (1.01x) | 7.4 / 8 | 7.5 / 8 | 0.43 / 0.22 | 0 / 0 | 0.0 / 0.0 | 2.9 / 3.3 |
| 16 | 260 | 265 (1.02x) | 14.7 / 16 | 14.5 / 16 | 0.86 / 0.43 | 0 / 0 | 0.0 / 0.0 | 6.8 / 7.4 |
| 32 | 249 | 292 (1.17x) | 17.1 / 19 | 28.8 / 32 | 1.00 / 0.85 | 10 / 0 | 14.0 / 0.0 | 21.0 / 16.0 |
| 48 | 250 | 298 (1.19x) | 17.1 / 19 | 33.4 / 38 | 1.00 / 1.00 | 10 / 5 | 30.0 / 11.0 | 35.7 / 28.1 |

**Predicted FP16-KV plateau** (pool 142896 tokens / measured mean context 7688 tokens): **~19 concurrent requests**; FP8-KV doubles it (~37). Compare against the measured batch columns.

## Regime per concurrency (FP16-KV)
- conc 8: FP16-KV not capacity-limited (kv-usage max 0.43, preemptions 0)
- conc 16: FP16-KV not capacity-limited (kv-usage max 0.86, preemptions 0)
- conc 32: FP16-KV CAPACITY-LIMITED (kv-usage max 1.00, preemptions 10)
- conc 48: FP16-KV CAPACITY-LIMITED (kv-usage max 1.00, preemptions 10)

## W capacity channel (AWQ weights, no spec)

AWQ frees ~10.4GB of weight memory -> larger KV pool -> higher sustainable concurrency at the SAME kv dtype. Compare batch/kv-usage against the FP16-weights table above.

| conc | goodput fp16kv | goodput fp8kv | batch fp16kv mean/max | batch fp8kv mean/max | kv-usage max | preemptions |
|---|---|---|---|---|---|---|
| 8 | 242 | 242 | 7.2 / 8 | 7.2 / 8 | 0.28 / 0.14 | 0 / 0 |
| 16 | 254 | 262 | 14.6 / 16 | 14.5 / 16 | 0.55 / 0.28 | 0 / 0 |
| 32 | 236 | 272 | 26.5 / 29 | 28.7 / 32 | 1.00 / 0.55 | 2 / 0 |
| 48 | 237 | 286 | 27.0 / 29 | 41.8 / 48 | 1.00 / 0.82 | 0 / 0 |

## KS long-context probe (EAGLE-3 on, ~7.4k-token contexts)

K-toggle-under-S at long context, same hardware/kernels as the factorial's short-context KS. Short-context reference (factorial @ c1, ~1k contexts): K-under-S ~x0.63, K-solo x0.94, tau invariant. If the long-context ratio here is materially higher than x0.63, context length buys back bandwidth credit; if it matches, the emulation tax dominates regardless of context.

**EAGER-MODE CAVEAT (2026-07-15 crash fix):** these probe cells run with --enforce-eager (vLLM 0.24.0's compiled eagle_head kernels device-assert at this context length; PREREQ 2026-07-15), while every other cell in the project runs compiled. Within-row ratios (fp8kv/fp16kv, both eager) and tau are clean; absolute probe tok/s vs any compiled cell -- including the K-solo column here and the factorial's short-context KS goodput -- is NOT like-for-like. Compare RATIOS across regimes, never raw goodput.

| conc | goodput S+fp16kv | S+fp8kv (K-under-S ratio) | tau fp16kv/fp8kv | K-solo ratio same conc (long ctx) |
|---|---|---|---|---|
| 1 | 34 | 30 (x0.89) | 1.14 / 1.13 | — |
| 8 | 166 | 151 (x0.90) | 1.14 / 1.14 | x1.01 |
