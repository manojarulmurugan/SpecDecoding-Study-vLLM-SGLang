# The Decision Guide: When to Stack W, K, and S in vLLM

Prose distillation of this project's measured findings into deployment advice. Every
recommendation cites the finding behind it (bracketed IDs, resolved in the
[Provenance](#provenance) section), and the [last section](#where-the-data-stops) says
where the data stops. The same rules exist as an executable CLI —
`python3 -m analysis.stack_advisor` — whose `--validate` mode recomputes every
quantitative claim below from the raw run records (11/11 checks pass against the
committed results).

**The three levers**, as vLLM 0.24.0 flags on Llama-3.1-8B-Instruct:

| Lever | What it is | vLLM config |
|---|---|---|
| **W** | Weight quantization, AWQ W4A16 | `quantization=awq_marlin` |
| **K** | KV-cache quantization, FP8 (E4M3) | `--kv-cache-dtype fp8` |
| **S** | Speculative decoding, EAGLE-3 | `speculative_config: {method: eagle3, ...}` |

**Scope of every number here:** Llama-3.1-8B-Instruct, vLLM 0.24.0, a single A100
(where FP8 is *emulated*, not native), greedy decoding, closed-loop load, three
workloads (GSM8K reasoning, HumanEval code, shared-prefix RAG). Outside that envelope
you are extrapolating — see [Where the data stops](#where-the-data-stops).

---

## The 90-second version

| Your situation | Recommended stack | Measured basis |
|---|---|---|
| Short context, low concurrency (1–8), quality-tolerant | **W + S** (leave K off) | x3.0 GSM8K/RAG, x5.2 HumanEval at conc 1 — the best measured stack in the study [P2-W-c1, P2-S-c1, F3-KS] |
| Short context, low concurrency, quality-sensitive | **S only** | x1.9–3.2, bit-identical output under greedy [P2-S-c1, QUAL-S] |
| Short context, high concurrency (32+), no KV pressure | **Nothing** (code workloads: keep S) | W and S both fade to ~x1.0 or below by conc 64; HumanEval's S holds x1.37 [P2-S-cross, F3-W-rev] |
| High concurrency **with** KV-capacity pressure | **K** (+ W if quality-tolerant) | K doubles the admitted batch, +19% goodput, −21% TTFT p95; W raises the admission ceiling ~17→~27 [3b-K-cap, 3b-W-cap] |
| Long context (~4k+ tokens per request) | **Turn S off**; K is ~neutral; W if quality-tolerant | S is a measured net *loss* at 7.4k context: x0.89–0.94 [D2-S-long] |
| Quality-sensitive, any regime | **Never W**; K and S are ~free | W is the only lever with a measured accuracy price: −3 to −8 pts [QUAL-W, QUAL-K, QUAL-S] |

**Three stacking rules that always apply:**

1. **Never multiply the levers' individual speedups.** The measured full stack
   underperforms the naive product of marginals by x1.30–2.97, worst at batch 1
   [F3-GAP]. All pairwise interactions are negative in all 12 measured
   workload × concurrency cells.
2. **Quality costs do NOT compound.** The full stack's accuracy loss equals the sum of
   the individual levers' losses to within 0.7 points in all 8 measured cells — which
   in practice means the stack's quality price ≈ W's price alone [F3-GAP]. Speed
   interferes; quality adds.
3. **On FP8-emulating GPUs (A100 and older), don't pair K with S at low concurrency.**
   K under S is x0.63 at short context, x0.89 at long — K's emulation tax multiplies
   with S's extra KV traffic [F3-KS].

---

## The three levers at a glance

| | Best case (measured) | Worst case (measured) | Quality price | One-line mechanism |
|---|---|---|---|---|
| **W** (AWQ W4A16) | x1.70–2.13 at conc 1–8 [P2-W-c1] | x0.90–0.92 at conc 64, GSM8K/RAG [F3-W-rev] | **−3 to −4 pts GSM8K, −6 to −8 pts HumanEval** [QUAL-W] | Quarters weight bytes; wins while decode is bandwidth-bound, becomes dequant overhead when compute-bound |
| **K** (FP8 KV) | x1.17–1.19 + 2x admitted batch at capacity [3b-K-cap] | x0.63 under S at low concurrency [F3-KS] | ~0 (slightly *positive* on HumanEval) [QUAL-K] | Halves KV bytes; pays a ~5% emulation tax on A100 until KV capacity is the binding constraint |
| **S** (EAGLE-3) | x1.90–3.16 at conc 1, short context [P2-S-c1] | x0.89–0.94 at 7.4k-token context — a net loss [D2-S-long] | ~0 under greedy (bit-identical, measured) [QUAL-S] | Drafts tokens with a small head; wins while acceptance is high and spare compute exists, loses when either goes |

---

## Recommendations by scenario

### Short context (≤ ~2k tokens per request), low concurrency (1–8)

This is S's and W's home regime: single-stream decode is memory-bandwidth-bound, so
quartering weight bytes (W) and amortizing weight reads over speculated tokens (S)
both pay off close to their full potential.

- **S: ON.** x2.13 (GSM8K), x3.16 (HumanEval), x1.90 (RAG) at conc 1; still
  x1.61–2.72 at conc 8. The speedup tracks the drafter's acceptance length τ
  (2.85 / 4.09 / 2.52 respectively) — code is speculative decoding's best workload
  [P2-S-c1]. Under greedy decoding this is quality-free: measured bit-identical
  accuracy spec-on vs spec-off [QUAL-S].
- **W: ON if you can pay the quality price.** x1.70–2.13 solo [P2-W-c1] — but W is
  the only lever that measurably costs accuracy: main effect −3.0 to −4.0 points on
  GSM8K and −6.2 to −7.9 points on HumanEval at every concurrency; toggling W alone
  costs up to −10 points on HumanEval [QUAL-W].
- **K: OFF.** On A100 it's a pure x0.94–0.98 tax below the capacity knee [P2-K-tax],
  and it actively erodes S (x0.63) [F3-KS].

**The best measured stack in the entire study is W + S here**: x3.01 (GSM8K),
x5.23 (HumanEval), x2.98 (RAG) at conc 1, measured directly in the factorial. Note
what interference does even to the winner: the naive product of W and S marginals
predicts x4.4–6.7; you get x3.0–5.2 [F3-WS, F3-GAP]. Adding K to make it the "full
stack" cuts it roughly in half: x1.40 / x2.28 / x1.38 [F3-KS].

### Short context, high concurrency (32+), KV cache *not* the constraint

Continuous batching saturates the GPU's compute; the bandwidth savings that powered
W and S at low concurrency no longer buy anything.

- **S: OFF for reasoning/RAG-shaped workloads.** The speedup crosses below x1.0 at
  conc 32–64 (GSM8K x0.97, RAG x0.90 at conc 64). Crucially, τ is *flat* in
  concurrency — the erosion is economics (no spare compute to speculate with), not
  acceptance [P2-S-cross]. **Exception — code workloads: keep S.** HumanEval's
  τ≈4.1 keeps S at x1.37 even at conc 64.
- **W: OFF.** Solo speedup fades to x1.00–1.18 by conc 64 [P2-W-c1], and the
  factorial's W main effect *reverses* to x0.90–0.92 on GSM8K/RAG — Marlin dequant
  overhead in a compute-bound regime is a net loss [F3-W-rev]. You'd be paying
  −3 to −8 accuracy points for nothing [QUAL-W].
- **K: OFF.** Still just the emulation tax if capacity isn't binding [P2-K-tax].

### High concurrency at KV-capacity pressure (the regime where K earns its keep)

When `concurrency × (context + output tokens)` approaches the KV pool, the server
stops admitting requests: batch plateaus, preemptions appear, queues grow. Measured
on A100-40GB with ~7.7k-token contexts (FP16-KV pool ≈ 143k tokens, plateau ≈ 17–19
admitted requests):

- **K: ON — this is the headline K result.** FP8-KV doubles the admitted batch
  (~17 → ~33–42), which converts to +17–19% goodput, TTFT p95 −21% (35.7s → 28.1s),
  and queue p50 collapsing 30s → 11s at conc 48 [3b-K-cap]. The ~5% tax is still
  being paid; it's just dominated by admission.
- **W: ON if quality-tolerant — as a *capacity* lever, not a speed lever.** AWQ
  frees ~10.4GB of weights, growing the KV pool so the admission ceiling rises
  ~17 → ~27 at the same KV dtype (measured plateau vs. predicted ~26) [3b-W-cap].
  Its throughput ratio stays ~x1.0; what you're buying is admitted requests.
- **W + K stack cleanly here**: freed weight memory and cheaper KV bytes attack the
  same pool from both sides (measured max batch 41.8 at conc 48 with both on).
- **S: only if context is short** — see the next section if it isn't.

### Long context (~4k+ tokens per request)

The study's sharpest negative result lives here.

- **S: OFF.** At 7.4k-token RAG contexts EAGLE-3 is measurably **counterproductive**:
  x0.94 at conc 1, x0.89 at conc 8 against a no-spec baseline in the same regime.
  The mechanism is an acceptance cliff: τ collapses from 2.85 (short context) to
  1.14 — with 5 draft tokens per round, ~77% of draft compute is discarded
  [D2-S-long]. This is a *drafter* property (out-of-distribution on long unique
  documents), not an engine artifact: it survived a fixed-checkpoint retest with
  compilation on, replicated 4/4 with τ = 1.138–1.144, and holds in both eager and
  compiled regimes (x0.75 supporting point, compiled). A drafter trained on
  long-context data could change this verdict; this checkpoint doesn't.
- **K: ~neutral on speed, unmeasured on quality.** At 7.4k context below the
  capacity knee K is ~x1.01 — the growing KV read stream earns back the emulation
  tax [3b-K-long]. But note: K's ~zero *accuracy* cost is measured at short context
  only [QUAL-K].
- **W: ON if quality-tolerant**, same accuracy caveat as everywhere [QUAL-W]. Long
  contexts also reach KV pressure sooner, where W's capacity channel helps [3b-W-cap].

### Quality-sensitive deployments (any regime)

The three levers do **not** cost the same thing on the quality axis, and the
asymmetry is stark [QUAL-W, QUAL-K, QUAL-S]:

- **S is free** under greedy decoding — bit-identical output, measured and
  theoretically guaranteed. (T>0 sampling: not measured.)
- **K is free** at short context — GSM8K main effect straddles zero at every
  concurrency; HumanEval is slightly *positive* (+1.4 to +2.1 pts).
- **W is not free** — −3 to −4 pts GSM8K, −6 to −8 pts HumanEval, every concurrency.

So: **reach for S and K first; W buys the most speed at low concurrency, but it is
the only lever you pay for in correctness.** One replicated curiosity: FP8-KV claws
back +1.7 to +3.5 points of W's HumanEval damage (WK interaction positive in all
4 concurrency cells) — mechanism unresolved (KV rounding vs. the attention-backend
switch that rides along with K), so treat it as an observed offset, not a promise
[QUAL-WK].

### Native-FP8 GPUs (H100 and newer) — extrapolation, clearly labeled

Every K penalty in this study is A100 FP8-*emulation* tax. On hardware with native
FP8, the tax's cause disappears, so K's ledger should improve across the board —
plausibly free-to-positive even below the capacity knee, and the "never K under S at
low concurrency" rule may soften [HW-FP8]. **This project measured A100 only.** The
capacity-doubling result [3b-K-cap] is the one K finding that should transfer
directly, since it's about bytes, not throughput.

---

## Why: the mechanisms in plain language

**Why W fades and then reverses.** Low-batch decode is memory-bandwidth-bound —
each generated token requires streaming all the weights through the GPU, so
quartering weight bytes nearly doubles speed. As the batch grows, the same weight
read is amortized over more tokens and the GPU becomes compute-bound; W's bandwidth
win vanishes (~x1.0 at conc 64) and its dequantization overhead turns it into a
small net loss on some workloads (x0.90–0.92) [P2-W-c1, F3-W-rev].

**Why K is a tax until it's a doubler.** FP8-KV halves KV-cache bytes. Below
capacity, on an A100, you pay ~5% for emulated FP8 arithmetic and the halved bytes
buy little (at short context the KV stream is small next to the weights). At
capacity, halved bytes mean **twice as many requests fit**, and admission — not
per-token speed — is what's throttling goodput, TTFT, and queueing [P2-K-tax,
3b-K-cap]. K is best understood as a capacity lever with a small speed tax.

**Why S erodes with concurrency without τ moving.** Speculative decoding spends
extra compute (draft + parallel verify) to save serial steps. At low concurrency the
GPU has idle compute to spend — the trade is nearly free. Under saturation, draft
compute comes out of the same budget serving other requests; τ stays flat (the
drafter is as accurate as ever) but the economics flip [P2-S-cross]. The crossover
is workload-dependent because it tracks τ: HumanEval (τ≈4.1) stays profitable to
conc 64, GSM8K/RAG (τ≈2.5–2.9) cross below x1.0 at conc 32–64.

**Why S falls off a cliff at long context.** τ is a measure of how well the drafter
predicts the target on *this* input distribution. On 7.4k-token unique documents,
this EAGLE-3 checkpoint's τ collapses to 1.14 — the drafter proposes 5 tokens per
round and ~77% of that work is thrown away, making S a net drag [D2-S-long]. This
is an acceptance cliff, not a compute cliff, and it was confirmed drafter-real by
the strongest falsification attempt available (see the [bug
diagnosis](analysis/vllm_2048_bug_diagnosis.md) for the full story of separating
this from a genuine vLLM crash bug found along the way).

**Why the speedups don't multiply.** All three levers compete for the *same* two
resources: memory bandwidth (W and K both shrink bytes; S amortizes byte-reads) and
spare compute (S spends it; W's dequant and K's emulation consume it). Stacking
levers means each one operates in the regime the others have already improved —
the second lever's "before" is the first lever's "after". The measured gap between
naive-product and reality is x1.30–2.97, worst exactly where each lever alone is
strongest (batch 1) [F3-GAP].

**Why quality costs DO add.** Quality damage flows through a different channel than
speed: W perturbs the weights themselves, K perturbs attention reads, S (greedy)
provably changes nothing. These perturbations are independent enough that the full
stack's accuracy delta equals the sum of mains within 0.7 points in all 8 measured
cells [F3-GAP]. Practical consequence: you can predict the stack's quality from the
levers' individual report cards — but never its speed.

**One stacking interaction has an acceptance channel — W on reasoning workloads.**
W drops τ by 14% on GSM8K (quantized weights change the target's distribution enough
to hurt the drafter's hit rate on chain-of-thought text) but not on HumanEval or RAG.
K never touches τ anywhere — QuantSpec's "quantized KV raises acceptance" channel
simply does not exist for EAGLE-3, whose drafter doesn't read the KV cache being
quantized [F3-WS, F3-KS].

---

## Provenance

Every bracketed ID above is a finding with a claim and a committed data source.
`python3 -m analysis.stack_advisor --list-findings` prints the same table;
`--validate` recomputes the quantitative claims from the raw per-run JSON records:

```
python3 -m analysis.stack_advisor --validate phase3_results phase3b_results phase3c_diagnostics_results
# 337 records -> 11/11 PASS
```

| ID | Finding (short form) | Source |
|---|---|---|
| P2-W-c1 | W solo x1.70–2.13 at conc 1–8, fading to ~x1.0 by conc 64 | [phase2_marginals_report.md](phase2_results/phase2_marginals_report.md) |
| F3-W-rev | W main effect reverses at conc 64 (x0.90–0.92, GSM8K/RAG) | [factorial_report.md](phase3_results/factorial_report.md) |
| 3b-W-cap | AWQ raises the admission ceiling ~17 → ~27 at 7.7k contexts | [k_stress_report.md](phase3b_results/k_stress_report.md) |
| P2-K-tax | K solo x0.94–0.98 below the capacity knee (A100 emulation tax) | [phase2_marginals_report.md](phase2_results/phase2_marginals_report.md) |
| 3b-K-cap | K doubles admitted batch at capacity; +17–19% goodput, TTFT p95 −21% | [k_stress_report.md](phase3b_results/k_stress_report.md) |
| 3b-K-long | K solo ~x1.01 at 7.4k context below the knee | [k_stress_report.md](phase3b_results/k_stress_report.md) |
| P2-S-c1 | S solo x1.90–3.16 at conc 1, tracking τ per workload | [phase2_marginals_report.md](phase2_results/phase2_marginals_report.md) |
| P2-S-cross | S crosses below x1.0 at conc 32–64 (except code); τ flat in concurrency | [phase2_marginals_report.md](phase2_results/phase2_marginals_report.md) |
| D2-S-long | S counterproductive at 7.4k context (x0.89–0.94; τ 2.85 → 1.14); drafter-real, 5+ measurements | phase3c diagnostics + KS-probe + [retest](phase3c_retest_results_full/) |
| F3-WS | W erodes S at every concurrency (x0.78–0.83); W drops τ 14% on GSM8K | [factorial_report.md](phase3_results/factorial_report.md) |
| F3-KS | K erodes S: x0.63 short / x0.89 long at conc 1; τ invariant under K | [factorial_report.md](phase3_results/factorial_report.md) + KS-probe |
| F3-GAP | Interference gap x1.30–2.97; quality compounding excess ≤ 0.7 pts | [factorial_report.md](phase3_results/factorial_report.md) + [quality_effects.json](phase3_results/quality_effects.json) |
| QUAL-W | W costs −3/−4 pts GSM8K, −6/−8 pts HumanEval (only lever with a quality price) | [quality_effects.json](phase3_results/quality_effects.json) |
| QUAL-K | K accuracy cost ~0 at short context (HumanEval slightly positive) | [quality_effects.json](phase3_results/quality_effects.json) |
| QUAL-WK | FP8-KV offsets +1.7 to +3.5 pts of W's HumanEval damage (mechanism open) | [quality_effects.json](phase3_results/quality_effects.json) |
| QUAL-S | S quality-free under greedy (measured bit-identical) | phase3 runs + [tests/test_repro_gate.py](tests/test_repro_gate.py) |
| HW-FP8 | Native-FP8 GPUs should erase K's tax — extrapolation, not measured | [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md) |

---

## Where the data stops

Honest boundaries. Everything above is measured **inside** this envelope; outside
it, you're extrapolating:

- **One model, one size**: Llama-3.1-8B-Instruct. Larger models shift the
  weight-bytes/KV-bytes ratio that drives W's and K's economics.
- **One engine version**: vLLM 0.24.0, pinned. Two real engine bugs were found at
  this version (documented in [analysis/vllm_2048_bug_diagnosis.md](analysis/vllm_2048_bug_diagnosis.md),
  reported upstream as [vllm#48894](https://github.com/vllm-project/vllm/issues/48894));
  later versions may move numbers.
- **A100 only** — all K findings carry the FP8-emulation tax; native-FP8 hardware
  is an extrapolation [HW-FP8].
- **Greedy decoding only** — S's quality-free guarantee is greedy-specific; T>0
  sampling is unmeasured.
- **Long-context quality under K is unmeasured** — K's ~zero accuracy cost is a
  short-context result; the long-context cells measured speed only.
- **One EAGLE-3 checkpoint** (`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`) — the
  long-context acceptance cliff is a property of this drafter's training
  distribution; a long-context-trained drafter could erase it.
- **The WK quality offset's mechanism is unresolved** — replicated in-sample, but
  not causally isolated from the attention-backend switch that accompanies K.
