# WRITEUP_NOTES — running material for the write-up series

Findings, framings, and explanations worth carrying into the LinkedIn/Medium write-up series
(README/IMPLEMENTATION_PLAN deliverable), captured as they surface during analysis rather than
reconstructed from memory afterward. Not a draft — raw material and the reasoning behind it.

---

## QuantSpec vs. EAGLE-3: two different mechanisms, not two points on one scale (2026-07-15)

**Why this matters for the write-up:** it explains *why* QuantSpec and this project's EAGLE-3
results disagree without either being wrong, and it gives the τ≈1.14 long-context anomaly
(Phase 3b KS-probe) a clean, mechanistic home instead of reading as an unexplained wrinkle.

**The core correction:** QuantSpec's "draft model" is not a smaller model. Verified directly
against the paper (arXiv 2502.10424): the draft pass runs the *same* weights through *all*
transformer layers — identical FLOPs to a normal step. The only thing that shrinks is the KV
cache's bit-depth during that step (a single physical cache stored as 8 bits, split into an
upper 4-bit "coarse" read for drafting and a lower 4-bit residual added back in for verification
— one stored value, read at two resolutions, not two separate caches).

**Why that still produces a speedup:** low-batch LLM decoding is memory-bandwidth-bound, not
compute-bound — each step's latency is dominated by reading (a) the model weights (fixed cost)
and (b) the KV cache (grows with context). QuantSpec attacks only (b), so it only helps when the
cache is large relative to the weights — i.e., very long context (their reported numbers: 1.61×
at 8k tokens, 2.49× at 128k, batch=1 only). It saves zero compute, so it buys nothing once the
GPU becomes compute-bound (i.e., under real concurrency).

**EAGLE-3 attacks the other pile:** a genuinely tiny, separately-trained head does a fraction of
the FLOPs and barely touches the weight-read cost, working across a much broader range of
context lengths and concurrency levels — the right generalist tool for a *serving-under-load*
study. QuantSpec is a batch-1, long-context, edge-inference specialist.

**No published head-to-head exists.** Checked QuantSpec's own tables directly — it's compared
only against sparse-KV baselines (StreamingLLM, SnapKV), never against EAGLE/EAGLE-2/3.
Reasoned expectation, not measured: QuantSpec likely wins at its own regime (single request,
very long context) precisely because its acceptance rate is a *structural* guarantee (same
weights reading the same cached value at two resolutions — near-impossible to disagree with
itself), not something a separately-trained small network has to learn to generalize. That's
exactly the failure mode this project's own data hit: EAGLE-3's τ collapses to ~1.14 at 7.4k
tokens (Phase 3b KS-probe) — the drafter is barely useful there. QuantSpec's design sidesteps
that failure mode by construction. Our long-context acceptance collapse is, in effect,
independent evidence for the problem QuantSpec exists to solve.

**Why we didn't switch / build it:** implementing QuantSpec's actual mechanism (one physical
cache read at two bit-depths) isn't a vLLM config flag — vLLM doesn't ship it as a built-in
method. Building it would be new serving-engine methods work, which conflicts with
PROJECT_SPEC.md's explicit framing ("not a new method... no claim beats anyone's benchmark").
Good material for a "future work" / discussion-section paragraph, not something to retrofit into
the current dataset.

**Write-up framing suggestion:** pair this with the τ=1.14 anomaly as one beat — "two papers,
two different bets on where LLM decoding's cost lives (compute vs. bandwidth), and our data
shows both bets paying off in their own narrow regime and failing outside it."

---

## Phase-3c verdicts: two headline candidates + two cleared confounds (2026-07-16)

**Headline candidate 1 — "speculative decoding has a context-length cliff, and it's an
acceptance cliff, not a compute cliff":** tau collapses 2.85 → 1.14 when the same EAGLE-3
head moves from ~1k-token conversational prompts to 7.4k-token unique RAG documents
(eager-mode confound experimentally excluded: tau=2.83–2.88 eager short-context). With 5
draft tokens per round, tau=1.14 means ~77% of drafted tokens are thrown away — and the
measured consequence is that S is net-NEGATIVE at long context at every concurrency tested
(x0.94 at c1, x0.89 at c8, no-spec eager baseline vs S-on, same regime). Pairs with the
QuantSpec-mechanism note above: this is the exact regime QuantSpec's structural-acceptance
design exists for.

**Headline candidate 2 — the decision guide's S rule is now context-conditional, measured
on both sides:** short context, low concurrency: S is the best quality-free lever
(x1.27–2.11 at c1, workload-dependent via tau). Long context: turn S off. Crossovers in
BOTH dimensions (concurrency AND context length) are now measured, not extrapolated.

**Cleared confound 1:** attention-backend switch inside K comparisons ≈ 0.2% (FLASHINFER
pinned on FP16-KV: 220.6 vs 221 tok/s reference). One sentence in methods, then never
worry again.

**Cleared confound 2 (negative result worth one line):** the vLLM 0.24.0 EAGLE-3
long-context crash is NOT CUDA-graph capture — same Triton assert with graphs off,
compile on; kernel bound stays 2048 even when the token budget is 8192. It's an
inductor-compiled eagle_head kernel bug; eager is the only working mode in 0.24.0.
Upstream issue material, and an honest "what it cost us" beat for the debugging post
(the 8 probe cells run eager; ratios/tau comparable, absolute tok/s not).
