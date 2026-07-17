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

---

## CAVEAT that may flip a headline: tau=1.14 might be the RoPE bug, not the drafter (2026-07-17)

The source-level crash diagnosis (`analysis/vllm_2048_bug_diagnosis.md`, independently
re-verified against a fresh v0.24.0 clone) has a consequence beyond the crash: in eager mode
the draft's rope kernel reads **out of bounds without any check**, so every draft token at
position >= 2048 gets garbage cos/sin rotations. That mechanism *predicts* exactly what Phase 3c
measured — healthy tau (2.83–2.88) on short prompts where all positions < 2048, collapsed tau
(1.14) at 7.4k context where nearly all draft positions >= 2048. Outside corroboration
(sgl-project/SpecForge#249): editing a draft checkpoint's `max_position_embeddings` to the
target's value restored normal acceptance lengths for checkpoints with the same 2048 training
artifact.

**RETEST RUN (2026-07-17) — settled, safe to write as a property of this drafter.** Local
draft checkpoint with `max_position_embeddings` patched 2048→8192, `--speculative-config`
pointed at it, compilation ON (no `--enforce-eager`): the crash is gone (confirms the
diagnosis), but tau came back at 1.144 — statistically identical to the original
broken-config measurement, not the healthy ~2.85 short-context reference. Fixing the RoPE
bug did not move tau. **The write-up can now say plainly: "EAGLE-3 (this checkpoint) is
measurably counterproductive at 7.4k-token context — x0.89–0.94 vs no-spec, tau collapses
2.85→1.14 — and this is a real property of the drafter on this workload, not an artifact of
the vLLM crash-workaround."** The two issues are separate: the RoPE/crash bug is still real
and worth reporting upstream (a server should never crash on a long prompt), but it is not
the explanation for the acceptance collapse. Old framing below, superseded, kept for the
record of how the question was resolved:

Retest recipe (15 min GPU, one throwaway config): local copy of
`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` with `config.json` `max_position_embeddings: 2048 -> 8192`,
`--speculative-config` model pointed at the local path, compilation ON, re-run the KS-probe
fp16kv c1 corner. Outcomes:
- **tau recovers to ~2.5+**: headline upgrades to "a one-line checkpoint metadata bug silently
  destroys speculative decoding on long contexts (crash under compile, silent perf collapse
  under eager) — diagnosis, fix, and upstream report." Both the crash AND the perf finding
  resolve to one root cause. QuantSpec juxtaposition (earlier note) softens accordingly: the
  drafter was never given a fair long-context shot, though QuantSpec's structural-acceptance
  argument still stands on its own merits.
- **tau stays ~1.1**: original finding stands untouched, now with the strongest possible
  falsification attempt behind it.

## Quality-side factorial: computed, and it found something (2026-07-17)

`analysis/quality_factorial.py` over all 288 records (`phase3_results/quality_effects.json`):
- **Quality does not compound**: full-stack accuracy delta minus sum-of-mains is within 0.7 pts
  in all 8 workload x concurrency cells. Pairs with the speed side's x1.3–3.0 interference gap
  as one write-up beat: *speed interferes, quality adds.*
- W main: -3.0..-4.0 pts (GSM8K), -6.2..-7.9 pts (HumanEval), every concurrency. S main: |<=0.7|
  pts everywhere (greedy guarantee, now measured as a computed contrast, not a spot check).
- **New robust finding: WK > 0 on HumanEval in all 4 concurrency cells (+1.7..+3.5 pts,
  per-repeat ranges exclude zero; K main there +1.4..+2.1)** — FP8-KV *recovers* part of
  W4A16's code-accuracy damage. Mechanism unresolved: KV rounding acting favorably vs the
  FlashInfer-backend numerics switch that rides along with K (the backend was cleared for
  speed, ~0.2%, but never isolated for quality). Honest write-up framing: a replicated,
  in-sample effect with two candidate mechanisms — do not oversell causally.

---

## Retest verdict + a corrected mechanism claim (2026-07-17)

**The tau collapse is REAL — the RoPE bug did not cause it.** The fixed-checkpoint retest
(draft config `max_position_embeddings` 2048→8192, compilation ON so no out-of-bounds access is
possible) reproduced tau ≈ 1.144 at 7.4k context, statistically identical to the original
eager/stock measurement (1.1441). The previous note's "may flip the headline" scenario did NOT
materialize: "EAGLE-3's acceptance collapses at long context" stands as a drafter property, now
with the strongest falsification attempt behind it (4-cell replication queued for the same
statistical footing as the original; single cell confirmed). The residual open nuance stays the
one already recorded in the diagnosis: the draft config carries no `rope_parameters`/llama3
scaling while the target uses it — noted, unexplored, don't claim beyond the data.

**Corrected along the way (keep the write-up honest):** the diagnosis originally predicted the
eager OOB reads should "silently degrade acceptance." Measured: they didn't — tau identical with
and without the possibility of OOB reads. The eager path DOES dispatch to the unchecked CUDA
kernel (source-verified: `--enforce-eager` → mode NONE → `custom_ops=['all']` per
`vllm/config/vllm.py`, visible in our own eager server log; kernel gather is raw pointer
arithmetic, `csrc/libtorch_stable/pos_encoding_kernels.cu:92-93`) — an undefined-behavior read
worth reporting regardless — but its measured effect on acceptance was nil. A review challenge
claimed eager routes to the checked native path instead; that was traced to reading the
*compiled* server's log (`custom_ops: ['none']`, `enforce_eager=False`) rather than the eager
one (`['all']`, `enforce_eager=True`). Empirical instrumentation (`scripts/debug_rope_oob.py`,
cell added to the retest notebook) will characterize the actual OOB values; the GitHub comment
is on hold until it runs. Write-up moral, worth a paragraph: a mechanistically plausible,
source-cited prediction ("garbage rotations → degraded acceptance") still failed measurement —
cite-and-verify beats cite-and-infer even when the source reading is correct.
