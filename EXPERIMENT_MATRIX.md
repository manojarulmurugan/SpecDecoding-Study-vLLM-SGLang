# EXPERIMENT_MATRIX — Design, Cells, Hypotheses, Baselines, Compute

---

## 1. Core design: a replicated 2³ factorial

Three optimizations = three factors, each off/on:

| Factor | "Off" | "On" |
|---|---|---|
| Weight quant (W) | FP16 | W4A16 (AWQ) |
| KV-cache quant (K) | FP16 KV | FP8 KV |
| Speculative decoding (S) | disabled | EAGLE-3 |

Crossing → **8 configurations** (the 8 corners of the optimization cube), from
`FP16 / FP16-KV / no-spec` (naive baseline) to `W4A16 / FP8-KV / EAGLE-3` (full stack). The
factorial structure yields all 3 main effects, all 3 pairwise interactions, and the 1 three-way
interaction from one clean design — the statistical machinery behind the "compound vs interfere"
claim.

Each config is then run across:
- **Workload (3 primary):** GSM8K, HumanEval, RAG-shared-prefix. (MT-Bench optional 4th.)
- **Concurrency / offered load (4 levels):** {1, 8, 32, 64} concurrent clients. Concurrency is
  SET; **batch size is MEASURED (emergent).**

Core matrix = **8 configs × 3 workloads × 4 concurrency × 3 repeats = 288 runs.**

---

## 2. Why runtime ≪ run count

For a given config, **load the vLLM server once**, run all workloads × concurrency × repeats
against it (36 runs/config). Only 8 server-load events total for the core factorial; the
expensive startup (model load, AWQ setup, EAGLE head, CUDA graph capture) is amortized.

**Estimates (state assumptions; recalibrate after the first real runs):** one benchmark run of
~150–250 requests at fixed concurrency ≈ 1–5 min wall-clock. 36 runs/config ≈ 1.5–2.5 hrs.
- Core 2³ factorial: **~15–20 GPU-hours.**
- Reproduction gate: ~2–3 hrs.
- SGLang RAG seam: ~4–6 hrs.
- **Whole project ≈ 25–30 GPU-hours.** Feasible vs ~550 Colab units IF spent on the right tier
  (see §6). Checkpointed/resumable runs mean a disconnect costs one cell, not one config.

---

## 3. The matrix by block (each block is independently shippable)

| Block | What | Cells | Concurrency | GPU | Est. hrs |
|---|---|---|---|---|---|
| **0 — Reproduction gate** | Validate harness vs SpecMQuant: {FP16, W4A16} × {EAGLE on}, Llama-3-8B, GSM8K + HumanEval, single-stream | ~6–8 | 1 | A100 | 2–3 |
| **1 — Serving baseline** | `FP16/FP16-KV/no-spec` across full sweep (the reference curve) | subset of core | 1,8,32,64 | A100 | (in core) |
| **2–4 — Core 2³ factorial** | 8 configs × 3 workloads × 4 conc × 3 repeats | 288 | 1,8,32,64 | A100 + H100* | 15–20 |
| **5 — SGLang RAG seam** | RAG only: {vLLM, SGLang} × {FP16-KV, FP8-KV} × overlap{low,mid,high}, 2 conc | ~24 | 8, 32 | A100/H100 | 4–6 |
| **6 — Optional** | W8A8/W4A8; MT-Bench; INT8-KV; parallel-drafting micro-seam | variable | — | — | if time |

\*FP8-KV cells (4 of 8 configs) run on H100 for native FP8 — see §6.

---

## 4. Workloads (the instrument, not the subject)

Chosen to span the four axes that drive the interaction, and aligned to Spec-Bench for
comparability.

| Regime | Dataset | Axis exercised | Why included |
|---|---|---|---|
| Math reasoning | **GSM8K** | low acceptance, long CoT, decode-heavy | SpecMQuant reproduction anchor; "spec struggles" end of acceptance spectrum |
| Code generation | **HumanEval** (+MBPP opt.) | high acceptance, templated | "spec shines" end; high-vs-low contrast vs GSM8K; also a SpecMQuant workload |
| Long-context RAG (anchor) | **controlled shared-prefix set** (see HARNESS_SPEC §RAG) | long context, high prefix overlap, prefill-heavy | business anchor (enterprise assistant); where KV-quant matters most; the SGLang seam |
| Multi-turn (optional) | MT-Bench | moderate, multi-turn | rounds out Spec-Bench comparability; only if time |

GSM8K + HumanEval are **non-negotiable** (reproduction anchors). The RAG set has a tunable
**prefix-overlap knob** (questions-per-document ratio) — the control variable for the SGLang
comparison; dial low→high to map where RadixAttention's advantage emerges and whether KV-quant
shifts that crossover.

---

## 5. The hypotheses (this is what makes it research, not benchmarking)

Seven effects from the 2³. Direction stated; **uncertain ones flagged** — those are the
genuinely-unknown findings.

| Effect | Hypothesis | Direction status |
|---|---|---|
| **W main** | Memory ↓ large; decode cheaper; small accuracy cost on GSM8K/HumanEval | KNOWN |
| **K main** | Memory ↓ most in long-context RAG, negligible on short prompts; speed help/neutral on native FP8 | KNOWN (mem) / UNCERTAIN (speed magnitude) |
| **S main** | Large latency win at conc=1, eroding as concurrency ↑ (goodput); high on HumanEval, low on GSM8K | KNOWN dir / UNKNOWN crossover concurrency |
| **W×S** | SpecMQuant's batch-1 "4-bit weights make spec counterproductive" is **amplified under batching** (batching independently pushes compute-bound — same direction) | PREDICTED; "amplified under batching" is novel. HEADLINE CELL |
| **K×S** | QuantSpec's batch-1 "quantized KV raises acceptance/helps" **shrinks or reverses as concurrency ↑** (was it a memory-bound-regime artifact?) | **GENUINELY UNCERTAIN** — most interesting if it flips. HEADLINE CELL |
| **W×K** | Roughly additive on memory, minimal quality interaction — the "boring" control pair | PREDICTED (near-additive). Control that strengthens credibility of interference found elsewhere |
| **W×K×S (three-way)** | Full stack is **sub-additive (interference)** — realized combined win < naive product of individual wins, gap widening with concurrency | PREDICTED sub-additive; magnitude + concurrency-dependence UNKNOWN. **CORE CONTRIBUTION.** Additive result = equally publishable (vindicates practitioners) |

**The through-line:** every interaction is hypothesized concurrency-dependent. The one-sentence
contribution: *"the published batch-1 findings change sign or magnitude under realistic serving
load."*

**SGLang seam hypothesis (Block 5):** On high-overlap RAG, RadixAttention + KV-quant compound on
memory (shared prefix stored once, at reduced precision), BUT quantizing a heavily-reused prefix
incurs a **disproportionate quality cost** (one error propagates to every reusing request) — so
SGLang's quality sensitivity to KV-quant > vLLM's at high overlap. Plus: map the prefix-overlap
crossover where SGLang pulls ahead, and whether KV-quant shifts it. Direction PARTLY UNCERTAIN.

---

## 6. Compute allocation (turn the FP8 problem into clean native data)

H100 is available on Colab Pro — use it deliberately:

- **A100 / L4 (unit-efficient):** reproduction gate, the FP16-KV half of the factorial (4 of 8
  configs), SGLang FP16-KV cells.
- **H100 (native FP8):** the FP8-KV cells (other 4 configs) + SGLang FP8 cells. Native FP8 =
  clean result, no emulation confound, no caveat needed. Reserve scarce H100 units for cells
  that *require* native FP8, not the whole sweep.

**Unit budgeting:** H100 burns units faster than A100, faster than L4. After the first H100
session, READ THE USAGE METER to calibrate units/hour; do not trust an a-priori estimate. FP8
cells ≈ 8–10 H100-hours, comfortably within ~550 units, but verify empirically on run one.

**When to use GCS instead (overflow only):** (a) a run outlasting a reliable Colab session;
(b) H100 units running short while native-FP8 validation still needed (H100/H200 GCS spot on the
$300); (c) the out-of-scope TensorRT-LLM stretch. Colab carries ~90% of the work. **Vertex
avoided.**

---

## 7. Baseline protocol — three layers

### Layer 1 — Single-stream reproduction baseline (CORRECTNESS GATE)
Purpose: prove the harness is correct by reproducing a published result before trusting any
novel number. Source of truth: SpecMQuant (2505.22179) + repo. Reproduce `FP16+EAGLE` and
`W4A16+EAGLE` on Llama-3-8B, EAGLE-2, greedy, A100, GSM8K + HumanEval. Metric: wall-clock
speedup + mean accepted length. **Read exact targets from their tables/repo — never fabricate.**
Tolerance: ±10–15% on speedup AND correct sign of the W4A16-vs-FP16 effect. Document gaps +
likely cause. Build by adapting Spec-Bench + SpecMQuant's quant-wiring (their eval folder is
modified Spec-Bench; GSM8K-eval + evalplus for HumanEval).

### Layer 2 — Serving baseline (foundation of the contribution)
Purpose: the FP16 / no-opt / under-concurrency reference. No published number to match — you
establish it. **Independent variable = offered load** (N concurrent clients, or Poisson rate λ);
**batch size = measured emergent quantity.** Tool: vLLM `benchmark_serving.py`. Fix and record:
model, pinned vLLM version, concurrency sweep {1,8,32,64}, input/output length mode (fixed +
natural), decoding (greedy for controlled cells, sampling temp≈0.7 for realistic cross-check),
warmup (discard first K), ≥3 repeats (report median + spread).

### Layer 3 — Metrics (adopt verbatim — see LITERATURE §7)
TTFT (p50/p95/p99), ITL/TPOT, end-to-end latency, throughput, **goodput**, peak GPU memory; for
spec cells also **average accepted length (τ)** and **verification-to-decoding ratio**; for the
SGLang seam also **cache hit rate**.

### Hardware caveat (foreground as a finding, not a flaw)
FP8 KV is native only on H100+ (SM9.0+); on A100 it is emulated (~10–20% penalty). Primary FP8
cells run on H100 (clean). If any FP8 must run on A100, report the emulation caveat explicitly —
it is a real deployment scenario for A100-class fleets, and the emulation tax is itself a
measurable result.

---

## 8. Reproducibility protocol (enforced from line one)

Every run writes, atomically and immediately, a structured record: full config (model, engine
version, precision settings, draft config, load level), git commit hash, GPU type + driver/CUDA
versions, random seed, raw metric outputs, timestamp. One version-controlled config file per
run. Results in JSONL/parquet so a disconnect never loses a completed cell and every number is
traceable. See HARNESS_SPEC for the schema.

---

## 9. Scope control — kill criteria (pre-committed, so the core always ships)

- **Core (must-ship):** Block 0 + the 8-config factorial on GSM8K+HumanEval+RAG at {1,8,32,64} +
  interaction analysis. This alone is complete and novel.
- **High-value optional:** Block 5 SGLang seam.
- **Low-priority optional:** extra weight precisions, MT-Bench, INT8-KV, parallel-drafting.
- **Kill levers, in pull-order if time compresses:** (1) drop optional precisions + MT-Bench;
  (2) trim concurrency 4→3 levels ({1,8,64}); (3) repeats 3→2 (report spread honestly);
  (4) defer SGLang seam to a "Part 2" follow-up post.
- **Non-negotiable:** the reproduction gate. If Block 0 fails to reproduce SpecMQuant's
  direction within tolerance, STOP and fix the harness before proceeding.
