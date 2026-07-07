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
PREREQ_RESULTS Check 6 measured this directly: ~172s per server launch (load + torch.compile +
spec-decoder graph capture + warmup) on A100 — a real, confirmed number, not an estimate.

**`num_requests` must scale with concurrency, not be fixed.** A single fixed count (e.g. 200)
is wrong at both ends: at concurrency=1 it's fully sequential (200 requests × ~10s/request at a
plausible 40–60 tok/s single-stream decode ≈ 15–35 min, not 1–5 min), and at concurrency=64,
200 requests is only ~3 batch "waves" — not enough to reach steady state. Scale roughly
~60–80 requests at conc=1 up to ~400+ at conc=64, targeting ≥2–3 minutes of steady-state
measurement per cell.

**Revised estimate, recalibrate after the first real runs:**
- Core 2³ factorial (all A100, see §3/§6): **~30–45 GPU-hours** once request counts are scaled
  correctly and dev/debug time is included.
- Reproduction gate: ~2–3 hrs.
- SGLang RAG seam: ~4–6 hrs.
- **Whole project ≈ 40–55 GPU-hours ≈ 470–650 units** against a confirmed **500-unit budget**
  (PREREQ_RESULTS Check 1: 200 + 300 across the two accounts). This is at-or-over budget —
  treat the kill levers (§9) as a near-certain requirement, not a contingency, and pull the
  first one (extra precisions/MT-Bench) proactively rather than waiting to run out.
- **Do development/debugging on L4, not A100.** You cannot build or debug the vLLM harness
  (EAGLE config strings, OOMs, adapter issues) without a GPU attached, and this routinely costs
  as much GPU time as the planned measurement runs. Llama-3.1-8B FP16 fits in L4's 24GB at low
  concurrency; AWQ fits easily. Reserve A100 hours for actual measurement.

---

## 3. The matrix by block (each block is independently shippable)

| Block | What | Cells | Concurrency | GPU | Est. hrs |
|---|---|---|---|---|---|
| **0 — Reproduction gate** | Validate harness vs SpecMQuant: {FP16, W4A16} × {EAGLE on}, Llama-3-8B, GSM8K + HumanEval, single-stream | ~6–8 | 1 | A100 | 2–3 |
| **1 — Serving baseline** | `FP16/FP16-KV/no-spec` across full sweep (the reference curve) | subset of core | 1,8,32,64 | A100 | (in core) |
| **2–4 — Core 2³ factorial** | 8 configs × 3 workloads × 4 conc × 3 repeats — **all 8 configs on A100** | 288 | 1,8,32,64 | A100 | 30–45 |
| **4b — H100 native-FP8 validation (optional, bonus)** | A small internally-consistent sub-factorial re-running the K=on corners (and their K=off partners) on H100 if sessions materialize, to show the native-FP8 picture alongside the A100-emulated core | ~8 | 1, 32 | H100 | if available |
| **5 — SGLang RAG seam** | RAG only: {vLLM, SGLang} × {FP16-KV, FP8-KV} × overlap{low,mid,high}, 2 conc | ~24 | 8, 32 | A100 | 4–6 |
| **6 — Optional** | W8A8/W4A8; MT-Bench; INT8-KV; parallel-drafting micro-seam | variable | — | — | if time |

**GPU routing correction (was a real design flaw, caught before any GPU time was spent on it):**
the original plan split the core factorial across A100 (FP16-KV configs) and H100 (FP8-KV
configs). That confounds the K factor with hardware — every K main effect and every interaction
containing K (K×S, W×K, the three-way) would measure "quantization effect + GPU difference,"
not quantization alone. **The entire core 2³ factorial now runs on A100**, accepting
software-emulated FP8-KV as a real, documented deployment condition (this is itself a finding,
not a caveat to apologize for — see §7). If H100 sessions become available (Colab H100 is
opportunistic per PREREQ_RESULTS Check 1), run Block 4b as a separate, internally-consistent
mini-factorial for the native-FP8 picture — never mixed into the core routing.

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

**SGLang seam hypothesis (Block 5) — revised.** The original quality-propagation hypothesis
("one KV-quant error propagates to every request reusing a shared prefix") does not survive
scrutiny: FP8 quantization with a fixed scale is a **deterministic** function of the input
bytes, so identical prefix bytes quantize to identical values whether computed once and shared
or recomputed independently per request. Sharing changes compute cost, not whether a
quantization error occurs — the predicted "disproportionate quality cost" is a near-guaranteed
null for a reason derivable on paper, which makes it a weak headline. **Dropped.**

**Salvaged hypothesis:** KV-quant doubles effective prefix-cache capacity (half the bytes per
cached token), so it should **shift the overlap-crossover point and cache hit rate under memory
pressure** — a real, measurable, performance-side interaction, comparable across both engines.
Map the prefix-overlap crossover where SGLang pulls ahead of vLLM, and whether KV-quant shifts
that crossover. This is the Block 5 hypothesis now; direction UNCERTAIN (genuinely interesting
either way).

**Confound to control, not ignore:** vLLM has had Automatic Prefix Caching (APC) on by default
since v0.6.0 — this is a real, currently-shipping feature, not a hypothetical. The vLLM-vs-SGLang
contrast in Block 5 is therefore APC-hash-block-vs-radix-tree, not "sharing vs no sharing." APC
on/off must be an explicit, recorded setting per RAG cell (HARNESS_SPEC config schema) or the K
effect in the RAG workload is contaminated by an unrecorded variable.

---

## 6. Compute allocation (500 confirmed units, GCS as real overflow)

- **A100 (primary, ~90% of the work):** reproduction gate, dev/debug (see §2), and **the entire
  core 2³ factorial including FP8-KV cells** (emulated on A100, ~10–20% penalty — a documented
  finding, not a flaw; see §7 and the GPU-routing correction in §3). SGLang seam also runs on
  A100 for the same confound-avoidance reason.
- **L4:** harness development and debugging only — never counted against measurement budget.
- **H100 (opportunistic bonus only):** PREREQ_RESULTS Check 1 confirmed H100 is selectable in
  Colab's UI but not reliably obtainable — it can silently fall back to a lesser GPU even when
  selected. Do not plan any core-path cell around it. If a session materializes, use it for
  Block 4b (§3) — a small, separate, internally-consistent native-FP8 validation — never as a
  substitute for an A100 cell in the core routing.
- **GCS (paid, real overflow — account already upgraded, quota requested):** for (a) a run
  outlasting a reliable Colab session, (b) A100 units running short, (c) the out-of-scope
  TensorRT-LLM stretch. Request 1x A100 quota as the primary ask (see PREREQ_RESULTS Check 4);
  treat GCS H100 as harder to actually schedule even once quota is approved, same caveat as
  Colab H100. **Vertex avoided.**

**Unit budgeting:** confirmed 500 units total (PREREQ_RESULTS Check 1: 200 + 300 across two
accounts) against a revised estimate of ~470–650 units needed (§2) — budget is tight, not
comfortable. Calibrate actual burn rate empirically (run a real workload for a fixed duration,
read the units delta) before trusting any further estimate in this document.

---

## 7. Baseline protocol — three layers

### Layer 1 — Single-stream reproduction baseline (CORRECTNESS GATE)
Purpose: prove the harness is correct by reproducing a published result before trusting any
novel number. Source of truth: SpecMQuant (2505.22179) + repo. Reproduce `FP16+EAGLE` and
`W4A16+EAGLE` on Llama-3-8B, EAGLE-2 (checkpoint: `yuhuili/EAGLE-LLaMA3-Instruct-8B`, confirmed
identical to SpecMQuant's own — PREREQ_RESULTS Check 3), greedy, A100, GSM8K + HumanEval. Metric:
wall-clock speedup + mean accepted length. **Read exact targets from their tables/repo — never
fabricate.**

**Tolerance, revised:** the original ±10–15% assumed SpecMQuant was a vLLM harness. It isn't —
PREREQ_RESULTS Check 5 confirmed it's bespoke C/CUDA. A cross-engine speedup-ratio match within
15% is optimistic, and since this gate is a hard STOP, a spuriously-failed gate stalls the whole
project over an apples-to-oranges comparison. **Gate on sign and ordering** (W4A16+EAGLE speedup
< FP16+EAGLE speedup, both directions correct on both datasets) as the hard requirement; treat
magnitude agreement within ~25–30% as pass-with-documentation, not a stop condition. Document any
gap and its likely cause (vLLM version, kernel differences, cross-engine effects).

Build by adapting Spec-Bench + SpecMQuant's **evaluation/scoring code** (their eval folder is
modified Spec-Bench; GSM8K-eval + evalplus for HumanEval) — not their quant-wiring, which targets
a different engine entirely (PREREQ_RESULTS Check 5).

**Free correctness test, in addition to the gate:** speculative decoding under greedy decoding is
output-preserving by construction — a spec-on cell's output must match its spec-off counterpart
at concurrency=1 (up to numerics). Add this as a harness regression test (HARNESS_SPEC §10); it
catches a broken adapter cheaply, independent of the SpecMQuant comparison.

### Layer 2 — Serving baseline (foundation of the contribution)
Purpose: the FP16 / no-opt / under-concurrency reference. No published number to match — you
establish it. **Independent variable = offered load** (N concurrent clients, or Poisson rate λ);
**batch size = measured emergent quantity.** Tool: **`vllm bench serve`** (the CLI that
superseded `benchmark_serving.py` — the old script is deprecated; verify feature parity against
the pinned version, since `vllm bench serve` is not a byte-for-byte replacement). Fix and record:
model, **pinned engine_version: vllm==0.24.0** (confirmed via PREREQ_RESULTS Check 6 to run the
full W4A16+FP8-KV+EAGLE-3 stack), concurrency sweep {1,8,32,64}, input/output length mode (fixed +
natural, scaled by concurrency — see §2), decoding (greedy for controlled cells, sampling
temp≈0.7 for realistic cross-check), warmup (discard first K), ≥3 repeats (report median +
spread, not point estimates only — see analysis notes in HARNESS_SPEC §9).

### Layer 3 — Metrics (adopt verbatim — see LITERATURE §7)
TTFT (p50/p95/p99), ITL/TPOT, end-to-end latency, throughput, **goodput**, peak GPU memory; for
spec cells also **average accepted length (τ)** and **verification-to-decoding ratio** (note:
not a metric vLLM exposes natively — derive it from τ, acceptance counters, and ITL deltas
between spec-on/spec-off cells, or profile a few cells directly; don't assume it's a free field
in vLLM's output); for the SGLang seam also **cache hit rate**.

### Hardware caveat (foreground as a finding, not a flaw)
**The entire core factorial runs on A100 (see §3/§6 GPU-routing correction)** — FP8-KV is
software-emulated there (~10–20% penalty), which is itself a real, measurable, reportable
deployment condition for A100-class fleets, not an apology. PREREQ_RESULTS Check 6 empirically
confirmed the full stack (W4A16 + FP8-KV + EAGLE-3) runs correctly on A100 specifically — L4 and
H100 have not been empirically tested for this combination, so don't assume either works without
verifying first. Native-FP8 (H100) is only ever a separate bonus validation block (4b), never
part of the core routing or its headline claim.

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
