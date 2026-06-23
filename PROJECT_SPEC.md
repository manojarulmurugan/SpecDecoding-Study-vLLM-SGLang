# PROJECT_SPEC — Frozen Specification

This file is the immutable spec. Other docs (matrix, harness, plan) may evolve during
implementation; this one defines what the project *is* and should not drift.

---

## 1. Problem statement

In production LLM serving, engineers stack inference optimizations — weight quantization,
KV-cache quantization, and speculative decoding — and prevailing practitioner guidance treats
their savings as independent and compounding (the canonical "stack PagedAttention +
quantization + speculative decoding for 10–50× cheaper inference" diagram seen across 2026
production guides).

The research literature contradicts this clean-compounding assumption, and contradicts itself
in instructive ways:

- At batch size 1, a **quantized KV cache has been shown to *raise* speculative acceptance
  rates** and improve speedup (QuantSpec, ICML 2025), via improved consistency between
  drafting and verification when both read the same quantized cache.
- Also at batch size 1, **4-bit weight quantization can make speculative decoding
  *counterproductive*** (SpecMQuant, ACL 2025): once per-token decode is cheap (because
  weights are 4-bit), the draft-verification compute becomes the bottleneck — the
  verification-to-decoding ratio crosses 1 and tree-style speculation loses.

But these interaction findings live almost entirely in **single-stream, batch-1, bespoke-CUDA
harnesses**. The parallel body of serving-regime work (TurboSpec; the 2026 interpretable-
latency-model line) studies speculative decoding under continuous batching but **without
stacking quantization**. The intersection is uncharacterized.

**Central question:** How does the three-way stack of weight-quant + KV-quant + speculative
decoding actually behave inside a production engine (vLLM) under continuous batching and
realistic concurrency, on a single commodity GPU?

This project maps that intersection with Llama-3.1-8B-Instruct: it **reproduces** the
published single-stream interaction findings as validated baselines, then **stress-tests**
them across concurrency levels and workload shapes to produce a **deployment decision guide**
for when stacking these optimizations compounds and when it interferes.

A **scoped extension** tests whether SGLang's RadixAttention prefix-sharing and KV-cache
quantization compound cleanly on a controlled high-overlap RAG workload — the setting where a
single quantization error in a shared prefix propagates to every request that reuses it.

---

## 2. The contradiction that is the hook

Lay the findings side by side:

- **QuantSpec (ICML 2025):** quantized KV cache → higher acceptance → helps. (batch-1)
- **SpecMQuant (ACL 2025):** 4-bit weights → spec decoding counterproductive. (batch-1)
- **Serving-regime line (TurboSpec etc.):** spec-decoding benefit collapses as batch grows.
- **Practitioner blogs (2026):** "these optimizations are independent and compound, minimal
  overlap or diminishing returns" — the opposite of all the above nuance.

The practitioner consensus ("stack them, they compound") directly contradicts the research
consensus ("they interact, sometimes destructively, regime-dependent"). **This project tests
who is right, in a real serving stack.** That sentence is the project's reason to exist and its
most hiring-manager-legible framing.

---

## 3. Positioning vs prior work (state this crisply; interviewers will probe it)

**Already done — we reproduce, do not claim:**
- Quantization × spec-decoding interaction at batch-1 (QuantSpec, SpecMQuant, QSpec).
- Spec-decoding alone under serving load (TurboSpec, interpretable-latency-model, AdaSpec,
  BanditSpec, SPIRe).
- The Spec-Bench workload methodology and metric definitions.

**Open — ours to claim:**
- The **three-way (weight-quant + KV-quant + spec-decoding) interaction under continuous
  batching and concurrency**, in a deployable engine, on commodity single-GPU, framed as a
  deployment decision guide — explicitly testing the research-vs-practitioner contradiction.
- No paper in the reviewed literature occupies this intersection. The contribution is
  **empirical/integrative**, not a new method.

**Closest competitor — SpecMQuant — and the differentiation:**
SpecMQuant is bespoke C/CUDA, single-stream, on Llama-3-8B with EAGLE-2, and its endpoint is
*proposing a new method*. This project is in **vLLM (the deployable engine), under continuous
batching and concurrency sweeps, measuring the three-way stack including KV-quant (which
SpecMQuant does not vary), with the endpoint being a deployment decision guide**. We reproduce
their single-stream finding as a *baseline*, then show whether it survives in the serving
regime.

---

## 4. Locked decisions and rationale

### 4.1 Engines
- **vLLM is primary** for the entire three-way stack across all workloads. Reason: the
  deployable standard, the baseline new research compares against, Colab-friendly.
- **SGLang is a scoped extension on the RAG shared-prefix regime ONLY.** Reason: SGLang's
  RadixAttention only diverges from vLLM mechanistically when there is high prefix overlap.
  On the spec-decoding and weight-quant axes the engines are at parity (running SGLang there
  would be keyword farming — explicitly rejected). See §6.
- **TensorRT-LLM is out of core scope.** Reason: does not run on Colab (root, CUDA containers,
  ~28-min per-config compiles); marginal third data point for large setup cost. Stretch-only,
  on a GCS VM, if everything else lands early.

### 4.2 Model
- **Anchor: meta-llama/Llama-3.1-8B-Instruct.** Reason: ubiquitous, mature AWQ/GPTQ + EAGLE
  checkpoints, universal engine support, fits single-GPU.
- **Reproduction gate: Llama-3-8B.** Reason: SpecMQuant evaluated this exact model, enabling
  direct comparison against their published numbers to validate the harness.

### 4.3 Optimization levels
- **Weight quant primary: W4A16 (AWQ).** Best-supported on Colab; matches SpecMQuant's
  headline cell. W8A8 / W4A8 optional if time permits.
- **KV-cache quant primary: FP8 (E4M3).** Native on H100; emulated (~10–20% penalty) on A100.
  INT8-KV optional pending vLLM support (Prereq Check 2).
- **Speculative decoding: EAGLE-3** (EAGLE-2 for the reproduction gate to match SpecMQuant).

### 4.4 Datasets — see EXPERIMENT_MATRIX §"Workloads" for full detail
- GSM8K (low acceptance, reasoning), HumanEval (high acceptance, code) — also the
  reproduction anchors. RAG shared-prefix set (long context, the SGLang seam + business
  anchor). MT-Bench optional 4th.

### 4.5 Compute
- Colab Pro primary (A100 + H100 + L4, ~550 units across two accounts). FP8-KV cells on H100
  for native FP8. GCS as overflow only. Vertex AI avoided.

---

## 5. Success criteria

- **Minimum success (must-ship):** reproduction gate passes (SpecMQuant direction reproduced
  within tolerance) + the 8-config 2³ factorial completed on GSM8K + HumanEval + RAG at the
  concurrency sweep + an interaction analysis answering the compound-vs-interfere question.
- **Target success:** the above + the SGLang RAG seam + the deployment decision guide + a
  clean write-up series.
- **Stretch:** optional weight precisions, MT-Bench, INT8-KV, parallel-drafting micro-seam.

A clean **negative or null result is a success**, not a failure. If the three-way stack turns
out additive (vindicating practitioners), that is an equally publishable, equally interesting
finding. The project is designed so that *the direction of several effects is genuinely unknown
in advance* (see EXPERIMENT_MATRIX §hypotheses).

---

## 6. The SGLang scoping logic (why it is not keyword farming)

The test: does adding SGLang change a result for a mechanistic reason, or just re-run the same
one? Answer per axis:

- **Spec decoding:** engines at parity (2026 benchmarks: "tie"). Running both = theater. REJECTED.
- **Weight quant:** same underlying Marlin-family kernels. No divergence. REJECTED.
- **KV-quant × prefix-sharing:** RadixAttention and KV-quant act on the *same KV bytes*.
  A quantization error in a *shared* prefix propagates to *every* request reusing it, so the
  quality stakes of KV-quant may be higher under RadixAttention. This is a real, unexplored
  interaction. ACCEPTED — and ONLY here, on the high-overlap RAG regime.

The engine is therefore a **targeted comparison on one workload regime**, never a global axis.
vLLM remains primary everywhere else.

---

## 7. Non-negotiables (the discipline that makes this credible)

1. **Reproduce before you trust.** The reproduction gate (Block 0) is a correctness gate; if it
   fails, stop and fix the harness before any novel number is believed.
2. **Batch size is emergent, not set.** In vLLM you sweep offered load (concurrency / request
   rate); batch size is a *measured* quantity. Claiming "batch size = 32" in vLLM is a tell of
   a fake serving experiment.
3. **No fabricated baselines.** Read ground-truth numbers from the cited papers/repos; never
   assert a target speedup from memory.
4. **Byte-identical shared prefixes** for the RAG/SGLang regime, or RadixAttention's match
   breaks and the comparison is invalid.
5. **Every technology earns its place mechanistically.** No component is added for keyword
   coverage.
6. **Every run is reproducible** from a version-controlled config and writes results atomically.
