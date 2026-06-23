# Serving-Regime Interaction Study of Stacked LLM Inference Optimizations

**One-line:** An empirical study of how weight quantization, KV-cache quantization, and
speculative decoding interact when stacked together inside vLLM under continuous batching
and realistic concurrency, on a single commodity GPU — reproducing published single-stream
findings and extending them into the serving regime where no one has characterized them.

This `docs/` set is the **complete, self-contained specification** for the project. It is
written for an implementing agent (Claude Code). You should not need any external context
beyond these files plus the cited papers/repos. Read in this order:

1. **README.md** (this file) — navigation + the prerequisite checks you MUST run first.
2. **PROJECT_SPEC.md** — the frozen problem statement, thesis, and all locked decisions.
3. **LITERATURE.md** — prior work, every source with IDs/repos, what to reproduce.
4. **EXPERIMENT_MATRIX.md** — the 2³ factorial, every cell, the 7 effect-hypotheses, metrics.
5. **HARNESS_SPEC.md** — the benchmark harness architecture to build against.
6. **IMPLEMENTATION_PLAN.md** — phase-by-phase build, checkpoints, kill criteria, timeline.

---

## What this project is (and is not)

It **is**: a reproduce-then-extend empirical/integrative study. The contribution is a
controlled, mechanistically-explained *measurement* of a three-way optimization interaction
in a deployable serving stack, framed as a deployment decision guide.

It **is not**: a new method, a new kernel, or a novelty-first research paper. No claim beats
anyone's benchmark. The value is rigor + the unoccupied seam (serving-regime three-way stack)
+ resolving a concrete research-vs-practitioner contradiction.

**Guiding principle (applies to every decision):** every technology must earn its place
*mechanistically*. If adding a component does not change a measured result for a stated
reason, it does not go in. This is why SGLang is scoped to ONE workload regime (see
PROJECT_SPEC), and why TensorRT-LLM is out of core scope.

---

## The core thesis in three sentences

Practitioner guidance says stacking these optimizations yields cleanly compounding savings
(the "PagedAttention + quantization + speculative decoding = 10–50× cheaper" diagram).
Research shows they interact, sometimes destructively, and the findings even contradict each
other (QuantSpec: quantized KV *raises* speculative acceptance; SpecMQuant: 4-bit weights make
speculative decoding *counterproductive*) — but always at batch-1, in bespoke harnesses.
This project tests who is right inside vLLM, under continuous batching, across concurrency.

---

## PREREQUISITE CHECKS — run these in Phase 0 BEFORE committing to the full matrix

Each of these can reshape the experiment matrix. Do not skip. Record results in
`docs/PREREQ_RESULTS.md` (create it) so decisions are traceable.

### Check 1 — Colab GPU availability and unit burn rate
- Confirm which GPUs the two Colab Pro accounts actually serve (A100 40GB, H100, L4).
- Budget is **~550 compute units total across two accounts**. After the first H100 session,
  read the usage meter to calibrate units/hour for A100 vs H100 vs L4. H100 burns fastest.
- Decision gated: how many native-FP8 (H100) hours are affordable. See EXPERIMENT_MATRIX
  "Compute allocation."

### Check 2 — vLLM KV-cache quantization support for the target model
- Confirm which KV-cache quant precisions the pinned vLLM version supports for
  Llama-3.1-8B-Instruct. FP8 (E4M3) is the primary target. INT8 KV support was an open
  feature request as of Jan 2026 — verify current status.
- **CRITICAL:** FP8 KV is native only on H100+ (SM9.0+). On A100 (SM8.0) it is software-
  emulated with a ~10–20% penalty. This is why FP8-KV cells run on H100 (see matrix).
- Decision gated: whether FP8-KV cells must run on H100, and whether INT8-KV becomes an
  optional axis.

### Check 3 — EAGLE-3 draft checkpoint for the target model
- Confirm a public EAGLE-3 draft head exists for Llama-3.1-8B-Instruct (check SafeAILab/EAGLE
  and the vLLM-supported spec-decode model list).
- For the reproduction gate (Block 0), confirm an **EAGLE-2** head for **Llama-3-8B** exists
  (SpecMQuant used EAGLE-2 on Llama-3-8B).
- Decision gated: if no EAGLE-3 head for 3.1-8B, fall back to EAGLE-2, or a draft-model pair,
  or n-gram speculation — and document the substitution.

### Check 4 — GCS GPU quota (overflow only)
- New GCP projects do NOT have GPU quota by default; a quota-increase request can take time.
- Request it NOW even though GCS is the overflow valve, so it is ready if needed.
- GCS is used only when: a run outlasts a reliable Colab session, H100 units run short, or for
  the (out-of-scope) TensorRT-LLM stretch. **Vertex AI is avoided.**

### Check 5 — Reproduction-harness availability
- Clone Spec-Bench (`github.com/hemingkx/Spec-Bench`) — the harness backbone.
- Clone SpecMQuant (`github.com/AI9Stars/SpecMQuant`) — shows exactly how quantization was
  wired into a Spec-Bench-derived harness. You ADAPT this; you do not write the harness from
  scratch. This is the single biggest de-risking factor in the project.

---

## Locked decisions (do not relitigate — see PROJECT_SPEC for rationale)

| Decision | Value |
|---|---|
| Primary serving engine | vLLM |
| Scoped second engine | SGLang — RAG shared-prefix regime ONLY |
| Out of scope | TensorRT-LLM (stretch only, GCS VM), custom kernels |
| Anchor model | meta-llama/Llama-3.1-8B-Instruct |
| Reproduction-gate model | Llama-3-8B (to match SpecMQuant) |
| Weight quant (primary) | W4A16 (AWQ) |
| Weight quant (optional) | W8A8, W4A8 if time permits |
| KV-cache quant (primary) | FP8 (E4M3), native on H100 |
| Speculative decoding | EAGLE-3 (EAGLE-2 for the reproduction gate) |
| Harness backbone | Spec-Bench, quant-wiring adapted from SpecMQuant repo |
| Serving load driver | vLLM `benchmark_serving.py` (request-rate control) |
| Compute | Colab Pro (A100 + H100 + L4, ~550 units), GCS overflow, no Vertex |

---

## Deliverables

1. A public GitHub repo with the validated harness, all configs, all results, and analysis.
2. The reproduction-gate result (validated reproduction of SpecMQuant's direction).
3. The 2³ interaction matrix + the deployment decision guide.
4. A LinkedIn/Medium write-up series documenting the build and findings.

See IMPLEMENTATION_PLAN for the per-phase shippable checkpoints and the matching write-up beats.
