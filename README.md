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

## Status (2026-07-07)

**Phase 0 (prerequisites) and Phase 1 (harness + reproduction gate) are complete.** See
`PREREQ_RESULTS.md` for the full prerequisite findings and `block0_results/repro_gate_report.md`
for the gate output. Headline result:

**Block 0 reproduction gate: PASS.** EAGLE's relative speedup over the FP16 baseline dropped
from 1.64×→1.26× (GSM8K) and 1.89×→1.37× (HumanEval) once the target model went 4-bit —
reproducing SpecMQuant's (ACL 2025) core finding that weight quantization erodes speculative
decoding's benefit, in vLLM (a production continuous-batching engine) rather than their bespoke
single-stream C/CUDA harness. Accepted length (τ) was essentially unchanged (2.49→2.46,
2.83→2.64), confirming the mechanism is economic (fixed spec-decode overhead against
cheaper-per-step decode), not a drop in draft-head accuracy.

A few things were corrected from the original spec along the way — the GPU routing that would
have confounded the KV-quant factor with hardware, the reproduction checkpoint (SpecMQuant's 8B
W4A16 is GPTQ, not AWQ), the vLLM version needed for FP8-KV and speculative decoding to coexist
(pinned to `vllm==0.24.0`), and the SGLang seam's original hypothesis (dropped a
quality-propagation claim that doesn't survive scrutiny, replaced with a cache-capacity framing).
Full detail in `PREREQ_RESULTS.md`; the numbers below already reflect the corrected state.

**Next up:** Phase 2 (serving baseline + single-optimization marginals) — see
`IMPLEMENTATION_PLAN.md`.

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

## PREREQUISITE CHECKS — status: complete, see PREREQ_RESULTS.md

All five checks below have been run; full findings, corrections, and provenance are in
`PREREQ_RESULTS.md` (repo root, not `docs/`). Summary per check, kept here for navigation only —
**PREREQ_RESULTS.md is the source of truth, not this section.**

### Check 1 — Colab GPU availability and unit burn rate — DONE
Confirmed 500 units total across two accounts (200 + 300, not the originally-assumed ~550).
A100 confirmed working; H100 is selectable in Colab's UI but unreliable in practice (silently
falls back to a lesser GPU). Measured burn rate: **~12 units/hr on A100 → ~40 A100-hours total
budget.** Treat H100 as opportunistic bonus only, never a planned dependency.

### Check 2 — vLLM KV-cache quantization support for the target model — DONE
FP8 (E4M3) KV-cache confirmed mature and working. INT8 KV-cache remains unshipped in stable
vLLM. **Routing correction:** the original plan split FP8-KV cells to H100 and FP16-KV cells to
A100 — this would have confounded the KV-quant factor with hardware. The entire core factorial
now runs on A100 (FP8 software-emulated there, ~10–20% penalty, itself a documented finding);
H100 native-FP8 is an optional separate validation block only, never part of core routing.

### Check 3 — EAGLE checkpoint availability — DONE
EAGLE-3 confirmed for the anchor model (`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`); EAGLE-1/2-style
checkpoint confirmed for the reproduction gate (`yuhuili/EAGLE-LLaMA3-Instruct-8B`, verified
identical to SpecMQuant's own). **Resolved: the vLLM config value is `method: "eagle"`, not
`"eagle2"` — no such method string exists in vLLM** (EAGLE-2 is a drafting-tree algorithm, not a
different checkpoint/config format).

### Check 4 — GCS GPU quota (overflow only) — account upgraded, quota request still pending
Account confirmed upgraded to paid billing with a payment method attached. Recommendation:
request 1x A100 quota (more reliably schedulable than H100) as the actual overflow path. This is
the one remaining open action item — everything else in this list is done.

### Check 5 — Reproduction-harness availability — DONE, with a correction
Both Spec-Bench and SpecMQuant cloned and inspected. **Correction:** SpecMQuant is bespoke
C/CUDA, not a vLLM harness — only its evaluation/scoring code and reference numbers carry over
directly; the vLLM engine-launch/quant-wiring was built against vLLM's own docs instead.

---

## Locked decisions (do not relitigate — see PROJECT_SPEC for rationale)

| Decision | Value |
|---|---|
| Primary serving engine | vLLM, **pinned `engine_version: vllm==0.24.0`** (Check 6: confirmed to run W4A16+FP8-KV+EAGLE-3 together; older 0.10.1 confirmed broken for this combination) |
| Scoped second engine | SGLang — RAG shared-prefix regime ONLY (seam hypothesis reframed to cache-capacity/crossover, not quality-propagation — see EXPERIMENT_MATRIX §5) |
| Out of scope | TensorRT-LLM (stretch only, GCS VM), custom kernels |
| Anchor model | meta-llama/Llama-3.1-8B-Instruct |
| Reproduction-gate model | Llama-3-8B (to match SpecMQuant; W4A16 cell uses their exact GPTQ checkpoint, not AWQ) |
| Weight quant (primary) | W4A16 — vLLM `quantization=awq_marlin` (not plain `awq`) for the anchor model |
| Weight quant (optional) | W8A8, W4A8 if time permits |
| KV-cache quant (primary) | FP8 (E4M3) — **entire core factorial runs on A100** (emulated FP8, ~10–20% penalty, a documented finding); H100 native-FP8 is bonus-only, never core routing (avoids confounding K with hardware) |
| Speculative decoding | EAGLE-3 (`method: eagle3`); EAGLE-1/2-style checkpoint for the reproduction gate uses `method: eagle` — there is no `eagle2` method string |
| Harness backbone | Spec-Bench + SpecMQuant's evaluation/scoring code only (SpecMQuant itself is bespoke C/CUDA, not a vLLM harness — its quant-wiring does not carry over) |
| Serving load driver | `vllm bench serve` (superseded `benchmark_serving.py`) for Phase 2+; Block 0 uses the harness's own streaming driver since the gate needs generated text for correctness scoring |
| Compute | Colab Pro (confirmed 500 units, ~12 units/hr on A100 ≈ 40 A100-hours), GCS paid overflow (account upgraded), no Vertex |

---

## Deliverables

1. A public GitHub repo with the validated harness, all configs, all results, and analysis.
2. The reproduction-gate result (validated reproduction of SpecMQuant's direction).
3. The 2³ interaction matrix + the deployment decision guide.
4. A LinkedIn/Medium write-up series documenting the build and findings.

See IMPLEMENTATION_PLAN for the per-phase shippable checkpoints and the matching write-up beats.
