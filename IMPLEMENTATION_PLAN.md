# IMPLEMENTATION_PLAN — Phases, Timeline, Checkpoints, Deliverables

Build sequence for Claude Code. Principles: reproduce-then-extend, single-stream-before-serving,
marginals-before-interactions, shippable-at-every-stage. Target window: ~6 focused weeks
(compresses via the kill levers in EXPERIMENT_MATRIX §9).

---

## Phase 0 — Foundations + prerequisite gates (Week 1)

**Do first, before any matrix commitment.** Run the five prerequisite checks (README
"PREREQUISITE CHECKS") and write outcomes to `PREREQ_RESULTS.md`. Each can reshape the matrix:
- GPU availability + unit burn rate → how many H100/FP8 hours are affordable.
- vLLM KV-quant support (FP8 confirmed; INT8 status) → whether INT8 becomes optional.
- EAGLE-3 head for 3.1-8B (EAGLE-2 for 3-8B repro) → spec-decode config or fallback.
- GCS GPU quota request filed (overflow readiness).
- Clone Spec-Bench + SpecMQuant.

In parallel, scaffold the repo (HARNESS_SPEC §2), commit the config schema + result schema +
atomic writer + resume logic, and stand up the vLLM adapter enough to serve Llama-3.1-8B and
return a health check.

**Checkpoint / ship state:** none yet (groundwork). **Study in parallel** (Tier-1 concepts):
arithmetic intensity / memory-bound vs compute-bound, prefill vs decode + KV cache, quantization
number formats, the spec-decoding draft-verify-accept loop + why it is lossless. (Book chapters
1, 2, 5.)

---

## Phase 1 — Harness + reproduction gate (Week 2)  ← FIRST SHIPPABLE ARTIFACT

Build the full measurement harness (load generator, metrics, correctness eval, sweep
orchestration with checkpointing) on the Spec-Bench backbone. Then execute **Block 0**: reproduce
`FP16+EAGLE` and `W4A16+EAGLE` on Llama-3-8B, single-stream, GSM8K+HumanEval, vs SpecMQuant's
published numbers.

**Correctness gate:** speedup within ±10–15% AND correct sign of the W4A16-vs-FP16 effect. If it
fails, STOP and fix the harness — every downstream number depends on this. Document any gap +
cause.

**Checkpoint / ship state:** "Validated reproduction of ACL 2025 (SpecMQuant) speculative-
decoding × quantization findings, in a vLLM-based harness." Already a real, credible artifact.

---

## Phase 2 — FP16 serving baseline + single-optimization marginals (Week 3)  ← COMPLETE TWO-CONFIG STUDY

Establish the Layer-2 serving baseline (`FP16/FP16-KV/no-spec`) across {1,8,32,64} on all three
workloads. Then measure each optimization ALONE (W, K, S separately) single-stream then under the
concurrency sweep — these marginals are both shippable and the required inputs to the interaction
analysis.

**Checkpoint / ship state:** a complete study of weight-quant + KV-quant + spec-decoding measured
independently across workloads and concurrency — publishable on its own.

---

## Phase 3 — The core 2³ factorial + interaction analysis (Week 4)  ← CORE CONTRIBUTION

Run the remaining factorial corners (the pairwise and three-way combinations) across the sweep.
Route FP8-KV cells to H100 (native FP8); FP16-KV cells to A100/L4. Then run `analysis/factorial.py`
to compute main effects, pairwise (focus: W×S and K×S), the three-way, and the **interference
gap**, each as a function of concurrency.

**Checkpoint / ship state:** the full interaction matrix answering compound-vs-interfere — the
project's headline result, complete and defensible. **This is minimum success** (with Phases 0–2).

---

## Phase 4 — SGLang RAG seam (Week 5)  ← HIGH-VALUE EXTENSION (optional)

Add the SGLang adapter. On the RAG shared-prefix workload only, run {vLLM, SGLang} ×
{FP16-KV, FP8-KV} × overlap{low,mid,high} at conc {8,32}. Test: does KV-quant + RadixAttention
compound on memory, and does quantizing a heavily-reused prefix cost disproportionate quality?
Map the overlap crossover and whether KV-quant shifts it. Enforce byte-identical prefixes
(HARNESS_SPEC §7).

**Checkpoint / ship state:** the scoped engine comparison with a real mechanistic finding (or a
clean null at low overlap, which is itself the expected control).

---

## Phase 5 — Synthesis, decision guide, write-up (Week 6)

Run `decision_guide.py`; assemble the deployment recommendations; polish the repo (README,
reproducible configs, result store, plots). Write the LinkedIn/Medium series (beats below).
Optionally pick up low-priority extensions (W8A8/W4A8, MT-Bench, INT8-KV) only if ahead.

**Checkpoint / ship state:** finished, packaged project + public write-up.

---

## Kill criteria (pre-committed — see EXPERIMENT_MATRIX §9)

If time compresses, pull in order: (1) drop optional precisions + MT-Bench; (2) concurrency
4→3 levels {1,8,64}; (3) repeats 3→2; (4) defer SGLang seam (Phase 4) to a follow-up. Core
(Phases 0–3) always ships. The reproduction gate is the one non-negotiable.

---

## Working with Claude Code (two-agent worktree split)

Mirrors the user's established pattern (parallel engine + benchmarking agents via git worktrees,
JSON-schema interface contract defined before agents write):
- **Agent A (engine/harness):** `harness/engines/`, `harness/load.py`, `harness/run.py`,
  server-launch logic, the config schema.
- **Agent B (benchmark/analysis/viz):** `harness/workloads/`, `harness/metrics.py`,
  `analysis/`, plotting.
- **Interface contract = the result-record schema (HARNESS_SPEC §4).** Freeze it before either
  agent starts so they integrate cleanly. Use separate worktrees to prevent collision.
- Long sweeps run unattended via the resumable `sweep.py`; schedule overnight where unit budget
  and session limits allow, GCS VM for anything outlasting a reliable Colab session.

---

## Per-phase deliverables: résumé bullets + write-up beats

(For the user's portfolio/LinkedIn/Medium documentation goal.)

- **Phase 1 →** "Reproduced ACL 2025 speculative-decoding × quantization findings in a vLLM-based
  benchmark harness; validated against published single-stream results." / Post: *"Reproducing a
  research result is the unglamorous skill nobody teaches — here's what broke."*
- **Phase 2 →** "Benchmarked weight quantization (AWQ W4A16), FP8 KV-cache quantization, and
  EAGLE-3 speculative decoding independently on Llama-3.1-8B across concurrency 1–64 in vLLM,
  characterizing latency/throughput/goodput trade-offs per workload." / Post: *"Why 'batch size'
  is a lie in vLLM — and what to measure instead."*
- **Phase 3 →** "Designed a 2³ factorial study of stacked LLM inference optimizations under
  continuous batching; quantified the interference gap between practitioner 'clean-compounding'
  guidance and measured combined speedup, finding [result] that [holds/breaks] with concurrency."
  / Post: *"Do inference optimizations actually stack? I tested the 10–50× claim."*
- **Phase 4 →** "Measured the interaction between SGLang RadixAttention prefix-sharing and FP8
  KV-cache quantization on a controlled high-overlap RAG workload; mapped the prefix-overlap
  crossover where RadixAttention overtakes vLLM and whether KV-quant shifts it." / Post: *"When a
  quantized shared prefix bites every request that reuses it."*
- **Phase 5 →** "Published a deployment decision guide for stacking inference optimizations by
  workload shape and concurrency regime." / Post: the synthesis piece.

---

## Definition of done

Public repo with: validated harness, all version-controlled configs, the full result store, the
interaction matrix + plots, the deployment decision guide, and a clear README that lets a reader
reproduce any cell from its config. Reproduction gate documented. Write-up series live.
