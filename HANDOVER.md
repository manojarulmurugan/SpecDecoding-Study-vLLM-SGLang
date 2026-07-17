# HANDOVER — session-independent project state (written 2026-07-14)

For a fresh assistant chat: read this, then PREREQ_RESULTS.md (the living
ops ledger), then skim README.md / EXPERIMENT_MATRIX.md / HARNESS_SPEC.md
for design rationale. This file states what is DONE, what is IN FLIGHT, and
the working norms. Repo: github.com/manojarulmurugan/SpecDecoding-Study-vLLM-SGLang.

## What this project is

Portfolio study (ML-engineer hiring audience, not publication): how weight
quantization (W = AWQ W4A16), KV-cache quantization (K = FP8), and
speculative decoding (S = EAGLE-3) interact when STACKED in vLLM under
continuous batching, on Llama-3.1-8B-Instruct, single A100 (Colab Pro).
Design: replicated 2^3 factorial, log-space effects, interference gap =
naive-product-of-marginals minus measured full stack.

## Status by phase (all data committed under *_results/)

- **Phase 0/1 (Block-0 reproduction gate): PASSED.** Reproduced SpecMQuant's
  direction (EAGLE speedup erodes under W4A16: 1.64x->1.26x GSM8K,
  1.89x->1.37x HumanEval) in vLLM. Gotcha history: SpecMQuant's published
  W4A16 checkpoint is their custom llamacu/Marlin format (unloadable by
  vLLM); substituted recipe-matched TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ.
- **Phase 2 (marginals, 48 cells): DONE.** Headlines: tau is flat in
  concurrency (erosion is economics, not acceptance); S crossover tracks
  tau (RAG 0.90x at conc 64); W fades to ~1.0x at conc 64; K ~null on
  80GB (admission-limited regime -- motivated Phase 3b).
- **Phase 3 (full 2^3, 288 cells): DONE, verified clean.** All pairwise
  interactions negative in all 12 workload x concurrency cells; interference
  gap 1.30x-2.97x, shrinking with concurrency; KS negative even at conc 1
  with tau invariant under K (QuantSpec's acceptance channel absent for
  EAGLE-3; A100 FP8 emulation tax x S's op multiplication explains it);
  WS "amplified under batching" hypothesis FALSIFIED (flat-to-shrinking);
  novel: W drops tau on GSM8K (-14%) but not HumanEval/RAG -- WS has an
  acceptance channel on reasoning workloads, KS never does.
- **Phase 3b (K-stress, 40 configs in configs/k_stress/): 32/40 DONE**
  (2026-07-14 session; K-isolation 16 + AWQ capacity corners 16, all
  `status: ok` in results/). The 8 long-context KS-probe cells (EAGLE-3)
  crashed the engine; fix VALIDATED on GPU, awaiting final rerun. vLLM
  0.24.0's compiled eagle_head kernels device-assert on the ~7.4k prompts;
  rung 1 (--max-num-batched-tokens 8192 alone) was FALSIFIED live, rung 2
  (--enforce-eager, bisection-confirmed: status=ok, tau=1.144) is adopted
  -- probe corners carry both flags, byte-identical to the validated
  launch (both PREREQ 2026-07-15 entries; the second corrects the first).
  CAVEAT that must follow the probe numbers everywhere: those 8 cells are
  EAGER, everything else compiled -- compare ratios, never raw tok/s
  (analysis/k_stress.py prints this in the probe section). Parked: probe
  tau=1.144 at long context vs ~2.5-2.8 short-context -- check once all 8
  land. Rerun = re-execute the sweep cell (resume skips the 32 ok cells).
  Prior failed attempts all root-caused and fixed: (1) doc sizing 400s ->
  tokenizer-exact sizing + prompt_token_budget; (2) Bug A launch stall +
  Bug B zombie EngineCore -> see below; (3) this KS-probe engine crash.

## Hard-won operational facts (do not re-learn these)

1. **Colab High-RAM toggle pins the A100 variant**: OFF=40GB, ON=80GB.
   Factorial/cube sessions MUST use 80GB (matches Phase-2/3 records; mixing
   confounds the cubes -- analysis/factorial.py warns on mixed gpu_name).
   k_stress uses 40GB deliberately (both KV ceilings fit the grid).
2. **Pinned engine: vllm==0.24.0** in an isolated virtualenv at
   /content/vllm_env (Colab kernel install breaks; recipe in PREREQ Check 6;
   ninja needed on PATH for FlashInfer JIT).
3. **Burn rate ~12 units/hr on A100**; balances tracked in PREREQ Check 1.
4. **Bug B (fixed, tested)**: vLLM V1 spawns an EngineCore child; teardown
   must kill the process GROUP (start_new_session + killpg escalation) and
   launch() refuses to start on an occupied GPU. tests/test_engine_lifecycle.py.
5. **Bug A (open, one occurrence)**: eagle3-fp16kv launch stalled once on
   40GB at a command byte-identical to one that served 36 cells on 80GB.
   Suspected transient. Watchdog now fails wedged launches early; probe
   corners ordered LAST in the sweep; rung 2 = uncomment
   VLLM_ATTENTION_BACKEND override in configs (see generator docstring).
6. **Stall watchdog is TWO-signal** (server log + HF cache growth): tqdm is
   silent on non-tty stdout, so cold downloads write nothing to redirected
   logs (empirically verified; first watchdog version false-killed a
   cold-cache launch). Never regress this to log-only.
7. **env.attention_backend is recorded per run** (FP8-KV historically
   selects FlashInfer; FP16-KV picks FlashAttention).

## Immediate blocker (2026-07-14): HF Xet CDN signature failure — ROUTED AROUND

Cold downloads of large files fail with `403 SignatureError: invalid key
pair id` from `us.gcp.cdn.hf.co/xet-bridge-us/...` presigned URLs (evidence
chain in colab/archive_phase3b_xet_debug_20260714.ipynb). Root-caused after
14h disproved "transient" (full analysis: PREREQ 2026-07-14/15 entry): the
hub routes hf-client requests to the GCP edge whose signing key is broken,
while browser-UA requests to plain resolve URLs are 302'd to the healthy
`cas-bridge.xethub.hf.co` edge — retries on the hf route can never succeed
(each fresh ticket is signed with the same broken key), user tokens are
irrelevant, and the status page stays green because only one route is down.
Mitigation in repo: `scripts/predownload.py` now AUTO-falls-back to
browser-UA curls of resolve URLs and reconstructs the standard HF cache
layout (blobs/<etag> + snapshot symlinks + refs/main; etag mapping verified
against live headers), size-verified per file, resumable. `--curl-only`
skips the doomed hf attempts while the incident lasts. GPU-free tested
(tests/test_predownload.py, 155 total). Cell 4b unchanged in usage;
UNBLOCKED pending the user's next 40GB session.

## Working norms

- Everything is built and tested GPU-free (fake vLLM server in tests/;
  `python3 -m pytest tests -q`, 140+ tests). The user runs Colab notebooks
  (colab/phase3_factorial.ipynb = 80GB cube sessions;
  colab/phase3b_kstress_40gb.ipynb = 40GB k_stress) and reports results back.
- **Never commit or push** — the user does that; Colab pulls from GitHub,
  so remind them to push before a session.
- Configs are GENERATED (configs/factorial/generate_phase2.py,
  configs/k_stress/generate_k_stress.py) — edit generators, never YAML.
- PREREQ_RESULTS.md is the append-only ops ledger; corrections are recorded
  as corrections, not overwritten.
- Sweeps are resumable (completed run_ids skipped); results are atomic JSON
  per run under results/runs/.

## Next milestones after 3b lands

**Phase 3c diagnostics: DONE 2026-07-16, 9/9 ok-or-informative (results in
phase3c_diagnostics_results/, full verdicts in PREREQ 2026-07-16 entry).**
(1) eager mode itself is innocent (short-context eager tau 2.83/2.88);
(2) EAGLE-3 measured COUNTERPRODUCTIVE at 7.4k context (S = x0.94 at c1,
x0.89 at c8 vs no-spec eager baseline); (3) crash bisection: NOT cudagraph
capture (cudagraph_mode=NONE still asserts; kernel bound 2048 even with
budget 8192) — upstream issue FILED:
github.com/vllm-project/vllm/issues/48894; (4) attention-backend confound
cleared for SPEED (~0.2%): K goodput effects are genuinely KV precision.

**vLLM crash root cause: RESOLVED side-thread (2026-07-17).** Source-level
diagnosis in `analysis/vllm_2048_bug_diagnosis.md`, every file:line
independently re-verified against a fresh v0.24.0 clone: the draft
checkpoint's max_position_embeddings=2048 sizes the draft RoPE cache;
compiled mode asserts on the gather past 2048, eager mode reads OUT OF
BOUNDS silently. Draft GitHub comment for #48894 is at the bottom of that
file — user posts it.

**OPEN CAVEAT gating the write-up headline (do not lose): tau=1.14 may be
the RoPE bug, not the drafter.** The diagnosis predicts exactly what 3c
measured: healthy tau below position 2048 (short-context eager 2.85),
garbage rotations => collapsed tau above it (long-context 1.14). The
S-counterproductive-at-long-context finding is therefore PENDING a retest:
download the draft checkpoint locally, edit its config.json
max_position_embeddings 2048 -> 8192, point --speculative-config's model
at the local path, re-measure tau at 7.4k context WITH compilation on
(also proves the crash fix end-to-end). tau back near ~2.5-2.85 => the
finding flips to "a checkpoint metadata bug silently destroys EAGLE-3 on
long contexts — and a one-line config edit fixes both crash and
performance" (arguably a better headline); tau stays low => original
finding stands. One throwaway config + 1 launch, ~15 min GPU. The advisor
carries this caveat on D2-S-long until settled.

Phase 5 progress: `analysis/stack_advisor.py` BUILT (scenario CLI with
per-recommendation provenance + `--validate`, which recomputes every
quantitative claim from raw records: 11/11 PASS against the 337 records on
disk); quality-side factorial BUILT (`analysis/quality_factorial.py`,
computed 2^3 accuracy contrasts -> phase3_results/quality_effects.json;
headline: quality does NOT compound, |excess| <= 0.7 pts in all 8 cells; W
main -3/-4 pts GSM8K and -6/-8 pts HumanEval; NEW ROBUST FINDING: WK
interaction POSITIVE on HumanEval +1.7..+3.5 pts in all 4 cells — FP8-KV
partially offsets W4A16's quality damage, mechanism unresolved). Remaining
Phase 5: write-up series; prose DECISION_GUIDE.md distilled from the
advisor findings (small). Phase 4 (SGLang RAG seam) still optional/open.

**Decision-guide requirement (2026-07-15, verified against phase3_results/runs/):
the three levers do NOT cost the same thing on the quality axis — this must be
an explicit dimension in analysis/decision_guide.py, not just a speed/goodput
recommendation.** Pulled real `measured.accuracy` values directly (GSM8K
exact-match, HumanEval unit-test pass/fail; RAG has no ground truth by design,
always null, not a gap):
- **W (W4A16) measurably costs accuracy**: -5 to -14 points vs FP16 on both
  GSM8K and HumanEval, every cell checked. A real quality-for-speed trade.
- **K (FP8-KV) costs ~nothing**: differences within 1-2 questions out of 64,
  consistent with float rounding noise, not degradation.
- **S (EAGLE-3) costs ~nothing under greedy decoding**: HumanEval shows
  bit-for-bit identical accuracy spec-on vs spec-off in every cell checked;
  this is the expected theoretical guarantee (greedy spec decode is
  output-preserving), and tests/test_repro_gate.py::test_accuracy_drift_under_greedy_spec_warns
  already exists to catch a violation.
The guide's recommendations should reflect this asymmetry explicitly (e.g.
"reach for K and S first if quality risk matters; W buys more but you pay for
it in correctness") rather than ranking the three levers on speed alone.
