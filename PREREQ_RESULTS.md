# PREREQ_RESULTS — Phase 0 Prerequisite Check Outcomes

Run 2026-07-02. Checks 1 and 4 require the user's own Colab/GCS accounts and are not
researchable; they're flagged below with the action needed. Checks 2, 3, 5 were run via web
research and by cloning the two reference repos into `external/`.

---

## Check 1 — Colab GPU availability and unit burn rate — DONE (updated 2026-07-08)

**Units confirmed:** 200 on account 1, 300 on account 2 → **500 total**, slightly under the
~550 the plan assumed. Not a blocker, but leaves less slack — burn rate discipline matters more
than the plan assumed.

**A100 confirmed working:** `nvidia-smi` on account 1 shows `A100-SXM4-40GB`, 40960MiB, idle.
Good — this is the primary compute for FP16/AWQ cells regardless of the H100 question below.

**H100 — flagged as unreliable, not confirmed usable.** H100 shows in the runtime-type dropdown
but selecting it does not reliably connect you to an actual H100 — this matches a known,
documented Colab behavior: H100/premium-GPU selection is subject to live availability and can
silently fall back to a lesser GPU even when selected. **Action:** when you do get an H100
session, immediately run `!nvidia-smi` before trusting it — don't assume the dropdown selection
was honored. Treat H100 minutes as opportunistic (grab them when offered), not something you can
schedule reliably.

**Decision gated — this changes GCS's role from "overflow only" to "primary path for H100
cells" if Colab H100 stays unreliable.** PROJECT_SPEC's "FP8-KV cells run on H100 (native)"
routing assumes H100 is obtainable on demand. If it isn't, there are two honest options: (a) run
FP8-KV cells on A100 (software-emulated, ~10-20% penalty — PROJECT_SPEC already anticipates this
exact fallback) and document the substitution, or (b) rent H100 on GCS specifically for those
cells (see Check 4 below — real availability/cost tradeoffs apply there too, so this isn't a
free escape hatch). Revisit once GCS quota is live and you've tried a few more Colab H100
attempts.

**Still not done: burn-rate calibration.** The `nvidia-smi` output above is from an idle
session (0% util, no process) — it confirms GPU assignment, not cost. Next step: run vLLM
serving a real workload (e.g., load Llama-3.1-8B, drive ~50 requests) for a fixed duration on
each GPU class you get, and read the compute-units delta before/after. That number is what
actually gates how many H100/A100 hours are affordable across the 6 weeks.

**Burn-rate calibration — first real data point (2026-07-07, Block-0 W4A16 session):**
8 units for the resumed session (190 → 182) covering the two W4A16 server groups —
checkpoint download, two ~3-min startups, ~14 min of timed measurement, plus overhead —
roughly 40 min end-to-end, i.e. **~12 units/hr on A100**. Planning number until more data:
**12 units/hr A100 → 500 units ≈ ~40 A100-hours total.** Consistent with the revised
EXPERIMENT_MATRIX §2 estimate that the budget is tight, not comfortable.

**Full config board sizing (2026-07-07, factorial build):** the complete core-factorial
board is now generated: 8 corners × 3 workloads × 4 concurrencies × **3 repeats** = 288
cells (repeats added per HARNESS_SPEC §9's revised uncertainty-reporting requirement).
Estimated from Block-0 *measured* rates (~3.5 min average per single-stream cell, faster
at higher concurrency): ≈ 25–40 min per corner-repeat × 24 corner-repeats ≈ **12–18
A100-hours ≈ 145–215 units** at the calibrated ~12 units/hr — comfortably inside
EXPERIMENT_MATRIX §2's 30–45 GPU-hour envelope (which also budgets dev/debug), and
affordable within the remaining unit balance. The 3-repeat addition does NOT blow the
budget because repeats share server launches (one launch per corner, not per repeat).
Which subset runs first is a sweep-time glob selection (see
`configs/factorial/generate_phase2.py` docstring), not a config decision; kill levers
(trim repeats at extreme concurrencies first) remain available if sessions run short.

**Bonus observation:** Colab served an **A100-SXM4-80GB** (not the 40GB the plan assumed).
More KV headroom and ~30% more memory bandwidth than a 40GB — good for the factorial, but
(a) don't count on it every session, and (b) the GPU variant is recorded per-run in the
result records (`env.gpu_name`); keep the core factorial on whichever A100 variant is
actually obtainable *consistently*, and note the variant in any cross-run comparison.

**Phase-2 marginals session, real data (2026-07-08):** 48/48 cells `status: ok`, zero
failures. Units burned: 165.61 → 153.88 = **11.73 units for the full 48-cell run** (4 server
groups, all four singles across 3 workloads × 4 concurrencies) — well under the notebook's
own 30–45 unit estimate. This single data point suggests the 145–215 unit estimate for the
full 288-cell board (line 53 above) is conservative; worth letting Fable recompute the
Phase-3 budget against this real number rather than the Block-0-only extrapolation, since the
Phase-2 session covers more diverse cell shapes (4 launch groups, 3 workloads, full
concurrency sweep) than Block-0's single-stream-only calibration did. Remaining balance as of
this session: **153.88 units** (this account only — check the other account separately).

Check 1 status: **DONE.** Burn rate calibrated from two independent real sessions
(Block-0 single-stream, Phase-2 full concurrency sweep); H100 unreliability and A100-80GB
variance both documented above.

Source: [H100 is not selected · googlecolab/colabtools#5976](https://github.com/googlecolab/colabtools/issues/5976)

## Check 2 — vLLM KV-cache quantization support — CONFIRMED, no changes to locked decisions

- **FP8 (E4M3) KV cache: mature and directly usable.** `--kv-cache-dtype fp8` quantizes the
  full attention computation, with both per-tensor and per-attention-head calibration strategies.
  vLLM's own April 2026 blog post benchmarks this exact config on **Llama-3.1-8B** — FP8 nearly
  halves the decode ITL slope vs BF16 with negligible TTFT penalty. This directly de-risks the
  anchor-model FP8-KV cells.
- **Native-vs-emulated split confirmed:** the H100-native / A100-emulated (~10-20% penalty)
  split assumed in PROJECT_SPEC §4.3 is correct — no change needed to the "FP8-KV cells run on
  H100" routing rule.
- **INT8 KV cache: still NOT shipped in stable vLLM as of July 2026.** Found two open threads:
  [vllm-project/vllm#33480](https://github.com/vllm-project/vllm/issues/33480) (opened Jan 31,
  2026, requesting INT8 support explicitly because FP8 has narrower hardware support than INT8)
  and an RFC, [#37319](https://github.com/vllm-project/vllm/issues/37319) (March 2026), proposing
  a per-token INT8 KV scale infrastructure. Neither is merged. **Decision: INT8-KV stays optional
  / out of the must-ship matrix**, exactly as PROJECT_SPEC already hedges — re-check these two
  issues right before Phase 3 in case one lands.

Sources: [vLLM FP8 KV-cache blog (Apr 2026)](https://vllm.ai/blog/2026-04-22-fp8-kvcache), [Quantized KV Cache docs](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/), [Issue #33480](https://github.com/vllm-project/vllm/issues/33480), [RFC #37319](https://github.com/vllm-project/vllm/issues/37319)

## Check 3 — EAGLE checkpoints — CONFIRMED for both models

- **EAGLE-3 for Llama-3.1-8B-Instruct (anchor model): exists.** Official checkpoint is
  `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` on Hugging Face. No fallback needed.
- **EAGLE-2/EAGLE-1-family for Llama-3-8B-Instruct (reproduction-gate model): exists, and it's
  the exact checkpoint SpecMQuant used.** Confirmed directly from the cloned SpecMQuant repo's
  own model table: `yuhuili/EAGLE-LLaMA3-Instruct-8B`, used for all their Llama-3-8B precision
  variants (W8A8, W4A16, W4A8). Using this same checkpoint gives you a clean, checkpoint-identical
  comparison against their published numbers for the reproduction gate — stronger than the plan
  assumed, since it removes "different EAGLE checkpoint" as a confound if the gate fails.
- **One open item, not a blocker:** vLLM's speculative-config `method` field needs the exact
  string mapping verified when the adapter is built — docs confirm `eagle3` as a method name, but
  the EAGLE-1/2-style checkpoint (`yuhuili/EAGLE-LLaMA3-Instruct-8B`) likely maps to plain
  `eagle`, not `eagle2` (vLLM does not appear to expose a distinct `eagle2` method string; EAGLE-2
  is a drafting-tree algorithm variant, not a different checkpoint format). Verify this when
  writing `vllm_adapter.py` — flag it for the Fable 5 build session rather than assuming
  HARNESS_SPEC's `spec_decode: eagle2` config value maps 1:1 to a vLLM flag.

Sources: [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE), [yuhuili/EAGLE3-LLaMA3.1-Instruct-8B](https://huggingface.co/yuhuili/EAGLE-LLaMA3.1-Instruct-8B), [vLLM EAGLE docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/eagle/), SpecMQuant repo (`external/SpecMQuant/README.md`)

## Check 3b (added) — AWQ checkpoint for the anchor model — CONFIRMED

`hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` exists (AutoAWQ, GEMM kernels, group size
128, ~4GB VRAM for weights alone). This is the W4A16 primary weight-quant checkpoint
PROJECT_SPEC §4.3 calls for — confirmed available, no substitution needed.

Source: [hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4](https://huggingface.co/hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4)

## Check 4 — GCS GPU quota — ACTION NEEDED (user), account already upgraded

Account confirmed already upgraded to paid billing with a payment method attached (2026-07-03)
— the free-trial GPU-quota block (see below) does not apply. **Remaining step: file the actual
quota request.** Recommendation: 1x A100 (e.g. `us-central1`) as the primary overflow target,
since it's proven working on Colab and more reliably schedulable on GCS than H100; treat H100
quota as a secondary/stretch ask given Colab H100 is already unreliable (Check 1) and H100
capacity on GCS is harder to actually schedule even once quota is approved. Navigation:
Console → IAM & Admin → Quotas → filter Service = Compute Engine API → search "GPU" → select
the row(s) → Edit Quotas.

Historical note (no longer applies, kept for reference): GPU quota increases are blocked while
an account is in "Free Trial" status regardless of credit balance — upgrading to paid retains
the $300 credit and only removes that restriction.

Source: [Google Developer forums — "$300 Free Trial is useless without GPU Quota"](https://discuss.google.dev/t/300-free-trial-is-useless-without-gpu-quota-my-trial-period-is-wasting-away/290091), [Google Cloud Free Program docs](https://docs.cloud.google.com/free/docs/free-cloud-features)

## Check 5 — Reproduction-harness availability — CONFIRMED, with one important correction

Both repos cloned successfully into `external/Spec-Bench` and `external/SpecMQuant`.

**What's directly reusable (confirms the plan):**
- Spec-Bench's `data/spec_bench/question.jsonl` includes an 80-example **`rag` category**
  (Natural-Questions-style: long retrieved passage + question), confirming HARNESS_SPEC §7's
  premise that the RAG workload can be built from Spec-Bench's RAG subtask. It is **one
  question per passage** in its raw form, so it still needs the multi-question-per-document
  restructuring HARNESS_SPEC §7 already specifies — that work was correctly scoped, not
  something this check removes.
- SpecMQuant ships its own `evaluation/gsm8k/` and `evaluation/humaneval/` correctness-eval
  code (adapted from GSM8K-eval and evalplus) plus matching datasets in `data/gsm8k` and
  `data/human_eval`. This is directly reusable for `harness/correctness.py` — genuinely the
  biggest de-risking win the README claims.

**Correction to the plan — SpecMQuant is not a vLLM harness.** Its README states outright:
*"all experiments are implemented in C/CUDA."* It runs its own inference engine (`llamacu/`),
not vLLM — the connection to vLLM is that its **W4A16 kernel is borrowed from
`vllm-project/vllm` and IST-DASLab/marlin** (per its Acknowledgments), not that it wires
quantization into a vLLM server. Practically: you can adapt SpecMQuant's **evaluation/scoring
layer** and its **published reference numbers** directly (as planned), but the "quant-wiring"
into the actual serving engine — `--quantization awq`, `--kv-cache-dtype fp8`,
`speculative_config` — has to be built against **vLLM's own docs and CLI**, not lifted from
SpecMQuant's code, since that code targets a different engine entirely. README's "you ADAPT
this; you do not write the harness from scratch" is accurate for the correctness/scoring half
of the harness, overstated for the serving/engine-launch half. Worth noting in HARNESS_SPEC so
whoever builds `vllm_adapter.py` doesn't go looking for quant-wiring patterns in SpecMQuant's
C/CUDA source.

Sources: `external/Spec-Bench/Readme.md`, `external/SpecMQuant/README.md` (both cloned in this repo)

## Check 6 — Full-stack compatibility (W4A16 + FP8-KV + EAGLE-3 simultaneously) — CONFIRMED (2026-07-06)

This was not covered by Checks 2/3 (each verified one axis in isolation) and is the single
biggest unverified assumption underneath the project's headline cell. Tested directly on a
Colab A100 — see `SpeculativeDec_Testing.ipynb`.

**Result: it works.** `LLM(model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
quantization="awq_marlin", kv_cache_dtype="fp8", speculative_config={"model":
"yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "method": "eagle3", "num_speculative_tokens": 5})`
initializes and generates coherent, consistent output across 10 prompts on **vLLM 0.24.0**.
Clean shutdown, no errors. **Pin `engine_version: vllm==0.24.0`** — this is now a confirmed,
not assumed, fact.

**What had to be true for this to work, learned the hard way:**
- **`quantization="awq"` is wrong; use `"awq_marlin"`.** Plain `awq` triggers a vLLM warning
  ("not fully optimized... forcing awq. Use quantization=awq_marlin for faster inference") and
  is not what was actually tested end-to-end. Update HARNESS_SPEC's config schema and the
  `vllm_adapter.py` launch logic accordingly — `factors.weight_quant: w4a16` should map to
  `quantization=awq_marlin`, not `awq`.
- **An older vLLM (0.10.1) genuinely could not do this** — `--kv-cache-dtype fp8` forced a
  fallback to the V0 engine, and V0 does not support speculative decoding at all
  (`NotImplementedError: Speculative decoding is not supported on vLLM V0`). This is fixed in
  0.24.0 (FP8-KV now runs under V1 alongside spec-decode) — do not pin an older vLLM version for
  this project, the gap was real and version-dependent.
- **Colab's stock environment cannot host this cleanly.** Its preinstalled RAPIDS/torch/xformers
  stack conflicts with whatever torch version a fresh vLLM install requires, breaking the
  install in confusing, unrelated-looking ways (`ModuleNotFoundError`, `ImportError:
  libcudart.so.13`). Fix: install vLLM inside an isolated `virtualenv` (not stdlib `venv` —
  Colab's `ensurepip` bootstrap fails; `virtualenv` bundles its own pip and sidesteps that), and
  run it as a subprocess (`/content/vllm_env/bin/python script.py`), not inside the notebook's
  own kernel. **Any future GPU work on Colab for this project should start from this recipe, not
  a bare `pip install` into the notebook kernel.**
- **FlashInfer (the attention backend vLLM selects for this stack) JIT-compiles a CUDA kernel at
  first use and needs `ninja` on `PATH`.** Installing `ninja` inside the venv isn't sufficient by
  itself — the venv's `bin/` also needs to be prepended to `PATH` explicitly for the subprocess
  call to find it (`PATH="/content/vllm_env/bin:$PATH" ...`), since the venv was never
  `source`d/activated.

**Useful data surfaced as a side effect, not just pass/fail:**
- A100-40GB: 415,856 KV-cache tokens available under FP8, "3.17x" max concurrency at 131K
  context — real capacity numbers for the capacity-mediated K-effect story.
- Full engine init (load + torch.compile + spec-decoder graph capture + warmup) took ~172s —
  confirms HARNESS_SPEC's "launch once per config, amortize across all workload/concurrency/
  repeat cells" design is necessary, not just an optimization.
- vLLM logged its own accuracy caveat for FP8-KV: *"may cause accuracy drop without a proper
  scaling factor."* Don't let `vllm_adapter.py` silently rely on default/uncalibrated FP8 scales
  later — this needs a deliberate decision (calibrated scales vs. documented as-is), not silence.

Source: `SpeculativeDec_Testing.ipynb` (this repo, direct empirical test, 2026-07-06)

## Check 3 resolution — vLLM speculative_config method string — RESOLVED (2026-07-06, harness build)

The open item from Check 3 is resolved against live vLLM docs; the harness is built on these
values and `tests/test_vllm_adapter.py` locks them in:

- **EAGLE-1-style checkpoints (`yuhuili/EAGLE-LLaMA3-Instruct-8B`) use `method: "eagle"`.**
  vLLM's own EAGLE docs example uses exactly this checkpoint with `"method": "eagle"`. As
  suspected, there is **no `eagle2` method string** — EAGLE-2 is a drafting-tree algorithm on
  the same checkpoint format. HARNESS_SPEC §3's sketched `spec_decode: eagle2` value is
  superseded by `eagle`; the harness config validator rejects `eagle2` loudly.
- **EAGLE-3 heads use `method: "eagle3"`** (unchanged, and matching what Check 6 ran).
- **No checkpoint conversion needed**: yuhuili checkpoints load directly from HF on
  vLLM ≥ 0.7.0 (post-PR #12304). The conversion caveat only applies to <0.7.0.
- **Acceptance metrics**: vLLM V1 exposes Prometheus counters
  `vllm:spec_decode_num_drafts`, `vllm:spec_decode_num_draft_tokens`,
  `vllm:spec_decode_num_accepted_tokens`; mean accepted length τ = 1 + accepted/drafts.
  The harness deltas these around each timed window (after warmup).
- **Drafting-shape caveat for the gate**: SpecMQuant ran EAGLE-2 *tree* drafting (depth 6,
  top-k 10, tree size 60 — from their run scripts). vLLM EAGLE is *chain* drafting
  (`num_speculative_tokens`, harness default 5, matching Check 6). Expect lower τ and lower
  absolute speedups; the gate fails only on direction (EXPERIMENT_MATRIX §7, revised).

Two further corrections found while wiring the reproduction configs:

1. **SpecMQuant's Llama-3-8B W4A16 is GPTQ-g128 (AutoGPTQ, sym), not AWQ**:
   `YudiZh/Meta-Llama-3-8B-Instruct-W4A16-g128` (their README model table). The repro configs
   use that exact checkpoint — closer to their setup than an AWQ substitute — and rely on
   vLLM auto-detection (which picks the marlin GPTQ path); the Check-6 `awq_marlin` mapping
   applies to the AWQ *anchor-model* checkpoints, and the adapter hard-rejects plain `awq`.
   Note their headline figure likely uses the **rotation** variant (`-Rot` + rotated EAGLE
   head), which vLLM cannot load; we run the plain g128 checkpoint + stock EAGLE head — their
   documented "without rotation" configuration.
2. **The paper has no per-benchmark speedup table for the 8B model** (per-task tables cover
   70B only). The 8B reference numbers for the gate are read from Figure 1(a) (A100):
   FP16+EAGLE 2.3×, W4A16 quant-only 2.1×, W4A16+EAGLE 1.3× relative. Recorded with
   provenance and tolerances in `configs/repro/reference_targets.yaml`.

**Addendum (2026-07-07, found during the first Block-0 GPU run): the YudiZh W4A16 repo is
NOT loadable by vLLM — correction 1 above is itself corrected.** The FP16 groups of the
sweep completed fine; the W4A16 groups failed at weight load with
`RuntimeError: start (0) + length (1792) exceeds dimension size (896)` on a row-parallel
layer. Diagnosis, confirmed against SpecMQuant's own `model_convert/convert_w4a16.py`:
`YudiZh/Meta-Llama-3-8B-Instruct-W4A16-g128` hosts the **post-conversion llamacu artifact**,
not an AutoGPTQ checkpoint. Evidence (all three match the converter's output exactly):

- weights file is `model_gptq.safetensors` — the converter's hardcoded output name;
- keys are fused (`self_attn.qkv_proj.qweight`, `mlp.gate_up_proj.qweight`) and
  `qzeros`/`g_idx` are absent — the converter fuses q/k/v and gate/up and keeps only
  marlin-permuted `scales`;
- qweight dim0 is `k/16` (down_proj: 14336/16 = 896) — the marlin tile repack
  (`C.gptq_marlin_weight_repack`, `packed_data = zeros((shape_0//16, shape_1*16//8))`) —
  where vLLM's GPTQ loader expects AutoGPTQ's `k/8` (= 1792). Hence 896-vs-1792.

Un-converting (inverse marlin tile permutation + reconstructing dropped tensors) is
checkpoint surgery with no validation path — rejected. **Substitute:
`TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ`** — a standard AutoGPTQ checkpoint, not gated,
complete (model.safetensors + tokenizer + inlined `quantization_config`), and matching
SpecMQuant's quantize recipe field-for-field where visible: bits=4, group_size=128,
sym=true, desc_act=true, **damp_percent=0.01** (their exact value; the astronomer
alternative uses 0.1). Calibration set unknown for both theirs and the substitute — one
more reason the gate is direction-based, not magnitude-based. Consequence for the repro
claim: "checkpoint-identical to SpecMQuant" now holds only for the **EAGLE draft head**
(`yuhuili/EAGLE-LLaMA3-Instruct-8B`); the W4A16 target is recipe-matched, not
artifact-identical. Configs updated (`configs/repro/repro_w4a16_*.yaml`); the debugging
transcript is archived at `colab/archive_block0_debug_20260707.ipynb`.

**One documented deviation from HARNESS_SPEC §5:** Block-0 load is driven by the harness's own
streaming OpenAI-client driver (`harness/load.py`), not `vllm bench serve`, because the gate
needs the generated text (GSM8K/HumanEval correctness scoring) and per-request timings, which
the bench CLI does not return. Same closed-loop concurrency semantics; `vllm bench serve`
remains an option for Layer-2 cross-checks in Phase 2.

---

## Net effect on the matrix / plan

No locked decision changes. Carry into Phase 1:
1. Use `yuhuili/EAGLE-LLaMA3-Instruct-8B` (confirmed SpecMQuant-identical) for the
   reproduction-gate draft model — removes a confound.
2. Verify the vLLM `speculative_config.method` string for EAGLE-1/2-style checkpoints before
   assuming `eagle2` is a valid config value (HARNESS_SPEC §3 config schema).
3. Scope the vLLM adapter's quant/spec wiring against vLLM's own docs, not SpecMQuant's source —
   only its evaluation/scoring code and reference numbers carry over directly.
4. **Pin `engine_version: vllm==0.24.0`**, map `weight_quant: w4a16` → `quantization=awq_marlin`
   (not `awq`), and use the isolated-venv install recipe from Check 6 for all future Colab GPU
   sessions on this project.
5. Decide explicitly how FP8-KV scaling factors are calibrated before trusting any FP8 accuracy
   number the harness reports.

Remaining open items: Check 1's burn-rate calibration (still not done — the nvidia-smi checks
so far were idle, not under load) and Check 4's actual quota request filing — both require the
user's accounts. The design-level corrections from the independent model review (GPU/K-factor
confound, request-count scaling, SGLang seam reframing, reproduction-gate tolerance, log-scale
interference gap, free correctness test, APC pinning) plus this session's Check 6 findings
(pinned engine version, awq_marlin, FP8 calibration note) have now been applied to
EXPERIMENT_MATRIX.md and HARNESS_SPEC.md — specs are current as of 2026-07-06.
