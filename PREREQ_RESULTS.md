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

**A100 variant is PINNABLE, not luck (user-confirmed empirically, 2026-07-09):** Colab's
**High-RAM runtime toggle controls the A100 variant directly** — OFF = A100-SXM4-40GB,
ON = A100-SXM4-80GB. This supersedes the "bonus observation" framing below. Routing
consequences, applied to the runbooks:
- **Cube/factorial sessions (phase3_factorial.ipynb): High-RAM ON, always.** Phase 2's
  marginal corners were measured on the 80GB card; a cube mixing 40GB and 80GB records is
  confounded exactly like the rejected H100/A100 split. The notebook asserts ≥70GB and
  `analysis/factorial.py` emits a MIXED HARDWARE warning if a cube ever mixes
  `env.gpu_name` values.
- **K-stress addendum (phase3b_kstress_40gb.ipynb): High-RAM OFF, deliberately.** On the
  40GB card both KV ceilings (~16 FP16-KV / ~32 FP8-KV at 7.7k-token contexts) fit inside
  a {8,16,32,48} concurrency grid, so FP8's own plateau is measurable — the 80GB design
  could only show FP16 plateauing (~47) while FP8 "keeps tracking" (its ~94 ceiling
  escapes any sane grid).

**Phase-3b session learnings (2026-07-11, Bugs A/B):** two harness-hardening facts from a
failed 40GB session, both now fixed in code (`harness/engines/base.py`,
`tests/test_engine_lifecycle.py`):
- **vLLM V1 servers must be killed as a process GROUP.** The engine spawns a separate
  EngineCore child; terminating only the launched pid orphans it holding ~16GB of GPU
  memory, and every later launch in the session then fails with a generic "Engine core
  initialization failed". Teardown now signals the whole group (TERM→KILL escalation +
  final sweep), verifies release via `nvidia-smi --query-compute-apps`, and `launch()`
  refuses to start on an occupied GPU with the offending PIDs named.
- **A wedged launch is distinguishable from a slow one.** The eagle3-fp16kv corner
  stalled once at launch (log frozen after backend selection, 0% util, idle power) at a
  command byte-identical to one that served 36 factorial cells on 80GB — so workload/
  context is ruled out; suspected transient init/download stall or 40GB-specific flake
  (one occurrence, unconfirmed). `wait_ready` now fails a wedged launch early instead of
  burning the full 2400s; each record captures the selected attention backend from the
  server log (`env.attention_backend`) since FP8-KV historically selects FlashInfer
  while FP16-KV picks FlashAttention — making backend a recorded variable, not a guess.
  Retry ladder for the parked corner is documented in
  `configs/k_stress/generate_k_stress.py` (rung 2: pin `VLLM_ATTENTION_BACKEND`).
- **Correction (2026-07-12): the watchdog's first version had a false-positive mode,
  caught on the next fresh session.** Its assumption — "weight downloads write progress
  to the server log" — is WRONG: huggingface_hub's tqdm auto-disables on non-tty stdout,
  so a cold ~16GB download writes *nothing* to a redirected log while it runs
  (empirically verified: a real download grew the HF cache 28KB→344MB over ~9s with the
  log frozen at 156 bytes). The single-signal watchdog killed a perfectly healthy
  cold-cache launch of the plainest config in the study after 600 silent seconds. Fix,
  two layers: (1) the watchdog now requires BOTH the server log AND the HF hub cache
  size to be frozen for 600s before declaring a stall — the genuine Bug-A wedge had both
  static, an active download grows the cache continuously, so detection is preserved
  without the false positive; (2) both Phase-3 runbooks pre-download all checkpoints
  (`hf download`, visible progress) before any server launch, so in-sweep cold caches
  don't occur in normal operation anyway. Incidentally, the failed session confirmed the
  Bug-B teardown fix works under fire: after the watchdog kill, the GPU was actually
  released and the next group launched cleanly.

**External-dependency risk: HF Xet CDN outage (2026-07-14, fully diagnosed).** Cold
downloads of large files (model shards; even an unrelated .gif in the EAGLE3 repo)
failed with `403 Forbidden ... SignatureError: invalid key pair id` on hub-issued
presigned URLs (`us.gcp.cdn.hf.co/xet-bridge-us/...`, `Key-Pair-Id=01KXEF...`).
Evidence chain (transcript: `colab/archive_phase3b_xet_debug_20260714.ipynb`): disk,
auth/gating, and raw egress all ruled out (a browser-UA curl of a 548MB file through
plain `resolve/main` succeeded via the healthy `cas-bridge.xethub.hf.co` edge);
`HF_HUB_DISABLE_XET=1` ineffective (known bug, huggingface_hub#3266); **`pip uninstall
hf-xet` also ineffective** — the broken URLs are issued by the hub itself (the Hub is
fully Xet-backed server-side), so no client-side change can fix a hub signing-key
mismatch. Verdict: transient HF infrastructure; wait and retry, never debug client-side
again. Mitigation now standard: both runbooks pre-download via `scripts/predownload.py`
(hard 30-min per-attempt timeout, 3 attempts with backoff, loud failure naming this
incident and the curl fallback) instead of a bare `hf download` that sits silent. Small
silver lining: small-file metadata downloads kept working, so gating/auth checks pass
even mid-outage — don't let that mislead a future diagnosis.

Check 1 status: **DONE.** Burn rate calibrated from two independent real sessions
(Block-0 single-stream, Phase-2 full concurrency sweep); H100 unreliability documented;
A100 variant selection resolved (High-RAM toggle, above).

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

---

## 2026-07-14/15 — Xet CDN 403 root-caused: edge-specific, not transient; automatic curl fallback added

The "wait a few hours" call in the original incident note was falsified: the failure ran
14+ hours with HF's status page green the whole time. Live probing from a second network
explained both facts at once. The same file (identical content hash
`3f45696c938b...`, `hugging-quants/...AWQ-INT4` shard 1) is served by **two different
edges with two different signing keys** depending on how the hub routes the client:

- **hf-client route (what Colab hits):** hub issues presigned URLs on
  `us.gcp.cdn.hf.co/xet-bridge-us` with `Key-Pair-Id=01KXEF4KZ...` → the edge rejects
  its own hub's key: `403 SignatureError: invalid key pair id`. Every retry gets a fresh
  ticket signed with the same broken key, so retries can never succeed.
- **browser-UA route:** plain `/<repo>/resolve/main/<file>` with `-A "Mozilla/5.0"` is
  302'd to `cas-bridge.xethub.hf.co` with `Key-Pair-Id=K3EPXBYC3CKDRZ` (AWS-side) →
  **serves bytes** (verified with a range GET during the outage).

Corollaries: (a) user token/keys are irrelevant — the hub authenticates fine and the
failing key is HF's CDN signing key, and tiny public files (`logo.png`) 403 identically;
(b) the status page stays green because only one route is broken; (c) geography of the
person is irrelevant — routing is decided per client/UA at the hub.

Two cache-layout facts verified against live headers (they make a faithful fallback
possible): on resolve URLs, `x-linked-etag` == the **LFS sha256** for LFS files and the
**git blob oid** for regular files — exactly the blob filenames huggingface_hub uses
under `blobs/`; `/api/models/<repo>/tree/main?recursive=true` reports both oids plus
exact sizes, and `/api/models/<repo>/revision/main` gives the commit sha.

**Mitigation (in repo, GPU-free tested):** `scripts/predownload.py` now auto-falls-back
when the hf CLI exhausts its attempts: it curls the hub API for the tree + sha (browser
UA — load-bearing), downloads every file through the healthy edge with per-file exact
size verification and resumable `.part` files, and reconstructs the standard HF cache
(`blobs/<etag>` + `snapshots/<sha>/` symlinks + `refs/main`). vLLM's hub lookups then
find complete blobs and never GET the broken CDN route. `--curl-only` skips the doomed
hf attempts entirely (~10 min/repo saved while the incident lasts). 8 new tests in
`tests/test_predownload.py` (155 total passing).

---

## 2026-07-15 — KS-probe crash (8/40 Phase-3b cells) root-caused: vLLM 0.24.0 spec-decode token-budget clamp x chunked prefill on long prompts

Symptom: both EAGLE-3 probe servers (fp16kv and fp8kv) died with
`torch.AcceleratorError: CUDA error: device-side assert triggered` ~2 min into
serving; all 8 probe cells `status: failed`, all 32 non-spec cells clean.
A `CUDA_LAUNCH_BLOCKING=1` repro (notebook debug cells) pinned the device-side
assert: Triton `index out of bounds: 0 <= tmpNN < 2048` from an
inductor-compiled kernel in the **eagle_head** torch.compile cache, with the
dumped scheduler state at `num_computed_tokens=[2048]`,
`total_num_scheduled_tokens=2048` on a resumed (chunked-prefill) request.

Root cause chain, each link verified in the server logs:
1. With spec decode on, vLLM 0.24.0 clamps the per-step token budget — launch
   warning `[vllm.py:1614] max_num_scheduled_tokens is set to 2048 based on
   the speculative decoding settings... Consider increasing
   max_num_batched_tokens to accommodate the additional draft token slots`.
   The eagle-head compile also specializes at that boundary
   (`compile_ranges_endpoints: [2048]`).
2. This addendum's ~7.4k-token prompts therefore chunk-prefill at 2048
   tokens/step under EAGLE-3; at the step where a resumed request sits
   exactly at the 2048 boundary, the compiled eagle_head kernels index past
   their `< 2048` bound → device assert → EngineCore fatal.
3. Why nothing hit this before: every prior EAGLE-3 measurement (Block-0,
   Phase 2, the full factorial) used short prompts that prefill in one step —
   never crossing the boundary; every long-prompt Phase-3b cell without spec
   decode never gets the clamp. Only the KS probe combines both. Upstream,
   spec-tokens x chunked-prefill is a known active bug seam (vLLM PR #33652
   "Don't schedule spec tokens with prefill chunks", Feb 2026, plus follow-on
   regressions), so this is engine, not harness.

Fix (rung 1, applied in `configs/k_stress/generate_k_stress.py`, probe corners
only): `--max-num-batched-tokens 8192` (= max_model_len) — every ≤7936-token
prefill becomes single-step (the same single-chunk regime as every other
EAGLE-3 run in the project, so comparability improves rather than degrades)
and the kernel bound rises clear of the largest possible step (7936 + 5 draft
slots < 8192). The 32 completed capacity cells are untouched — non-spec
configs deliberately do NOT carry the flag, enforced by a regression test
(`test_k_stress_config_set`). Rung 2 if a probe corner still dies:
`--enforce-eager` — demoted from first choice because eager-vs-compiled
contaminates the tok/s economics the probe exists to measure (fine as a
diagnostic, poor as the recorded configuration). Rung 3: drop the 8 cells and
record a vLLM 0.24.0 limitation — user decision, never silent.

Rerun ergonomics: `harness.sweep` skips only `status: ok`, so re-running the
sweep cell redoes exactly the 8 failed cells (2 server launches). The
notebook's debug cell (single-cell `harness.run` of `eagle3-fp16kv_c1_r0` to a
throwaway results dir) validates the fix cheaply before committing to the
full probe rerun.

---

## 2026-07-15 (later) — CORRECTION: KS-probe rung-1 fix falsified on GPU; --enforce-eager adopted (rung 2)

The previous entry's rung-1 theory (2048 chunk-boundary; fix = raise
`--max-num-batched-tokens` to 8192) was **falsified** in the live session:
with the flag verified present in the launch command, all 8 probe cells
crashed identically (`CUDA error: device-side assert triggered` →
`EngineDeadError`). A fresh `CUDA_LAUNCH_BLOCKING=1` repro surfaced the
assert at a different host line (`model_runner.py:861 prepare_inputs →
buffer_utils.py:40 pin_memory`) than the original block-table trace —
consistent with one poisoned compiled kernel being caught at whatever CUDA
call syncs next, i.e. the host-side frames were never the defect. Lesson
recorded: for device-side asserts, only the failing Triton/kernel source and
the bisection are evidence; host stacks are noise even under
CUDA_LAUNCH_BLOCKING.

Bisection (2026-07-15, notebook debug cells): same corner
(`kstress_eagle3-fp16kv_c1_r0`), same launch plus `--enforce-eager` →
**status=ok, 34.2 tok/s mean/request, tau=1.144**. Conclusion: the defect is
specific to vLLM 0.24.0's VLLM_COMPILE/CUDA-graph path for the EAGLE-3 head
at long context; exact root cause unpinned (upstream-report material), not
the draft/verify logic and not the scheduler budget.

Adopted fix (generator, probe corners only):
`extra: ["--max-num-batched-tokens", "8192", "--enforce-eager"]` —
byte-identical to the validated working launch (8192 retained because the
proven-good config carried both flags and it silences the draft-slot clamp
warning). Regression tests updated: eagle3 commands must carry both flags,
non-spec commands neither.

**Standing measurement caveat (also emitted by analysis/k_stress.py in the
KS-probe section):** the 8 probe cells are measured in EAGER mode; the 32
capacity cells and the entire factorial are compiled. Within-probe ratios
(fp8kv/fp16kv) and tau are clean; absolute probe tok/s is NOT comparable to
any compiled cell. Every write-up that touches probe economics must compare
ratios across regimes, never raw goodput.

**Parked observation (do not lose, do not chase yet):** first eager cell
measured tau=1.144 at 7.4k-token RAG context vs ~2.5–2.8 in the
short-context repro gate — potentially a real finding (long-context RAG
content harder for this draft head), or an artifact to sanity-check once all
8 cells land.

---

## 2026-07-16 — Phase-3c diagnostics: all four questions answered (9/9 cells, verified against phase3c_diagnostics_results/)

1. **tau collapse is REAL, eager is innocent.** EAGLE-3 in eager mode on
   short GSM8K (server byte-compatible with the probe): tau = 2.830 / 2.878
   across repeats — squarely in the healthy 2.5–2.9 range every compiled
   short-context cell shows. The probe's tau≈1.14 at 7.4k-token RAG context
   is therefore a genuine long-context/draft-head finding, not a
   compilation-mode artifact.
2. **EAGLE-3 is measurably counterproductive at 7.4k-token context.**
   No-spec eager baseline vs the probe's S-on cells (same server flags,
   same workload, same regime): c1 36.0 vs 34.0 tok/s (S = x0.94), c8 187.9
   vs 166.4 (S = x0.89). tau=1.14 with 5 draft tokens per round means ~77%
   of draft compute is discarded — S burns more than it saves once the
   drafter is out-of-distribution. Decision-guide + write-up headline.
3. **Crash bisection FINAL: the vLLM 0.24.0 defect is in the
   inductor-compiled eagle_head kernels, NOT CUDA-graph capture.** With
   compile ON and `cudagraph_mode=NONE` (verified in effect in
   server_20260716_235052.log) the same Triton assert fires
   (`index out of bounds: 0 <= tmp10 < 2048`). Decisive extra fact: this
   run had NO 2048 scheduler clamp (8192 budget respected,
   compile_ranges_endpoints=[8192]) yet the kernel bound is still 2048 —
   the bound is baked into the compiled eagle_head artifact independent of
   the token budget. Full matrix: FULL_AND_PIECEWISE crash, NONE crash,
   eager OK, short-prompt compiled OK, long-prompt no-spec compiled OK.
4. **Attention-backend confound CLEARED.** FP16-KV c8 with FLASHINFER
   pinned: 220.8 / 220.4 tok/s vs 221 on FLASH_ATTN — the backend switch
   that rides along with every K comparison contributes ~0.2%. The
   project's K effects are genuinely about KV precision.

---

## 2026-07-22 — vLLM PR #49343 (maintainer's fix for #48894) independently validated on GPU

A vLLM maintainer opened PR #49343 citing our issue, implementing the same
fix we drafted (raise EAGLE/EAGLE-3 draft `max_position_embeddings` to the
target's `max_model_len`), placed slightly more cleanly (override runs
before the `EAGLEConfig` wrap, so the wrapper inherits the fixed value
without a manual double-update). Review turned up one narrow, verified-latent
gap: checkpoints that load directly *as* `EAGLEConfig` keep a stale inner
`.model` copy at 2048, since nothing in `vllm/` currently reads it;
`SpeculatorsConfig` checkpoints are unaffected (flattens config to top-level
attributes).

Validation notebook: `colab/pr49343_validation_a100.ipynb`
(A100-SXM4-40GB, PR head `f5a7f2eda`, `VLLM_USE_PRECOMPILED=1 pip install
-e .`, no rebuild needed since the PR is Python-only). Raw results in
`pr49343_validation_bundle/` (`partA_fix.log`, `partA_summary.json`,
`partB_control.log`, `partB_excerpt.txt`):

- **Part A (fix applied):** override log line fires exactly once
  (`Overriding draft model max_position_embeddings from 2048 to the target
  model's max_model_len (8192)...`); zero crash-assertion hits, zero
  `AcceleratorError` hits across the full log; all 8 long-context requests
  (~7.4k tokens) returned `200 OK`; tau = 1.1448 (1782 drafts, 8910 draft
  tokens, 258 accepted) — statistically identical to the earlier
  manual-checkpoint-edit validation (tau=1.144, PREREQ 2026-07-17/18
  entries), confirming the actual code fix behaves like the known-good
  workaround.
- **Part B (control — `vllm/config/speculative.py` reverted to `main`,
  same build, same prompt):** override line absent (confirms the revert
  took effect); the original assert returns 96 times, exact same signature
  (`index out of bounds: 0 <= tl.broadcast_to(tmp10, [XBLOCK]) < 2048`) as
  the original #48894 report. Crash is attributable to the code change,
  not environment/version drift.

Posted and confirmed live via `gh api` (2026-07-22): a review comment on
PR #49343 with the pass/control numbers and the `EAGLEConfig` gap note (no
fix/test code included — kept for a follow-up PR); a reply inside
Copilot's review thread correcting its claim that eager mode's docstring
("silent garbage reads") was unverified — our earlier GPU instrumentation
(2026-07-18 entry) already measured this directly; a closing reply on
issue #48894 pointing to the PR review. Status: PR open, unmerged, some
CI red (broad, likely-unrelated suites — worth re-checking `v1-spec-decode`
specifically once CI reruns). Follow-up PR (the `EAGLEConfig` gap fix +
missing test coverage) is prepared in `analysis/vllm_followup_pr_plan.md`,
gated on this PR merging first.
