# Future work — deferred extensions and ideas that didn't make the cut

The study shipped complete as of 2026-07-20 (all pre-registered hypotheses answered;
see [README](README.md)). This file records what a next iteration could add, and — as
honestly as the findings themselves — what was tried on paper and deliberately cut.
Nothing here is required for the project to stand on its own.

Grouped by nature, not ranked. Where one item is clearly the highest-leverage, it says so.

---

## Ideas explored and cut (kept for the record)

### The optimization-advisor CLI / live-server advisor

**What it was.** The decision guide packaged as an executable tool: a `stack-advisor`
CLI that took a deployment scenario (context length, concurrency, quality tolerance,
GPU) and printed a recommended W/K/S stack with expected effect ranges, provenance, and
a `--validate` mode that recomputed each claim from the raw records. A richer version
would have *inspected a live vLLM server* via its Prometheus `/metrics` endpoint (the
mysqltuner pattern) and advised from the deployment's own observed τ, KV pressure, and
concurrency.

**Why it was cut.** The tool split into two halves and neither justified shipping it as
a product:

- The half that **transfers** across models/GPUs (read live τ → compute the discarded-
  draft fraction → flag a likely net loss) is thin — it's spec-decode arithmetic plus a
  metrics scrape, and doesn't need this study behind it. Existing tools already surface τ.
- The half that **needs this study** (the magnitudes, crossover concurrencies, the
  capacity knee, the context cliff) is valid only inside the measured envelope: one
  model, one GPU family, greedy decoding, two context points. To be impressive the tool
  would have to state those numbers on deployments it never measured; to be honest it
  would have to disclaim them. The honest version is thin; the impressive version
  overclaims — and overclaiming is the one thing that would undercut the project's
  actual asset, its rigor and bounded claims.

The combinatorial space (model × GPU × context × concurrency × sampling × workload)
cannot be covered on a Colab-Pro budget, so no amount of extra experiments rescues a
*general* quantitative advisor. The decision guide survives as **prose backed by
committed records** — which is what it always was — rather than as a CLI implying a
generality the data doesn't support. The recommendation engine lives in git history
(`analysis/stack_advisor.py`, removed 2026-07-20) if a future, better-scoped iteration
wants it back. The one piece worth keeping was extracted:
`analysis/validate_claims.py` still recomputes every headline number from the raw
records (11/11 PASS), covered by `tests/test_validate_claims.py`.

**If revisited, the only honest framing** is "the interference structure is
deployment-specific; here is the instrument that measures it for *your* stack" — i.e.
lead with the harness as a reusable benchmark and treat any advice as bounded to the
measured example. That reframing needs no new experiments.

### Config-safety linter (spun out of the vLLM 2048 bug)

**What it would be.** A small static checker for a vLLM launch config + checkpoint
metadata that flags the exact failure class this project hit: a draft checkpoint whose
`max_position_embeddings` is below the target's serving context length, which silently
destroys EAGLE speculative acceptance (and crashes compiled mode) at long context — see
[analysis/vllm_2048_bug_diagnosis.md](analysis/vllm_2048_bug_diagnosis.md).

**Why it's promising and not cut for the same reason as the advisor.** Unlike the
advisor, this is a *config invariant*, not a performance claim — it's correct on any
model and any GPU with zero additional measurement. It's the one "tool" idea that is
genuinely model-agnostic. Deferred only because closing the study out came first; it's
the most natural small build if a tool is wanted later, and it pairs directly with the
upstream bug report.

---

## Cheap measurement that would strengthen the existing story

### Pin the S context cliff onset (~half a session)

The context axis has only two measured points: ~1k tokens (τ≈2.85, healthy) and ~7.4k
(τ≈1.14, collapsed). The acceptance cliff is real, but *where* it sets in is unmeasured
— the "long context" thresholds in [DECISION_GUIDE.md](DECISION_GUIDE.md) are
descriptive, not a measured onset. A handful of S-on RAG cells at ~2k / 3k / 4k / 5k
tokens would turn the cliff from an interpolation between two points into an actual
curve. Cheapest high-value follow-up; a few GPU units.

---

## Planned extensions not executed (project ships without them)

### Native-FP8 hardware validation on H100 — highest-leverage follow-up

Every K (FP8-KV) finding here is on an A100, where FP8 is *emulated* (~10–20% penalty).
The guide's entire native-FP8 section [HW-FP8] is labeled extrapolation. On an H100 the
emulation tax's cause disappears, so K's whole ledger should improve — plausibly making
K free-to-positive below the capacity knee and softening the "never K under S at low
concurrency" rule. This is the single question a practitioner on modern hardware would
actually ask, and the one most likely to *change* a recommendation. Not done because
Colab H100 proved unreliable (PREREQ Check 1: "opportunistic, not schedulable"); it
needs a reliably-schedulable H100 (GCS VM or equivalent).

### SGLang RAG shared-prefix seam (Phase 4)

The one scoped second-engine extension in PROJECT_SPEC §6: the KV-quant ×
prefix-cache-capacity crossover in RAG serving, where SGLang's RadixAttention prefix
reuse changes the capacity arithmetic FP8-KV operates on. Pre-registered kill criteria;
explicitly lowest priority. Not started. ~1.5–2 weeks part-time, ~80–120 GPU units.

### Sampling temperature T > 0

S's "quality-free" guarantee is measured and proven only under **greedy** decoding.
Real deployments sample. τ, the acceptance economics, and the bit-identical quality
guarantee are all unmeasured off-greedy — and sampling params are per-request and
invisible in server metrics, so this is a genuine blind spot, not just an untested knob.

### Long-context quality under FP8-KV

K's ~zero accuracy cost is a **short-context** result; the long-context cells measured
speed only. Whether FP8-KV rounding degrades accuracy on long inputs (where the KV
cache dominates attention) is open.

### Additional weight precisions and a 4th workload

Stretch items from PROJECT_SPEC §5, none attempted: W8A8 / W4A8 as intermediate weight
precisions; MT-Bench as a 4th (open-ended, judged) workload alongside GSM8K / HumanEval
/ RAG; a parallel-drafting micro-seam. Each is additive, none blocks the core result.

### INT8-KV — recheck before calling permanently out of scope

INT8 KV-cache was unshipped in stable vLLM as of the last check (PREREQ Check 2, issues
#33480 / RFC #37319). Worth a 5-minute re-check of those threads before declaring it
permanently out of scope; if it has landed, it becomes a natural second K precision to
contrast against FP8's emulation tax.

### TensorRT-LLM

Explicitly out of core scope (no Colab support). Stretch-only, on a GCS VM.

---

## Known boundaries the project already owns (not action items)

Listed for completeness; these are stated as scope in
[DECISION_GUIDE.md § Where the data stops](DECISION_GUIDE.md#where-the-data-stops) and
are honest limitations, not gaps to close: one model/size (Llama-3.1-8B-Instruct), one
pinned engine version (vLLM 0.24.0), one EAGLE-3 checkpoint, and the WK quality-offset
mechanism being causally unresolved (KV rounding vs. the attention-backend switch that
accompanies K).
