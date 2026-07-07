# HARNESS_SPEC — Benchmark Harness Architecture

The technical contract for the measurement infrastructure. Build against this. The harness is
adapted from Spec-Bench (backbone) + SpecMQuant's quant-wiring (reference), not written from
scratch.

---

## 1. Design goals

1. **One config → one run → one atomic result record.** Fully reproducible, resumable.
2. **Engine-agnostic core** with thin per-engine adapters (vLLM primary; SGLang for the RAG
   seam). The same workload + metric code drives both.
3. **Offered-load driver, not batch-size setter.** Concurrency/request-rate is the knob; batch
   size is recorded as emergent.
4. **Checkpointed sweeps** — a Colab disconnect costs one cell, never a whole sweep.

---

## 2. Repository layout

```
inference-stack-study/
├── docs/                       # the spec set (this folder)
├── configs/                    # one YAML per run-cell; version-controlled
│   ├── repro/                  # Block 0 reproduction-gate configs
│   ├── factorial/              # the 8 core configs × sweep
│   └── sglang_seam/            # Block 5 configs
├── harness/
│   ├── run.py                  # entrypoint: takes a config, executes, writes result
│   ├── sweep.py                # orchestrates a set of configs with checkpointing/resume
│   ├── engines/
│   │   ├── base.py             # EngineAdapter ABC
│   │   ├── vllm_adapter.py     # vLLM server launch + benchmark_serving driver
│   │   └── sglang_adapter.py   # SGLang server launch + same metric collection
│   ├── workloads/
│   │   ├── gsm8k.py
│   │   ├── humaneval.py
│   │   ├── rag_shared_prefix.py   # builds the overlap-controlled RAG set
│   │   └── mt_bench.py            # optional
│   ├── metrics.py              # goodput, accepted length, verif/decode ratio, TTFT/ITL...
│   ├── load.py                 # concurrency / Poisson arrival generator
│   ├── results.py              # atomic JSONL/parquet writer + schema validation
│   └── correctness.py          # accuracy eval per workload (GSM8K/HumanEval/RAG)
├── analysis/
│   ├── factorial.py            # main effects + pairwise + 3-way interaction computation
│   ├── plots.py                # heatmaps, stacked-savings-with-interference, goodput curves
│   └── decision_guide.py       # derives the deployment recommendations
├── results/                    # JSONL/parquet records (git-ignored or LFS)
└── PREREQ_RESULTS.md           # output of the Phase-0 checks
```

---

## 3. Config schema (one YAML per cell)

```yaml
run_id: factorial_w1k1s1_rag_c32_r2        # unique, deterministic from fields below
block: core_factorial                       # repro | serving_baseline | core_factorial | sglang_seam | optional
engine: vllm                                # vllm | sglang
engine_version: "vllm==0.24.0"              # PINNED — confirmed via PREREQ_RESULTS Check 6 to
                                             # run the full w4a16+fp8-kv+eagle3 stack; do not
                                             # substitute an older version (0.10.1 confirmed
                                             # broken: kv-cache-dtype forces V0, which rejects
                                             # speculative decoding entirely)
model: meta-llama/Llama-3.1-8B-Instruct     # Llama-3-8B for repro block
factors:
  weight_quant: w4a16                       # fp16 | w4a16 | w8a8 | w4a8
                                             # w4a16 -> vLLM quantization="awq_marlin", NOT "awq"
                                             # (plain awq works but is unoptimized; awq_marlin is
                                             # the confirmed-tested config, Check 6)
  kv_quant: fp8                             # fp16 | fp8 | int8 (int8 unshipped in vLLM, Check 2)
                                             # fp8 -> vLLM kv_cache_dtype="fp8"; vLLM itself warns
                                             # this "may cause accuracy drop without a proper
                                             # scaling factor" — decide calibration explicitly,
                                             # don't silently rely on defaults
  spec_decode: eagle3                       # none | eagle3 | eagle2
enable_prefix_caching: true                 # EXPLICIT, always recorded — vLLM defaults this on
                                             # (v0.6.0+); for RAG cells this is a real factor in
                                             # the vLLM-vs-SGLang comparison (hash-block APC vs
                                             # radix-tree), not a hypothetical to ignore
draft_model: "<EAGLE3 head id>"             # null if spec_decode=none
workload: rag_shared_prefix                 # gsm8k | humaneval | rag_shared_prefix | mt_bench
workload_params:
  prefix_overlap: high                      # low | mid | high  (RAG only)
  num_requests: 200                         # SCALE with concurrency, don't fix — see
                                             # EXPERIMENT_MATRIX §2 (roughly 60-80 at conc=1,
                                             # 400+ at conc=64, targeting >=2-3 min steady state)
  input_len_mode: natural                   # fixed | natural
  output_len_cap: 512
concurrency: 32                             # SET. batch size is MEASURED.
decoding: greedy                            # greedy (controlled) | sampling (cross-check)
seed: 1234
repeat_idx: 2
warmup_requests: 16
gpu_target: a100                            # a100 (primary, all core factorial cells) | h100
                                             # (bonus native-FP8 validation subset only, Block 4b
                                             # — never mixed into core routing, see
                                             # EXPERIMENT_MATRIX §3/§6) | l4 (dev/debug only)
```

`run_id` must be deterministic from the fields so reruns overwrite cleanly and resume logic can
detect completed cells.

---

## 4. Result record schema (atomic write per run)

```json
{
  "run_id": "...",
  "config": { "...": "verbatim copy of the YAML" },
  "env": {
    "git_commit": "...", "gpu_name": "...", "driver": "...", "cuda": "...",
    "engine_version": "...", "timestamp_utc": "..."
  },
  "measured": {
    "ttft_ms": {"p50": 0, "p95": 0, "p99": 0},
    "itl_ms":  {"p50": 0, "p95": 0, "p99": 0},
    "e2e_latency_ms": {"p50": 0, "p95": 0, "p99": 0},
    "throughput_tok_s": 0,
    "goodput_tok_s": 0,
    "peak_gpu_mem_gb": 0,
    "emergent_batch_size": {"mean": 0, "p50": 0, "max": 0},
    "accepted_length_tau": 0,          // spec cells only
    "verif_to_decode_ratio": 0,        // spec cells only
    "cache_hit_rate": 0,               // sglang seam only
    "accuracy": 0                      // workload correctness metric
  },
  "status": "ok"                        // ok | failed | partial
}
```

Write atomically (temp file → rename) so a kill mid-write never corrupts the store. `sweep.py`
resumes by skipping any `run_id` already present with `status: ok`.

---

## 5. Engine adapter contract (`engines/base.py`)

```
class EngineAdapter(ABC):
    def launch_server(config) -> handle      # start server with the config's factors applied
    def is_ready(handle) -> bool             # health check before timing
    def teardown(handle)                      # clean shutdown, free GPU
    # metric collection is shared via benchmark_serving-style driving in load.py
```

- **vLLM adapter:** launch with `quantization=awq_marlin` (not plain `awq` — confirmed via
  PREREQ_RESULTS Check 6 as the tested, optimized config), `kv_cache_dtype=fp8`,
  `speculative_config` (EAGLE-3 draft + `num_speculative_tokens`), and the
  `--disable-spec-on-high-load` style guard left OFF for controlled cells (we WANT to observe
  erosion). Drive load with **`vllm bench serve`** (the CLI that superseded
  `benchmark_serving.py` — verify feature parity against the pinned 0.24.0 version before relying
  on it for closed-loop concurrency control). Pin `engine_version: vllm==0.24.0` everywhere.
- **SGLang adapter:** launch with `--quantization` (weights) + KV-quant flag + EAGLE spec; same
  load driver; additionally collect radix-tree cache hit rate. Record `enable_prefix_caching`
  explicitly for the vLLM side of every RAG comparison cell (vLLM defaults APC on since v0.6.0) —
  the seam is APC-hash-block-vs-radix-tree, not sharing-vs-no-sharing (EXPERIMENT_MATRIX §5).

**Amortize server startup:** `sweep.py` groups runs by config (the 8 factorial configs), launches
the server once per config, and drives all workload × concurrency × repeat cells against it before
teardown. Only ~8 launches for the core factorial.

---

## 6. Load generation (`load.py`) — the methodological core

- Independent variable is **offered load**: either a fixed pool of N concurrent clients
  (closed-loop) or a Poisson arrival process at rate λ (open-loop). Prefer closed-loop
  concurrency {1,8,32,64} for the controlled factorial; optionally add an open-loop λ sweep for
  the realistic cross-check.
- **Scale `num_requests` with concurrency — never fix it.** A fixed count is wrong at both ends:
  too slow at conc=1 (sequential), too few batch "waves" to reach steady state at conc=64. See
  EXPERIMENT_MATRIX §2 for the scaling guidance and worked example.
- **Record the emergent batch size** by sampling the engine's running-batch metric during the
  run. NEVER set batch size directly. This is non-negotiable (PROJECT_SPEC §7.2).
- Discard `warmup_requests` before timing (engine warmup, CUDA graph capture).

---

## 7. RAG shared-prefix workload (`workloads/rag_shared_prefix.py`) — handle with care

This powers both the business anchor and the SGLang seam. A naive build silently breaks the
RadixAttention comparison.

Requirements:
1. **Small pool of long documents**; generate/collect **many questions per document**. The
   questions-per-document ratio is the `prefix_overlap` knob (low | mid | high).
2. **Byte-identical shared prefix** across a question batch about the same document — same
   system prompt + same document text + same ordering, character-for-character. A single
   differing character breaks the tokenizer-level prefix match and invalidates cache hits.
   Enforce via a fixed template; assert prefix-token-ID equality across a batch in a unit test.
3. **Consistent prefix ordering** (system prompt → document → question), never reordered.
4. Base it on Spec-Bench's RAG subtask (NQ + retrieved passages) restructured for
   multi-question-per-document, for comparability. Keep a fully synthetic controlled set
   (a few public-domain long docs + templated questions) in reserve as a clean-room control.

Output: batches where, at `high` overlap, most tokens are the shared doc prefix and only the
short question suffix is unique — the case RadixAttention is built for. At `low` overlap
(1 question/doc), expect vLLM ≈ SGLang (no finding) — that is the intended low end of the
crossover sweep.

---

## 8. Metrics (`metrics.py`) — definitions are fixed (LITERATURE §7)

- **goodput** = verified-and-generated target tokens / sec (exclude rejected speculative tokens).
- **accepted_length_tau** = mean accepted draft tokens per verification step.
- **verif_to_decode_ratio** = verification-step compute time / single-token decode time; >1
  signals spec decoding has stopped paying off.
- TTFT / ITL / TPOT / e2e via the serving driver; peak GPU mem via engine/`nvidia-smi` sampling.
- **cache_hit_rate** (SGLang) = prefix tokens served from radix tree / total prefix tokens.

---

## 9. Analysis layer (`analysis/`)

- `factorial.py`: from the 8-corner results compute the 3 main effects, 3 pairwise interactions,
  and the 3-way interaction (standard 2³ effect estimates), **per workload and per concurrency
  level** — so concurrency-dependence of each interaction is explicit. Note this means ~12
  mini-factorials (3 workloads × 4 concurrencies), not one — the pre-stated hypothesis directions
  (EXPERIMENT_MATRIX §5) are your protection against cherry-picking sign flips across them; cite
  them as such in the write-up.
- **Define "clean compounding" on the right scale: multiplicative, so analyze in log space.**
  "Clean compounding" is a multiplicative claim (savings multiply, not add), so compute effects
  on log(goodput) / log(latency); the three-way interaction contrast **in log space** is the
  actual test of clean compounding. **Interference gap** = naive product of individual-optimization
  wins (in log space: sum of individual log-effects) minus the measured combined win (the actual
  three-way log-effect). Positive gap = sub-additive interference; ~0 = clean compounding. Pin
  this definition — don't waffle between multiplicative and additive framings.
- **Report uncertainty, not point estimates alone.** With `repeat_idx` up to 3, show the spread
  across repeats on every effect estimate at minimum; bootstrap over requests within a run is
  better if time allows. An effect reported without spread is unfalsifiable — "we found an
  interaction" needs a stated uncertainty to mean anything.
- `plots.py`: (a) per-engine/per-config goodput-vs-concurrency curves with the speedup=1.0 line;
  (b) stacked-savings bars per workload with the interference gap shaded; (c) the K×S and W×S
  interaction plots vs concurrency (the headline cells); (d) SGLang overlap-crossover plot.
- `decision_guide.py`: turn the matrix into "for workload shape X at concurrency Y, stack {…};
  avoid {…} because {mechanism}." This is the deployment decision guide deliverable.

---

## 10. Testing requirements

- Unit test: byte-identical-prefix assertion for the RAG workload (token-ID equality across a
  same-document batch).
- Unit test: result-record schema validation + atomic-write/resume correctness.
- Smoke test: a 1-config × 1-workload × conc=1 × 1-repeat end-to-end run on the smallest GPU
  before launching any sweep. (The full-stack version of this — w4a16+fp8-kv+eagle3 together —
  was run and confirmed working on A100/vLLM 0.24.0; see PREREQ_RESULTS Check 6.)
- **Free correctness regression test:** speculative decoding under greedy decoding is
  output-preserving by construction — at concurrency=1, a spec-on cell's output must match its
  spec-off counterpart (up to numerics). Assert this in CI/harness tests; a mismatch means the
  adapter is broken, independent of anything about the SpecMQuant reproduction gate.
- Sanity: `accepted_length_tau` ≥ 1 and `verif_to_decode_ratio` finite whenever spec is on. Since
  vLLM does not expose `verif_to_decode_ratio` natively, derive it (from τ, acceptance counters,
  and ITL deltas between matched spec-on/spec-off cells) rather than expecting it as a raw field.
