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
engine_version: "<PINNED>"                  # recorded, not assumed
model: meta-llama/Llama-3.1-8B-Instruct     # Llama-3-8B for repro block
factors:
  weight_quant: w4a16                       # fp16 | w4a16 | w8a8 | w4a8
  kv_quant: fp8                             # fp16 | fp8 | int8
  spec_decode: eagle3                       # none | eagle3 | eagle2
draft_model: "<EAGLE3 head id>"             # null if spec_decode=none
workload: rag_shared_prefix                 # gsm8k | humaneval | rag_shared_prefix | mt_bench
workload_params:
  prefix_overlap: high                      # low | mid | high  (RAG only)
  num_requests: 200
  input_len_mode: natural                   # fixed | natural
  output_len_cap: 512
concurrency: 32                             # SET. batch size is MEASURED.
decoding: greedy                            # greedy (controlled) | sampling (cross-check)
seed: 1234
repeat_idx: 2
warmup_requests: 16
gpu_target: h100                            # a100 | h100 | l4   (FP8 cells → h100)
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

- **vLLM adapter:** launch with weight-quant (AWQ), `--kv-cache-dtype fp8`, speculative config
  (EAGLE-3 draft + max draft len), and the `--disable-spec-on-high-load` style guard left OFF
  for controlled cells (we WANT to observe erosion). Drive load with `benchmark_serving.py`.
- **SGLang adapter:** launch with `--quantization` (weights) + KV-quant flag + EAGLE spec; same
  load driver; additionally collect radix-tree cache hit rate.

**Amortize server startup:** `sweep.py` groups runs by config (the 8 factorial configs), launches
the server once per config, and drives all workload × concurrency × repeat cells against it before
teardown. Only ~8 launches for the core factorial.

---

## 6. Load generation (`load.py`) — the methodological core

- Independent variable is **offered load**: either a fixed pool of N concurrent clients
  (closed-loop) or a Poisson arrival process at rate λ (open-loop). Prefer closed-loop
  concurrency {1,8,32,64} for the controlled factorial; optionally add an open-loop λ sweep for
  the realistic cross-check.
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
  level** — so concurrency-dependence of each interaction is explicit.
- Key derived quantity: **interference gap** = (naive product/sum of individual wins) − (measured
  combined win). Positive gap = sub-additive interference; ~0 = clean compounding.
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
  before launching any sweep.
- Sanity: `accepted_length_tau` ≥ 1 and `verif_to_decode_ratio` finite whenever spec is on.
