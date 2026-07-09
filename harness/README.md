# Harness quickstart (Phase 0–2 scope)

Implements HARNESS_SPEC.md through Phase 2: config schema, atomic/resumable
result store, vLLM adapter, closed-loop load driver, GSM8K + HumanEval +
RAG-shared-prefix workloads, emergent-batch-size sampling, goodput, the
Block-0 gate analysis, and the Phase-2 marginals analysis. Still to come
(Phase 3+): the remaining factorial corners' analysis (`analysis/factorial.py`
with log-space interaction contrasts), repeats/spread reporting, the SGLang
adapter, and MT-Bench.

## Phase-2 additions

- `harness/workloads/rag_shared_prefix.py` — byte-identical shared prefixes
  (system → document → question, fixed order), `prefix_overlap` knob
  (low/mid/high → 1/8/32 questions per document), sources: Spec-Bench `rag`
  subtask / JSONL / synthetic clean-room docs. `check_shared_prefix_token_ids`
  is the token-ID-equality guard; `scripts/check_rag_prefix.py` runs it with
  the real tokenizer before a sweep.
- `harness/sampling.py` — polls `vllm:num_requests_running` during the timed
  window; `measured.emergent_batch_size` = {mean, p50, max, num_samples}.
  Concurrency is SET, batch size is MEASURED (PROJECT_SPEC §7.2).
- `measured.goodput_tok_s` — TurboSpec goodput: verified-and-generated
  tokens/sec. Client-side completion tokens are exactly the kept tokens, so
  goodput = completion tokens / wall; `spec_rejected_tok_s` quantifies the
  rejected-draft waste from the counter deltas.
- `configs/factorial/cube_*.yaml` — the FULL core-factorial board (written by
  `configs/factorial/generate_phase2.py` — edit the generator, not the files):
  all 8 corners × 3 workloads × concurrency {1,8,32,64} × repeats {0,1,2}
  = 288 cells, num_requests scaled per concurrency (64/160/320/512).
  Sweep-time globs pick the subset to run; the marginals runbook selects the
  four single-factor corners at repeat 0 (`cube_{base,w,k,s}_*_r0.yaml`, 48
  cells). Repeats share a server launch (fixed seed; the request stream is
  reshuffled per repeat via seed+repeat_idx in run.py).
- `analysis/marginals.py` — the Phase-2 report: goodput vs concurrency per
  optimization, speedup vs baseline, emergent batch, tau.
- `analysis/factorial.py` — the core 2^3 analysis in log space: main effects,
  pairwise + three-way contrasts, and the interference gap (naive
  product-in-logs vs measured full-stack gain), per workload × concurrency,
  with min..max spread across complete repeats. Statistics verified against
  synthetic cubes with known injected effects (`tests/test_factorial.py`).
- Colab runbook: `colab/phase2_marginals.ipynb`.

## K-stress addendum (Phase-3 Session D)

Phase 2's flat K marginal is a regime-specific null: demand peaked at ~18% of
the KV pool. `configs/k_stress/` (16 cells, written by
`generate_k_stress.py`) recreates K's capacity channel: unique ~7.4k-token
documents (`doc_target_tokens` sizing in the RAG workload, overlap low),
{FP16-KV, FP8-KV} × conc {8,16,32,48} × 2 repeats, FP16 weights, no spec —
on a **pinned A100-40GB** (Colab High-RAM OFF; the toggle selects the
variant, PREREQ_RESULTS Check 1), where both ceilings (~16 FP16 / ~32 FP8)
fit inside the grid. Cube/factorial sessions pin High-RAM ON (80GB) to match
Phase 2's records; `analysis/factorial.py` warns on mixed-hardware cubes.
Runbook: `colab/phase3b_kstress_40gb.ipynb`.
`harness/sampling.py` now samples queue depth and `vllm:kv_cache_usage_perc`
alongside the running batch, and run.py deltas `vllm:num_preemptions` — the
three signals that make "the ceiling was hit" a measurement instead of an
inference. Analysis: `analysis/k_stress.py` (goodput ratio, plateau vs
predicted pool/context, capacity-limited verdicts). Records carry
`block: k_stress` so the factorial/marginals analyses ignore them.

## Layout

- `harness/config.py` — one YAML = one run cell; deterministic `run_id`;
  validation (rejects `spec_decode: eagle2` — see PREREQ_RESULTS Check 3
  resolution).
- `harness/engines/vllm_adapter.py` — builds the `vllm serve` command from the
  config factors (pin: `vllm==0.24.0`, Check 6). Command construction is pure
  and fully unit-tested without a GPU.
- `harness/load.py` — closed-loop concurrency driver over streaming
  `/v1/completions`; TTFT/ITL per request; token counts from `usage` (chunk
  counting under-counts with spec decoding).
- `harness/metrics.py` — percentiles, throughput, and τ = 1 + accepted/drafts
  from vLLM's Prometheus counters, delta'd around the timed window.
- `harness/run.py` / `harness/sweep.py` — one cell / grouped cells (one server
  launch per distinct launch command), atomic records, resume-by-skipping.
- `analysis/repro_gate.py` — Block-0 verdict: fails on direction, warns on
  magnitude (EXPERIMENT_MATRIX §7, revised tolerance).

## Run locally (no GPU)

```bash
python3 -m pytest tests -q                 # 70+ tests incl. fake-server e2e
python3 -m harness.sweep "configs/repro/repro_*.yaml" --dry-run
```

## Run Block 0 (Colab, A100)

Open `colab/block0_repro_gate.ipynb` and run top to bottom. It uses the
isolated-virtualenv recipe from PREREQ_RESULTS Check 6 — do not bare-pip vLLM
into the notebook kernel.

## Documented deviations from HARNESS_SPEC

- Load driver is our own streaming client, not `vllm bench serve`: the gate
  needs generated text for correctness scoring (see PREREQ_RESULTS, Check 3
  resolution, "documented deviation").
- `enable_prefix_caching` lives under `engine_args` (it is a server-launch
  flag) rather than top-level; it is explicit, defaulted off for controlled
  cells, and recorded verbatim in every result record.
