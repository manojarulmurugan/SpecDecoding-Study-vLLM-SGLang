# Harness quickstart (Phase 0–1 scope)

Implements HARNESS_SPEC.md for the Block-0 reproduction gate: config schema,
atomic/resumable result store, vLLM adapter, closed-loop load driver, GSM8K +
HumanEval workloads, and the gate analysis. Serving-sweep workloads (RAG,
MT-Bench), the SGLang adapter, and `analysis/factorial.py` land in Phase 2+.

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
