# When LLM Inference Optimizations Collide

**An empirical study of how weight quantization (AWQ W4A16), KV-cache quantization
(FP8), and speculative decoding (EAGLE-3) interact when stacked in vLLM under
continuous batching** — a replicated 2³ factorial across three workloads and four
concurrency levels on a single A100, reproducing published single-stream findings and
extending them into the serving regime where they had never been characterized.
Practitioner guidance says these optimizations compound; this project measured what
actually happens: **speedups interfere (up to x2.97 below the naive product), quality
costs add cleanly, and every pairwise interaction is negative.**

![tests](https://img.shields.io/badge/tests-177%20passing%20(no%20GPU%20needed)-brightgreen)
![runs](https://img.shields.io/badge/measured%20serving%20runs-397-blue)
![validation](https://img.shields.io/badge/advisor%20claims%20validated-11%2F11-brightgreen)
![engine](https://img.shields.io/badge/vLLM-0.24.0%20(pinned)-informational)
![hardware](https://img.shields.io/badge/GPU-A100%20(Colab)-orange)

**Status: complete.** All experimental phases done and verified · [decision guide](DECISION_GUIDE.md) ·
upstream bug found, diagnosed to file:line, and [reported](https://github.com/vllm-project/vllm/issues/48894).

---

**Contents:** [Headline findings](#headline-findings) ·
[Runnable artifacts](#the-runnable-artifacts) · [Study design](#study-design) ·
[Results in depth](#results-in-depth) · [Three debugging stories](#three-debugging-stories) ·
[Repo map](#repo-map) · [Scope and limitations](#scope-and-limitations) ·
[Reading order](#reading-order)

---

## Headline findings

1. **Speedups do not multiply.** The full W+K+S stack delivers x1.30–2.97 *less* than
   the naive product of its parts — worst exactly where each lever alone is strongest
   (batch 1). All pairwise interactions are negative in all 12 workload × concurrency
   cells of the factorial.
2. **Quality costs DO add.** Full-stack accuracy loss equals the sum of individual
   losses within 0.7 points in all 8 measured cells. Speed interferes; quality adds —
   you can predict the stack's accuracy from the levers' report cards, never its speed.
3. **The three levers price quality completely differently.** W4A16 costs −3 to −8
   accuracy points (the only lever with a real quality price); FP8-KV costs ~0;
   EAGLE-3 under greedy decoding is measured bit-identical. Bonus replicated oddity:
   FP8-KV *claws back* +1.7 to +3.5 points of W4A16's HumanEval damage.
4. **Speculative decoding has a context-length cliff.** EAGLE-3's acceptance collapses
   from τ≈2.85 to τ≈1.14 at 7.4k-token contexts, making it a measured net **loss**
   (x0.89–0.94) — confirmed drafter-real by 5+ measurements across two sessions,
   including a fixed-checkpoint retest that ruled out an engine bug.
5. **FP8-KV is a capacity lever wearing a speed lever's costume.** Below the KV
   capacity knee on A100 it's a ~5% tax; at the knee it doubles the admitted batch
   (~17 → ~33–42 requests), converting to +19% goodput and −21% TTFT p95.
6. **A real vLLM 0.24.0 bug**, found via the long-context cells: the EAGLE-3 draft
   checkpoint's `max_position_embeddings=2048` sizes the draft RoPE cache; long prompts
   crash compiled mode and silently read out-of-bounds in eager mode. Diagnosed to
   file:line, GPU-instrumented per-position, reported upstream
   ([vllm#48894](https://github.com/vllm-project/vllm/issues/48894)).

**The practical distillation of all of this is [DECISION_GUIDE.md](DECISION_GUIDE.md)**
— which optimization to enable for your context length, concurrency, and quality
tolerance, with the measured finding behind every rule.

---

## The runnable artifacts

This repo is a study, but it ships four things you can run or reuse today:

1. **[The decision guide](DECISION_GUIDE.md)** — deployment recommendations per
   scenario, each citing the measured cells behind it and stating where the data stops.
2. **`stack-advisor`** — the same guide as an executable CLI. Give it your deployment
   scenario; it recommends a stack with expected effect ranges and provenance. Its
   `--validate` mode recomputes every quantitative claim from the raw run records —
   the guide cannot silently drift from the data:
   ```bash
   python3 -m analysis.stack_advisor --context-tokens 7400 --concurrency 48
   python3 -m analysis.stack_advisor --context-tokens 512 --concurrency 1 --workload code --quality-sensitive
   python3 -m analysis.stack_advisor --validate phase3_results phase3b_results phase3c_diagnostics_results
   # -> 337 records, 11/11 PASS
   ```
3. **The benchmark harness** (`harness/`) — a reusable, engine-agnostic serving
   benchmark: generated configs (never hand-edited YAML), resumable sweeps with
   server-launch amortization, process-group-safe engine lifecycle (survives vLLM V1's
   multi-process EngineCore), a two-signal launch watchdog that doesn't false-kill
   cold model downloads, per-run atomic JSON records with full environment capture,
   and a fake-server test suite — **177 tests, all runnable with no GPU**
   (`python3 -m pytest tests -q`).
4. **An upstream bug report** with a full source-level diagnosis:
   [vllm-project/vllm#48894](https://github.com/vllm-project/vllm/issues/48894) and
   [analysis/vllm_2048_bug_diagnosis.md](analysis/vllm_2048_bug_diagnosis.md) — one
   wrong checkpoint config value producing two different failure modes (hard crash
   under compilation, silent out-of-bounds reads under eager), separated cleanly from
   a real performance finding that initially looked like its symptom.

---

## Study design

**The question.** Practitioner guidance treats PagedAttention + quantization +
speculative decoding as cleanly compounding ("10–50× cheaper"). Research says they
interact — sometimes destructively — and even contradicts itself (QuantSpec: quantized
KV *raises* speculative acceptance; SpecMQuant: 4-bit weights make speculation
*counterproductive*) — but always at batch 1, in bespoke single-stream harnesses.
This project tests who is right inside a production engine, under continuous batching,
across concurrency.

**The method.** A replicated 2³ factorial over the three levers, analyzed with
log-space effects; interference is quantified as the gap between the naive product of
marginals and the measured full stack. Reproduce-then-extend: Phase 1 gates on
reproducing SpecMQuant's published direction before any new measurement is trusted.

| Factor | Off | On |
|---|---|---|
| **W** — weight quant | FP16 | AWQ W4A16 (`awq_marlin`) |
| **K** — KV-cache quant | FP16 KV | FP8 (E4M3) KV |
| **S** — speculative decoding | none | EAGLE-3, 5 draft tokens |

Workloads: GSM8K (reasoning), HumanEval (code, with correctness scoring), shared-prefix
RAG (7.7k-token documents for the long-context cells). Concurrency: 1 / 8 / 32 / 64
closed-loop. Model: Llama-3.1-8B-Instruct. Engine: vLLM 0.24.0 (pinned). Hardware:
single A100 (40GB and 80GB variants, recorded per run, never mixed within a comparison).

**The phases** — 397 measured serving runs total, all committed under `*_results/`:

| Phase | What | Cells | Verdict |
|---|---|---|---|
| 0/1 | Harness + reproduction gate (SpecMQuant's direction in vLLM) | 8 | **PASS** — EAGLE speedup erodes under W4A16: 1.64x→1.26x (GSM8K), 1.89x→1.37x (HumanEval), τ unchanged → mechanism is economic, not acceptance |
| 2 | Single-lever marginals across concurrency | 48 | W and S fade with concurrency at fixed τ; K is a flat tax below capacity |
| 3 | Full replicated 2³ factorial | 288 | All pairwise interactions negative everywhere; interference gap x1.30–2.97; quality adds |
| 3b | K-stress addendum at KV-capacity pressure (40GB) | 40 | FP8-KV doubles admitted batch; AWQ raises the admission ceiling ~17→~27 |
| 3c | Diagnostics + fixed-checkpoint τ retest | 13 | Long-context S collapse is drafter-real; vLLM crash root-caused; attention-backend confound cleared (~0.2%) |

---

## Results in depth

### Speed: each lever alone, then stacked

Single-lever goodput ratios vs. the FP16/FP16-KV/no-spec baseline
([full tables](phase2_results/phase2_marginals_report.md)):

| Lever | conc 1 | conc 8 | conc 32 | conc 64 |
|---|---|---|---|---|
| W (AWQ) | x2.07–2.13 | x1.70–1.95 | x1.26–1.57 | x1.00–1.18 |
| K (FP8-KV) | x0.94 | x0.95 | x0.94–0.98 | x0.97–1.03 |
| S (EAGLE-3) | x1.90–3.16 | x1.61–2.72 | x1.13–1.84 | x0.90–1.37 |

The best measured stack is **W+S at low concurrency**: x3.01 (GSM8K), x5.23
(HumanEval), x2.98 (RAG) at conc 1 — against a naive-product prediction of x4.4–6.7.
Adding K (the "full stack") cuts it roughly in half on A100: x1.40 / x2.28 / x1.38.
Every pairwise interaction in the factorial is negative
([full effects](phase3_results/factorial_report.md)); the WS erosion reproduces
SpecMQuant's finding under batching (and adds: it is *flat-to-shrinking* in
concurrency, falsifying the "amplified under batching" hypothesis), and the KS erosion
happens with τ invariant — QuantSpec's acceptance channel does not exist for EAGLE-3.

### Quality: the axis where the levers differ most

From the computed 2³ accuracy contrasts
([quality_effects.json](phase3_results/quality_effects.json)):

| Lever | GSM8K accuracy | HumanEval accuracy |
|---|---|---|
| W (AWQ) | **−3.0 to −4.0 pts** | **−6.2 to −7.9 pts** |
| K (FP8-KV) | ~0 (straddles zero) | +1.4 to +2.1 pts |
| S (EAGLE-3, greedy) | ~0 (bit-identical) | ~0 (bit-identical) |

Quality does not compound: the full stack's loss ≈ W's loss alone, in every cell.
The WK interaction is *positive* on HumanEval (+1.7 to +3.5 pts, all four
concurrencies) — FP8-KV partially offsets W4A16's damage; mechanism unresolved and
honestly labeled as such.

### The long-context cliff

At 7.4k-token contexts, EAGLE-3's τ collapses 2.85 → 1.14 (~77% of draft compute
discarded) and S becomes a net loss: x0.94 (conc 1), x0.89 (conc 8), with a
supporting compiled-regime point at x0.75. This finding survived the strongest
falsification available: the draft checkpoint's RoPE-cache bug (below) was fixed
locally and the measurement repeated with compilation on — τ did not move
(1.1376–1.1441 across a 4-cell replication). The cliff is a property of the drafter's
training distribution, not the engine.

### The capacity story

At KV-capacity pressure (40GB, ~7.7k-token contexts,
[k_stress report](phase3b_results/k_stress_report.md)): FP16-KV admission plateaus at
~17–19 concurrent requests with preemptions and 30s queue p50; FP8-KV doubles the
admitted batch (~33–42), +17–19% goodput, TTFT p95 −21%, queue p50 → 11s. AWQ
independently raises the ceiling ~17 → ~27 by freeing ~10.4GB of weights for KV
(measured plateau matches the predicted ~26). W and K are capacity levers here —
their speed ratios stay near x1.0 while admission does the work.

---

## Three debugging stories

The kind of thing that only shows up when you run real engines under real load —
each fully documented in [PREREQ_RESULTS.md](PREREQ_RESULTS.md) (the append-only ops
ledger, corrections recorded as corrections):

1. **The 2048 bug** — the EAGLE-3 probe cells crashed vLLM with a device-side assert.
   Root cause (two wrong theories falsified on GPU along the way): the draft
   checkpoint declares `max_position_embeddings=2048`, sizing the draft's RoPE cache;
   compiled kernels assert on the gather past row 2048, while eager mode reads
   out-of-bounds **silently** — GPU instrumentation showed correct values through
   position 2047, then garbage (err ~3e19) from 2048 on. One config value, two failure
   modes, and the garbage happened to be harmless only because the drafter was already
   at its acceptance floor. [Full diagnosis](analysis/vllm_2048_bug_diagnosis.md) ·
   [upstream issue](https://github.com/vllm-project/vllm/issues/48894).
2. **The process-group teardown** — vLLM V1 spawns an EngineCore child; killing only
   the launched PID orphans ~16GB of GPU memory and poisons every later launch in the
   session. The harness kills the process group with TERM→KILL escalation, verifies
   release via `nvidia-smi`, and refuses to launch on an occupied GPU
   ([tests/test_engine_lifecycle.py](tests/test_engine_lifecycle.py)).
3. **The watchdog that cried wolf** — the launch watchdog's first version assumed
   model downloads write progress to the server log. They don't: tqdm silences itself
   on non-tty stdout, so a healthy cold-cache launch looks identical to a wedged one
   for 600+ seconds. The fix is a two-signal watchdog (server log AND HF-cache growth
   must both freeze) — plus, during an HF CDN outage that broke one signing route for
   14+ hours, [scripts/predownload.py](scripts/predownload.py) grew an automatic
   browser-UA curl fallback that reconstructs the HF cache layout byte-for-byte.

---

## Repo map

```
DECISION_GUIDE.md          The deliverable: measured deployment advice, with provenance
analysis/
  stack_advisor.py         The guide as a CLI (+ --validate against raw records)
  factorial.py             Log-space 2^3 effect estimation + interference gap
  quality_factorial.py     The same contrasts on the accuracy axis
  marginals.py, k_stress.py, repro_gate.py
  vllm_2048_bug_diagnosis.md
harness/                   Engine-agnostic serving benchmark (launch, load, sweep, records)
configs/                   GENERATED configs — edit the generate_*.py, never the YAML
tests/                     177 GPU-free tests (fake vLLM server included)
colab/                     The GPU-session notebooks (numbered by phase) + debug archives
block0_results/            Reproduction gate (8 runs)     -> repro_gate_report.md
phase2_results/            Marginals (48 runs)            -> phase2_marginals_report.md
phase3_results/            Full factorial (288 runs)      -> factorial_report.md, quality_effects.json
phase3b_results/           K-stress addendum (40 runs)    -> k_stress_report.md
phase3c_*/                 Diagnostics + retest (13 runs)
scripts/                   predownload.py (CDN-outage-proof), debug_rope_oob.py (GPU probe)
```

---

## Scope and limitations

Stated as precisely as the findings: one model (Llama-3.1-8B-Instruct), one engine
version (vLLM 0.24.0, pinned — two real bugs found at this version), one GPU family
(A100, where FP8 is emulated; native-FP8 hardware is labeled extrapolation wherever it
appears), greedy decoding only, one EAGLE-3 checkpoint (the long-context cliff is a
property of its training distribution). Long-context accuracy under FP8-KV and T>0
sampling are documented gaps, not silent ones. An optional SGLang extension (the
KV-quant × prefix-cache-capacity seam in RAG serving) was scoped with pre-registered
kill criteria and is out of the shipped core.

**Guiding principle throughout:** every claim must earn its place mechanistically, and
every number must be recomputable from committed records — which is what
`stack_advisor --validate` enforces.

---

## Reading order

For the findings: **[DECISION_GUIDE.md](DECISION_GUIDE.md)** → the per-phase reports
linked in the [repo map](#repo-map) → [WRITEUP_NOTES.md](WRITEUP_NOTES.md) (the
findings narrative, including the debugging sagas in full).

For the design rationale: [PROJECT_SPEC.md](PROJECT_SPEC.md) (frozen problem statement
and locked decisions) → [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md) (the factorial,
all seven pre-registered hypotheses, metrics) → [HARNESS_SPEC.md](HARNESS_SPEC.md) →
[LITERATURE.md](LITERATURE.md) (SpecMQuant, QuantSpec, Spec-Bench, and where this
study sits between them).

For the operational history: [PREREQ_RESULTS.md](PREREQ_RESULTS.md) — the append-only
ledger of every empirical gotcha, wrong theory, and correction, kept as a record
rather than rewritten.
