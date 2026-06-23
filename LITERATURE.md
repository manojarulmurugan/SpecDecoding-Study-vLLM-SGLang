# LITERATURE — Prior Work, Sources, and What to Reproduce

All sources gathered during project scoping. Grouped by the role each plays. Verify exact
arXiv IDs/versions when you fetch (some were captured from search and a few are recent
2026 preprints). The two repos in **bold** are load-bearing for implementation.

---

## 1. The interaction papers (the core — these define the seam)

| Paper | ID / venue | What it found | Role here |
|---|---|---|---|
| **QuantSpec** | arXiv 2502.10424, ICML 2025 | Quantized KV cache *raises* draft acceptance (>90%) and improves speedup (~2.5×) — self-speculative, shared-architecture, long-context, **batch-1, edge**. | The "K helps S" result we test under batching (K×S hypothesis). |
| **SpecMQuant** ("Speculative Decoding Meets Quantization") | arXiv 2505.22179, ACL 2025. Repo: **github.com/AI9Stars/SpecMQuant** | 4-bit weight quant makes tree-spec *counterproductive*; once decode is cheap the **verification-to-decoding ratio** dominates. Single-stream, Llama-3-8B + EAGLE-2, A100 & RTX 3090, Spec-Bench-derived harness, GSM8K + HumanEval. | **Reproduction anchor** (Block 0) + the W×S hypothesis. Closest prior work; differentiate carefully. |
| QSpec | arXiv 2410.11305 | Complementary quant schemes for drafting vs verification, exploiting acceptance behavior under quantization. | Context for the design space; not reproduced. |

---

## 2. The serving-regime papers (our methodology comes from here)

| Paper | ID | Key idea to adopt |
|---|---|---|
| **TurboSpec** ("Optimizing Speculative Decoding for Serving LLMs Using Goodput") | arXiv 2406.14066 | **Goodput** metric = rate of *verified-and-generated* tokens/sec; derived from acceptance rate and (emergent) batch size. First spec-decoding-in-vLLM integration. |
| Interpretable Latency Model for Spec Decoding in LLM Serving | arXiv 2605.15051 (2026) | **Batch size is scheduler-dependent, not tunable** — sweep request rate, infer effective batch via Little's Law. The methodological correction that shapes our load design. |
| AdaSpec / SpecServe | arXiv 2503.05096 | Adaptive draft length under load; ~2000 LOC vLLM extension; draft/target KV split. |
| BanditSpec | arXiv 2505.15141 | "Spec decoding does not always yield gains due to batch size and acceptance variation; as batch grows the system becomes compute-bound." Confirms the erosion thesis. |
| SPIRe | arXiv 2504.06419 | Speedup varies with draft architecture, batch size, context length; small drafts for low-latency/small-batch, larger drafts viable at high-throughput/large-batch. |

---

## 3. KV-cache quantization depth

| Source | ID | Use |
|---|---|---|
| KVTuner | arXiv 2502.04420 | Sensitivity-aware layer-wise mixed-precision KV quant — reference for the (optional) per-layer angle. |
| KV-cache compression survey | arXiv 2508.06297 | Taxonomy + citation web entry point. Read its related-work tables first. |
| KIVI / KVQuant / XQuant | (in survey) | Per-channel-key/per-token-value insight; 4-bit maintains accuracy, 2-bit drops. |

---

## 4. The SGLang seam

| Source | ID / link | Use |
|---|---|---|
| SGLang paper | arXiv 2312.07104 | RadixAttention mechanics: radix-tree LRU KV reuse; reduces prefill compute and TTFT; up to 6.4× on shared-prefix workloads. |
| SGLang P-EAGLE issue | github.com/sgl-project/sglang/issues/23171 | Current spec-decoding status; parallel-drafting divergence vs vLLM (optional micro-seam). |
| KVFlow | arXiv 2507.07400 | Workflow-aware prefix-cache mgmt; up to 2.19× over SGLang radix cache on concurrent workflows. Context for agentic prefix sharing. |
| FastSwitch | arXiv 2411.18424 | Context-switching / fairness in KV management; vLLM-paging vs SGLang-radix contrast. |
| KV offloading bottlenecks | arXiv 2601.19910 | Prefix-caching mechanism comparison (vLLM hash-block vs SGLang radix-tree). |

**Critical implementation caveat from this group:** RadixAttention's 6.4× gain "degrades
sharply when prefix ordering is inconsistent" — the radix match fails and every request falls
back to full prefill. The RAG workload MUST use byte-identical, consistently-ordered shared
prefixes (see HARNESS_SPEC).

---

## 5. Reproducibility backbone + engine references

- **Spec-Bench** — github.com/hemingkx/Spec-Bench — the standard spec-decoding benchmark
  harness. Workload subtasks: multi-turn, translation, summarization, QA, math, RAG. **Our
  harness backbone.**
- **EAGLE** — github.com/SafeAILab/EAGLE — draft-head checkpoints (EAGLE-2 / EAGLE-3).
- vLLM speculative-decoding blog — blog.vllm.ai/2024/10/17/spec-decode.html — how spec decode
  integrates with continuous batching; draft runner / target runner.
- vLLM FP8 KV-cache blog (April 2026) — FP8 native vs emulated; the A100 penalty.
- vLLM `benchmarks/benchmark_serving.py` — the serving load driver (request-rate control,
  TTFT/ITL/TPOT percentiles).

---

## 6. The practitioner "consensus" we are testing (cite as the foil, not as truth)

2026 production guides asserting clean compounding (CallSphere, DigitalApplied, Spheron,
MyEngineeringPath, Morph). Keep these specifically as the "stack-them-they-compound" claim the
project tests against. Representative claims: model/system/application optimizations are
"independent and compound, minimal overlap"; the stacked diagram = "10–50× cheaper, no mention
of interference."

---

## 7. Metric definitions to adopt verbatim (credibility + comparability)

- **Goodput** (TurboSpec): verified-and-generated tokens/sec. Primary load-aware speedup
  metric — counts only tokens the target actually keeps, so it correctly captures spec
  decoding's erosion under batching. Raw throughput does NOT.
- **Verification-to-decoding ratio** (SpecMQuant): the diagnostic that *explains* when spec
  decoding stops paying off; >1 means verification compute has overwhelmed the quant memory
  benefit. The mechanistic explanatory variable — turns "what" into "why."
- **Average accepted length (τ)** (EAGLE / Spec-Bench): mean draft tokens accepted per
  verification step. More informative than raw acceptance rate; it is what speedup depends on.
- **Cache hit rate** (for the SGLang seam): fraction of prefix tokens served from the radix
  tree / prefix cache; connects RadixAttention's mechanism to observed TTFT.

---

## 8. What to reproduce, exactly (Block 0 correctness gate)

Reproduce **2–3 cells** of SpecMQuant's single-stream table, not the whole thing:
- `FP16 + EAGLE` (standard spec-decoding speedup), and
- `W4A16 + EAGLE` (their headline "4-bit weights erode spec decoding" finding).

Match: Llama-3-8B, EAGLE-2, greedy decoding, A100, GSM8K + HumanEval, metric = wall-clock
speedup ratio + mean accepted length. **Read their exact numbers from the paper tables / repo
configs — do not take them from memory or from this doc.** Success tolerance: speedup within
~±10–15% AND the *sign* of the W4A16-vs-FP16 effect correct. Document any gap and its likely
cause (vLLM version, kernel differences). See EXPERIMENT_MATRIX §"Baseline protocol."

---

## 9. How to extend this review yourself (you may find what the scan missed)

1. **Start at surveys** (2508.06297 and a model-compression survey) — read related-work tables
   first to build the map before individual papers.
2. **Follow the citation graph** on the three anchors (2502.10424, 2505.22179, 2410.11305) via
   Semantic Scholar / Connected Papers; check *forward citations* (who cites them) to catch
   anything newer than this scan.
3. **Read the SpecMQuant repo** before its paper — a repo tells you what is actually runnable.
4. For engine reality, go to **primary sources**: vLLM docs (quantization, spec-decoding pages),
   vLLM GitHub *issues* (search "INT8 KV cache", "EAGLE3" — issues reveal real vs aspirational
   support), the vLLM blog.
5. **Search-term patterns:** pair the two techniques explicitly ("speculative decoding" +
   "quantization"; "KV cache quantization" + "acceptance rate"), add the year, add a venue for
   quality. Once you find one strong paper, mine its references and citing-papers rather than
   new keyword searches — the graph is denser signal.
6. **Distrust any single headline.** QuantSpec's "quantized KV helps acceptance" holds under
   self-speculative, shared-architecture, long-context, batch-1 conditions — do not generalize.
   The contradiction with SpecMQuant exists *because* conditions differ; pinning down which
   condition flips the sign is the project.
