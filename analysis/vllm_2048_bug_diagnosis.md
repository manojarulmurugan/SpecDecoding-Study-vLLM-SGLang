# vLLM EAGLE-3 2048-token crash: source-level diagnosis (v0.24.0)

**Status:** FULLY CONFIRMED — source-verified against tag `v0.24.0` AND empirically
instrumented on an A100 (2026-07-18, `scripts/debug_rope_oob.py`; results in the appendix).
Issue filed as vllm-project/vllm#48894; the finalized follow-up comment at the bottom is
ready for the user to post (the user posts, never the assistant).

## The causal chain (file:line, v0.24.0)

1. **The draft model is built from the draft checkpoint's own hf_config.**
   `vllm/model_executor/models/llama_eagle3.py:142`
   ```python
   self.config = vllm_config.speculative_config.draft_model_config.hf_config
   ```
   and each `LlamaDecoderLayer` is constructed with `config=self.config`.
   For `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`, that config declares
   `max_position_embeddings: 2048`.

2. **That value sizes the rotary-embedding cache.**
   `vllm/model_executor/models/llama.py:266`
   ```python
   max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
   ```
   → passed into `LlamaAttention` → `_init_rotary_emb` (`llama.py:243–245`):
   ```python
   self.rotary_emb = get_rope(
       self.head_dim,
       max_position=self.max_position_embeddings, ...)
   ```
   → `vllm/model_executor/layers/rotary_embedding/base.py:97`:
   ```python
   t = torch.arange(self.max_position_embeddings, dtype=torch.float)
   ```
   The draft's `cos_sin_cache` has **exactly 2048 rows**.

3. **The crash site is the position gather into that cache.**
   `vllm/model_executor/layers/rotary_embedding/base.py:173` (`forward_static`,
   the torch-native path):
   ```python
   cos_sin = cos_sin_cache.index_select(0, positions)
   ```
   Under compilation (`support_torch_compile` decorates the EAGLE-3 `LlamaModel`,
   `llama_eagle3.py:125`), Inductor lowers this gather to a Triton kernel with a
   device-side bounds assert — literally the observed
   `index out of bounds: 0 <= tmp < 2048`. Any request whose draft prefill
   reaches position ≥ 2048 fires it.

4. **Why `--enforce-eager` "avoids" it.** [REVISED 2026-07-17 after a
   challenge; see "Dispatch dispute" appendix. Dispatch claim CONFIRMED at
   source + our own logs; the "acceptance degrades" consequence claim was
   WRONG and is corrected here.]
   Under `--enforce-eager`, compilation mode is NONE, so the custom-ops
   default resolves to `'all'` (`vllm/config/vllm.py` `__post_init__`:
   `custom_ops.append("none")` only when `backend == "inductor" and
   mode != NONE`, else `append("all")`) — confirmed in our own successful
   eager server log (`server_20260715_014443.log`: `enforce_eager=True`,
   `mode: <CompilationMode.NONE>`, `custom_ops': ['all']`). With `'all'`,
   `CustomOp.default_on()` is True → `RotaryEmbedding.enabled()` → the
   dispatcher returns `forward_cuda` (`vllm/model_executor/custom_op.py`,
   `dispatch_forward`). `forward_cuda` calls the hand-written CUDA rope
   kernel, whose position gather is raw pointer arithmetic with **no
   bounds check** (`csrc/libtorch_stable/pos_encoding_kernels.cu:92-93`:
   `pos = positions[token_idx]; cache_ptr = cos_sin_cache + pos*rot_dim`).
   So eager mode performs out-of-bounds *reads* past the 2048-row cache —
   undefined behavior that must be fixed regardless of its measured effect.
   **However, the consequence we originally predicted ("acceptance should
   silently degrade") was NOT observed:** the fixed-checkpoint retest
   (max_position_embeddings 2048→8192, compilation ON, no OOB possible)
   measured tau = 1.144 at 7.4k context — statistically identical to the
   eager/stock run (1.1441). The long-context tau collapse is a property
   of the drafter at this context length, not of the OOB reads; whatever
   bytes the OOB gather returns, they made no measurable difference to
   acceptance in our runs. GPU instrumentation
   (`scripts/debug_rope_oob.py`, run 2026-07-18 on the A100) confirmed
   every link empirically — see the appendix's "Instrumentation results".

5. **Why `--max-num-batched-tokens` never mattered.**
   The assert bound is the rope cache size, which comes only from the draft
   checkpoint's config. It is fully decoupled from the scheduler budget.

6. **vLLM computes a 2048 clamp for the draft — then never enforces it.**
   `vllm/config/speculative.py:887` `_maybe_override_draft_max_model_len` sets
   `draft_model_config.max_model_len = min(2048, 8192) = 2048` (and logs
   "Overriding draft model max model len"). But the v1 runtime never consumes
   it: the proposer clamps positions with the **target's** limit
   (`vllm/v1/spec_decode/llm_base_proposer.py:79`,
   `self.max_model_len = vllm_config.model_config.max_model_len` → 8192), and no
   v1 code path reads `draft_model_config.max_model_len` at schedule/run time.
   So the engine neither sizes the cache for 8192 nor stops speculating at 2048.

7. **Still present on `main`** (checked 2026-07-17): `vllm/config/speculative.py`
   on main has no `max_position_embeddings` handling for draft configs.

## Corroboration

- vLLM issue #21986 (v0.10.0, same model pair, same assert) — the reporter's
  hypothesis is exactly this; auto-closed stale, never triaged.
- SpecForge issue #249 (sgl-project): draft checkpoints trained with
  `max_position_embeddings: 2048` as a training artifact; editing the draft
  `config.json` to the target's value restored normal acceptance lengths.
  Supports the position that the checkpoint value is not a real architectural
  limit for EAGLE drafts (rope is parametric; the draft shares the target's
  positional space).

## Empirical proof / immediate workaround (no code change)

Download the draft checkpoint locally, edit its `config.json`
`max_position_embeddings` from 2048 → 8192 (or 131072 to match the target),
and point `--speculative-config` at the local path. If the crash at 2048
disappears with compilation on, the diagnosis is proven end-to-end. This is
also the immediate unblock for experiments.

## Fix sketch

**Where:** `SpeculativeConfig.__post_init__` in `vllm/config/speculative.py`,
after `draft_model_config` is constructed (there is already an
`hf_overrides=SpeculativeConfig.hf_config_override` hook doing draft-config
surgery — a natural home). Gated to methods that share the target's positional
space (`eagle`, `eagle3`, and arguably `mtp`), NOT independent draft models:

```python
# EAGLE-style drafts operate in the target's positional space; the
# checkpoint's max_position_embeddings is a training-time artifact and
# under-sizes the draft's RoPE cache (see #21986 / #48894).
if self.method in ("eagle", "eagle3"):
    draft_hf = self.draft_model_config.hf_config
    target_len = self.target_model_config.max_model_len
    if getattr(draft_hf, "max_position_embeddings", 0) < target_len:
        draft_hf.max_position_embeddings = target_len
```

(Alternative placement: inside `hf_config_override`. Alternative value:
target hf_config's `max_position_embeddings` instead of serving
`max_model_len`; serving length is the minimal safe bound.)

**Size/risk:** ~5 lines + a unit test asserting the override for an
eagle3 draft config. Contained; rope cache memory cost is trivial.

**Known caveats to raise in the PR (honestly):**
- Whether the draft should also inherit the target's `rope_parameters` /
  `rope_scaling` (Llama-3.1 target uses llama3 rope scaling; the EAGLE-3 draft
  config has none) is a related-but-separate correctness question. Maintainers
  may want it addressed together; that's where scope could grow.
- Must not change behavior for independent-draft-model speculation, where a
  genuinely small context is real — hence the method gate.
- A defensive alternative (disable speculation for requests past the draft's
  limit) exists but throws away EAGLE's value on long contexts; the override is
  the right fix given the SpecForge evidence.

---

## DRAFT — GitHub comment for vllm-project/vllm#48894 (and cross-post note to #21986)

> **Root cause found — draft checkpoint's `max_position_embeddings` sizes the draft's RoPE cache; nothing overrides or enforces it.**
>
> I traced the `index out of bounds: 0 <= tmp < 2048` device-side assert to its source in v0.24.0:
>
> 1. The EAGLE-3 draft model is built from the **draft checkpoint's own hf_config** (`vllm/model_executor/models/llama_eagle3.py:142`). `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` declares `max_position_embeddings: 2048`.
> 2. That value flows through `LlamaAttention` → `get_rope(max_position=...)` (`vllm/model_executor/models/llama.py:266, 243–245`), so the draft's `cos_sin_cache` has exactly 2048 rows (`vllm/model_executor/layers/rotary_embedding/base.py:97`).
> 3. The crash is the gather `cos_sin_cache.index_select(0, positions)` (`base.py:173`). Under compilation Inductor emits a Triton bounds assert — hence the exact `< 2048` message the moment any draft position reaches 2048. This is fully decoupled from `--max-num-batched-tokens`, which is why raising it never helped.
> 4. `--enforce-eager` only *hides* the bug: with compilation mode NONE the custom-ops default resolves to `'all'` (`vllm/config/vllm.py` `__post_init__`; visible in our eager server log as `custom_ops': ['all']`), so `RotaryEmbedding` dispatches to `forward_cuda`, whose position gather is unchecked pointer arithmetic (`csrc/libtorch_stable/pos_encoding_kernels.cu:92-93`) — a silent out-of-bounds *read* for every draft position ≥ 2048. We instrumented this directly on an A100 (standalone `get_rope(max_position=2048)`, per-position comparison against independent RoPE math, subprocess-isolated): `forward_cuda` returns correct values through position 2047 (max abs err ≤ 4e-5) and **garbage from 2048 on with no error raised** (max abs err ~3e19 at 2048/2049, ~3.4 at 3000–7399), while an 8192-row control cache is correct at every position, and `forward_native` on the same 2048-row cache raises a device-side assert at exactly 2048 — which also proves the eager dispatch by elimination, since real eager serving runs never crash. Two honest empirical notes: (a) the OOB reads did not measurably change end-to-end acceptance in our workload — a retest with the draft config's `max_position_embeddings` raised to 8192 and compilation ON reproduced the same accepted length (tau ≈ 1.14 at ~7.4k-token context, 4/4 cells) as the eager/stock runs, so for this drafter the low long-context acceptance is a drafter property, not OOB corruption; (b) that makes the OOB read no less worth fixing — it is undefined behavior whose blast radius depends on allocator layout, not on our luck with it.
> 5. vLLM even computes `draft max_model_len = min(2048, target) = 2048` (`vllm/config/speculative.py:887`), but the v1 runtime never consumes it — the proposer clamps positions with the **target's** `max_model_len` (`vllm/v1/spec_decode/llm_base_proposer.py:79`). So the engine neither sizes the cache correctly nor stops speculating at 2048.
>
> This confirms the hypothesis from #21986. Corroborating evidence that the checkpoint value is a training artifact rather than a real limit: sgl-project/SpecForge#249, where editing the draft `config.json` to the target's value restored normal acceptance lengths.
>
> **Workaround:** local copy of the draft checkpoint with `max_position_embeddings` raised to the target's serving length.
>
> **Proposed fix:** in `SpeculativeConfig.__post_init__` (or `hf_config_override`), for `method in ("eagle", "eagle3")`, raise `draft hf_config.max_position_embeddings` to at least the target's `max_model_len` before the draft model is built. Happy to send a PR if maintainers agree with this direction — the open question is whether the draft should also inherit the target's `rope_parameters`.

---

## Appendix: dispatch dispute and resolution (2026-07-17)

A review challenged step 4's dispatch claim: reading
`CustomOp.dispatch_forward()`/`enabled()`, with `custom_ops = ['none']`
RotaryEmbedding would route to `forward_native` (checked `index_select`),
not `forward_cuda` — and a server log appeared to show `['none']` under
eager. Resolution, verified against the v0.24.0 tree and our own logs:

- The `custom_ops: ['none']` observation came from the **crashing compiled
  server's** log (`server_20260715_004237.log`: `enforce_eager=False`,
  `mode: VLLM_COMPILE`). The **successful eager** server's log
  (`server_20260715_014443.log`) shows `enforce_eager=True`,
  `mode: NONE`, `custom_ops': ['all']`.
- The resolution point is `VllmConfig.__post_init__`
  (`vllm/config/vllm.py`): `'none'` is appended only when
  `backend == "inductor" and mode != CompilationMode.NONE`; otherwise
  `'all'`. `--enforce-eager` forces mode NONE → `'all'` → `enabled()` True
  → `forward_cuda`.
- Independent consistency check: if eager HAD dispatched to
  `forward_native`, the OOB `index_select` would device-assert on CUDA the
  same way compiled mode does — and our eager runs completed. The absence
  of an eager crash is itself evidence for the unchecked-kernel path.
- What the challenge DID catch: our original consequence claim
  ("acceptance should silently degrade") was an unverified prediction, and
  the fixed-checkpoint retest falsified it (tau identical, 1.144 vs
  1.1441). Step 4 and the draft comment now state the measured reality.

### Instrumentation results (2026-07-18, A100-40GB — DISPUTE CLOSED)

`scripts/debug_rope_oob.py` ran on the real GPU (each probe in its own
subprocess). Verified against the notebook output, not summarized:

- **control** (8192-row cache, correct sizing): CORRECT at every tested
  position including 2048/2049/3000/4096/7399 (max_abs_err ≤ 8.4e-5) —
  the methodology itself is sound.
- **cuda** (2048-row cache = the stock checkpoint): CORRECT through 2047
  (≤ 4.1e-5), then **GARBAGE with no error raised**: max_abs_err
  3.1e19 @ 2048, 5.0e19 @ 2049, then ~3.20–3.41 absolute at
  3000/4096/7399. Direct observation of the silent OOB read.
- **native** (the checked path): CORRECT at 2047, **RAISED a device-side
  assert at exactly 2048** — the compiled-mode crash signature. Since
  real eager runs never crashed, this excludes `forward_native` and
  proves the `forward_cuda` dispatch by elimination, independently of the
  source/log analysis above.
- **dispatch** probe errored on a script bug (`CustomOp.enabled()` needs
  an active `set_current_vllm_config` context; fixed 2026-07-18). Its
  "resolves to None / dispatches to: None" line in the original output is
  a meaningless artifact of that crash — never quote it. The probe is no
  longer load-bearing: cuda+native together settle dispatch.

**Hold lifted: the draft comment below is finalized with these observed
values and is ready for the user to post.**
