# vLLM EAGLE-3 2048-token crash: source-level diagnosis (v0.24.0)

**Status:** hypothesis CONFIRMED against vLLM v0.24.0 source (tag `v0.24.0`, shallow clone).
Nothing here has been posted anywhere — draft comment for the GitHub issues is at the bottom.

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

4. **Why `--enforce-eager` "avoids" it.**
   Eager mode dispatches to `forward_cuda` (`base.py:221`), the hand-written CUDA
   rope kernel, which does **unchecked** pointer arithmetic into `cos_sin_cache`.
   No bounds check → no assert → out-of-bounds *read*. The final output text is
   still correct (rejection sampling is lossless), but draft tokens at positions
   ≥ 2048 get garbage cos/sin, so their acceptance rate should degrade — eager
   mode hides the bug, it does not fix it.

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
> 4. `--enforce-eager` only *hides* the bug: the eager path uses the CUDA rope kernel (`base.py:221`), which reads out of bounds unchecked. Output stays correct (rejection sampling), but drafts at positions ≥ 2048 get garbage rotations, so acceptance should silently degrade on long contexts.
> 5. vLLM even computes `draft max_model_len = min(2048, target) = 2048` (`vllm/config/speculative.py:887`), but the v1 runtime never consumes it — the proposer clamps positions with the **target's** `max_model_len` (`vllm/v1/spec_decode/llm_base_proposer.py:79`). So the engine neither sizes the cache correctly nor stops speculating at 2048.
>
> This confirms the hypothesis from #21986. Corroborating evidence that the checkpoint value is a training artifact rather than a real limit: sgl-project/SpecForge#249, where editing the draft `config.json` to the target's value restored normal acceptance lengths.
>
> **Workaround:** local copy of the draft checkpoint with `max_position_embeddings` raised to the target's serving length.
>
> **Proposed fix:** in `SpeculativeConfig.__post_init__` (or `hf_config_override`), for `method in ("eagle", "eagle3")`, raise `draft hf_config.max_position_embeddings` to at least the target's `max_model_len` before the draft model is built. Happy to send a PR if maintainers agree with this direction — the open question is whether the draft should also inherit the target's `rope_parameters`.
