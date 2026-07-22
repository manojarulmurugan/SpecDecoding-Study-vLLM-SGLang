# PR handoff for vllm-project/vllm — fix #48894 (EAGLE draft RoPE cache under-sized)

Working tree with the change (NOT a git checkout — a `main` snapshot from
2026-07-20, tarball-extracted):
`/private/tmp/claude-503/-Users-manoja-Documents-GitHub-SpecDecoding-Study-vLLM-SGLang/4ac7db5d-9056-4013-b5b7-489ceb3afb4a/scratchpad/vllm-main-snapshot`

Files changed:
- `vllm/config/speculative.py` — new static helper
  `_maybe_raise_draft_max_position_embeddings` + one call in `__post_init__`
  (inserted just before the `_verify_and_get_draft_tp` /
  `_maybe_override_draft_max_model_len` block)
- `tests/config/test_speculative_draft_max_position_embeddings.py` — new,
  8 CPU-only tests, style-matched to
  `tests/config/test_speculative_draft_hf_overrides.py`

A clean unified diff is at
`.../scratchpad/fix.patch` (same directory tree). Apply to your own clone with
`git apply` after path-fixing, or just copy the two files.

Verification done locally (macOS x86_64, no GPU):
- `pytest tests/config/test_speculative_draft_max_position_embeddings.py`
  → 8 passed (run with `--noconftest`; the root tests/conftest.py imports the
  full engine, which needs torch 2.11 — unavailable on mac x86_64. Upstream CI
  will run it with the real conftest.)
- `pytest tests/config/test_speculative_draft_hf_overrides.py` → 6 passed
  (no regression in the neighboring draft-config tests)
- `ruff check` and `ruff format --check` clean on both files

---

## PR title

```
[Bugfix] Raise EAGLE draft max_position_embeddings to the target's max_model_len
```

## PR description

```markdown
FIX #48894 (also the earlier, auto-closed report of the same bug: #21986)

## Root cause

EAGLE/EAGLE-3 draft models are built from the draft checkpoint's own
`hf_config` (`vllm/model_executor/models/llama_eagle3.py`), and that config's
`max_position_embeddings` sizes the draft's rotary-embedding cos/sin cache
(`llama.py` → `get_rope(max_position=...)` →
`rotary_embedding/base.py: torch.arange(max_position_embeddings)`).

`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` declares `max_position_embeddings: 2048`
— a training-time artifact of how the drafter was trained, not a real
architectural limit (EAGLE drafts operate in the target model's positional
space; corroborated by sgl-project/SpecForge#249, where patching the draft
config to the target's value restored normal acceptance lengths).

The result: the draft's RoPE cache has exactly 2048 rows while the proposer
feeds it positions up to the *target's* `max_model_len`
(`vllm/v1/spec_decode/llm_base_proposer.py` clamps with
`vllm_config.model_config.max_model_len`). As soon as any draft position
reaches 2048:

- **compiled** (default): Inductor's bounds assert on the
  `cos_sin_cache.index_select(0, positions)` gather fires —
  `device-side assert ... index out of bounds: 0 <= tmp < 2048`, killing the
  engine on any prompt longer than ~2048 tokens;
- **eager** (`--enforce-eager`): the CUDA rope kernel
  (`csrc/pos_encoding_kernels.cu`) does the same gather with unchecked pointer
  arithmetic, i.e. an out-of-bounds read. In our runs this happened to be
  benign for output quality (spec decoding is lossless and measured acceptance
  was unchanged), but it is undefined behavior and should not be relied on.

This is fully decoupled from `--max-num-batched-tokens`, which is why raising
the scheduler budget never helps (verified during the bisection in #48894).

## Fix

In `SpeculativeConfig.__post_init__`, after the draft config is finalized
(including the `EAGLEConfig` wrapping), raise the draft `hf_config`'s
`max_position_embeddings` to the target's serving `max_model_len` when it is
smaller. Both the `EAGLEConfig` wrapper and the wrapped inner config are
updated, since `EAGLEConfig` copies the checkpoint attributes onto itself.

**Gated to `method in ("eagle", "eagle3")` only**: an independent draft model
(`method="draft_model"`) may genuinely have a smaller context window than the
target, and clamping/skipping speculation is the correct behavior there. Only
EAGLE-style drafts are known to share the target's positional space.

## Validation

- Repro from #48894: Llama-3.1-8B-Instruct target (`--max-model-len 8192`) +
  EAGLE3 draft, long-document prompt, compilation on → crashes at position
  2048 without this change.
- Equivalent of this fix validated on an A100: serving the same setup with a
  local copy of the draft checkpoint whose `config.json`
  `max_position_embeddings` was patched 2048 → 8192, compilation ON, no longer
  crashes at long context (and produced acceptance statistics consistent with
  the eager baseline).
- New CPU-only unit tests cover: the raise for eagle3, wrapper+inner
  consistency for `EAGLEConfig`, no-op when the draft already covers the
  target length, no-op for non-EAGLE methods, and no-op when the attribute is
  absent.

## Two related issues deliberately NOT addressed here (scoping)

1. **`rope_parameters` / `rope_scaling` inheritance.** The Llama-3.1 target
   uses llama3 rope scaling while the EAGLE-3 draft config declares none, so
   even with a correctly sized cache the draft's rope geometry differs from
   the target's beyond the original context window. Whether the draft should
   inherit the target's rope parameters wholesale is a separate correctness
   question that deserves its own discussion — happy to follow up if
   maintainers have an opinion.
2. **The dead draft `max_model_len` clamp.** `SpeculativeConfig` already
   computes `draft max_model_len = min(draft, target)`
   (`_maybe_override_draft_max_model_len`), but no v1 runtime code consumes
   it — both proposer implementations clamp positions with the *target's*
   `max_model_len`. So today the engine neither sizes the draft cache
   correctly (fixed here) nor stops speculating at the draft's declared
   limit. This PR fixes only the cache-sizing bug; the unused clamp is left
   as-is.
```

---

## Draft follow-up comment for #21986 (optional, after PR is open)

```markdown
This was re-reported with a full source-level root cause in #48894 and a fix
is proposed in <PR link>: the draft checkpoint's `max_position_embeddings`
(2048) sizes the draft's RoPE cos/sin cache, while the EAGLE proposer feeds it
positions up to the target's `max_model_len` — the compiled path device-asserts
and the eager path reads out of bounds. Your original hypothesis was correct.
```
