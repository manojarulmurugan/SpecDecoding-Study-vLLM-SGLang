# Follow-up PR plan: wrapper/inner consistency hardening (after #49343 merges)

**Decision (2026-07-22):** the recommended target is options 1+2 COMBINED — a
single small "harden the fix" PR: sync the wrapped inner `EAGLEConfig.model`
config in `_maybe_override_draft_max_position_embeddings`, plus the two test
cases #49343 lacks. Filed AFTER #49343 merges (same helper, same file —
parallel filing guarantees a conflict and reads as competing). Option 3
(rope_scaling inheritance) is parked as future work — see bottom.

Fact checked 2026-07-22: `SpeculatorsConfig.from_pretrained` flattens
`transformer_layer_config` into top-level attributes
(`vllm/transformers_utils/configs/speculators/base.py`), so #49343's override
DOES cover speculators-format checkpoints live. The stale-inner gap is
EAGLEConfig-only and latent. The PR must be framed honestly as consistency
hardening + test coverage, NOT as a live-bug fix.

## The patch (already written and verified)

Prepared on top of #49343's applied diff, in the scratchpad snapshot:

- Code: `scratchpad/followup_speculative.diff` — 10 lines appended to
  Jared's `_maybe_override_draft_max_position_embeddings`: after setting the
  outer config, sync `draft_hf_config.model.max_position_embeddings` if that
  inner config exists and has the attribute.
- Tests: `scratchpad/followup_tests.diff` — two tests added to #49343's own
  `tests/config/test_speculative_draft_max_position_embeddings.py`:
  - `test_override_reaches_wrapped_inner_config` — direct-loading EAGLEConfig
    case; asserts wrapper AND inner both end at 8192.
  - `test_override_tolerates_wrapper_without_inner_max_position` — a config
    with a non-config `model` attribute doesn't break the override.
- Verified locally: both new tests pass (venv + torch-2.6 shim,
  `--noconftest`); ruff check + format clean. #49343's own three integration
  tests fail LOCALLY only because this mac can't run vLLM's model-registry
  inspection (torch 2.6 ceiling on x86_64 macOS) — they are upstream-CI
  territory, unrelated to this patch.

Full patched files (post-49343 + follow-up) live in
`scratchpad/vllm-main-snapshot/` — but regenerate against real main after
#49343 merges rather than trusting the snapshot; the tree drifts.

## Filing checklist (user runs everything)

1. Wait for #49343 to merge. Watch `v1-spec-decode` in its CI rerun.
2. `git clone` fresh main (or pull), create branch, apply the two diffs (or
   hand-copy the ~10 code lines + 2 tests — they're small).
3. `pytest tests/config/test_speculative_draft_max_position_embeddings.py`
   (all of it, on a machine that can — or lean on CI), `pre-commit run`.
4. **DCO: commit with `git commit -s`** (vLLM requires Signed-off-by, no CLA).
5. PR title: `[BugFix] Keep wrapped EAGLE draft config consistent when
   overriding max_position_embeddings`
6. Tag @JaredforReal, reference the #49343 review thread where the follow-up
   was claimed.

## PR body draft

```markdown
Follow-up to #49343 (fix for #48894), as discussed in that PR's review.

#49343 raises an EAGLE draft's `max_position_embeddings` to the target's
`max_model_len` before the `EAGLEConfig` wrapping step, which keeps the
wrapper and the wrapped checkpoint config consistent for drafts that get
wrapped. However, checkpoints whose config already loads *as* `EAGLEConfig`
skip the wrapping branch, so the override reaches only the outer object and
the wrapped `.model` config keeps the stale value (e.g. 2048).

Nothing in `vllm/` reads the inner config's `max_position_embeddings` today,
so this is a consistency hardening, not a live bug: `EAGLEConfig` mirrors the
wrapped config's attributes onto itself, and the two objects should never
disagree about the positional range — a future reader (or any code that
re-materializes the config from its inner `model`) would otherwise
re-introduce exactly the #48894 under-sized RoPE cache.

Changes:
- `_maybe_override_draft_max_position_embeddings` also syncs
  `draft_hf_config.model.max_position_embeddings` when that inner config
  exists and declares the attribute (10 lines).
- Two tests in the existing test file: the direct-loading `EAGLEConfig` case
  (wrapper and inner both raised), and robustness when a config exposes a
  non-config `model` attribute.

CPU-only, no GPU needed; `pytest tests/config/test_speculative_draft_max_position_embeddings.py`.
```

## Honest odds & sequencing rationale

- Combined fix+tests, filed after merge, referencing an on-record review
  discussion, tagged to an engaged maintainer: ~60–70% merged, realistic
  window 3 days–3 weeks. Cold-filed without the review thread: ~30–40% and
  may sit for weeks (vLLM has thousands of open PRs; unshepherded first-timer
  PRs routinely go stale).
- Biggest risk: a maintainer folds the suggestion into #49343 instead. That
  is why the review comment flags the gap but does NOT include the patch, and
  explicitly claims the follow-up (review notes section D, revised).
- Option 1 alone reads as cosmetic; option 2 alone reads as drive-by; the
  combination is a coherent hardening PR. Option 3 (rope_parameters
  inheritance) needs GPU A/B measurement of acceptance with/without the
  target's rope scaling injected into the draft — weeks, not days. Note for
  FUTURE_WORK: the mismatch is a candidate mechanism for the study's
  long-context tau collapse (draft runs unscaled RoPE while the Llama-3.1
  target runs llama3-scaled RoPE — geometry diverges with depth), which would
  make it a measurement-backed issue/PR later.
