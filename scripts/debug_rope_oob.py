"""Empirically settle what vLLM's eager-mode RoPE does at positions >= 2048.

Context (analysis/vllm_2048_bug_diagnosis.md): the EAGLE-3 draft head's RoPE
cache has 2048 rows (draft config max_position_embeddings). Compiled mode
device-asserts on the OOB gather. What EAGER mode does was disputed:
 - diagnosis v1: dispatches to forward_cuda (unchecked custom kernel) ->
   silent OOB reads, garbage rotations;
 - proposed correction: custom_ops=['none'] under eager -> forward_native
   (checked index_select) -> would NOT read garbage.
Source + our own eager server log (custom_ops': ['all'], enforce_eager=True,
mode NONE; resolution at vllm/config/vllm.py __post_init__) say v1 is right
about DISPATCH — this script verifies dispatch AND the returned values on a
real GPU instead of trusting either chain of reasoning.

Run inside the Colab vLLM venv (needs a GPU):
    /content/vllm_env/bin/python scripts/debug_rope_oob.py

Three probes, each in its OWN subprocess (a device-side assert poisons the
CUDA context, so they must not share one):
  A. dispatch  — build a VllmConfig with compilation mode NONE (what
     --enforce-eager produces), instantiate get_rope(max_position=2048)
     under it, and report the resolved custom_ops + which forward_* method
     the dispatcher actually selected. No OOB calls.
  B. cuda      — call forward_cuda with in-bounds positions (control: must
     match an independent math reference) and OOB positions 2048..7399
     (compare against the reference a correctly-sized cache would use).
     Reports per-position max-abs-error -> correct / garbage / crash.
  C. native    — call forward_native with one OOB position; report whether
     it raises (checked gather) or returns silently.

The verdict block at the end states, in plain words, what eager mode does.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys

IN_BOUNDS = [0, 100, 2040, 2047]
OOB = [2048, 2049, 3000, 4096, 7399]
HEAD_SIZE = 64          # single head keeps the tensors tiny; rope math is
NUM_TOKENS = 1          # per-position anyway
CACHE_LIMIT = 2048
FULL_LIMIT = 8192


def _reference_cos_sin(pos: int, head_size: int, base: float = 10000.0):
    """Independent RoPE math (no vLLM): interleaved-half (neox) layout."""
    import torch
    half = head_size // 2
    inv_freq = torch.tensor(
        [1.0 / (base ** (2 * i / head_size)) for i in range(half)],
        dtype=torch.float64)
    angles = pos * inv_freq
    return torch.cos(angles), torch.sin(angles)


def _reference_rotate(q, pos: int, head_size: int, base: float = 10000.0):
    """Apply neox-style rotation to q [head_size] at position pos."""
    import torch
    cos, sin = _reference_cos_sin(pos, head_size, base)
    half = head_size // 2
    q = q.to(torch.float64)
    q1, q2 = q[:half], q[half:]
    return torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin])


def _build_rope(max_position: int):
    """Instantiate the same object vLLM builds for the draft head, under an
    eager-equivalent config context. Returns (rope, resolved_custom_ops,
    vcfg) — callers that consult CustomOp class methods afterwards (e.g.
    .enabled(), which asserts a current vLLM config) must re-enter
    set_current_vllm_config(vcfg) around those calls; the 2026-07-17 GPU
    run showed the build-time context does not stay active for them."""
    import torch
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    try:
        from vllm.config.compilation import CompilationMode
        mode_none = CompilationMode.NONE
    except Exception:
        mode_none = 0
    vcfg = VllmConfig(compilation_config=CompilationConfig(mode=mode_none))
    with set_current_vllm_config(vcfg):
        from vllm.model_executor.layers.rotary_embedding import get_rope
        rope = get_rope(
            head_size=HEAD_SIZE,
            max_position=max_position,
            is_neox_style=True,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
            dtype=torch.float32,
        ).cuda()
    return rope, list(vcfg.compilation_config.custom_ops), vcfg


def probe_dispatch() -> dict:
    from vllm.config import set_current_vllm_config
    rope, custom_ops, vcfg = _build_rope(CACHE_LIMIT)
    method = getattr(rope, "_forward_method", None)
    name = getattr(method, "__name__", str(method))
    with set_current_vllm_config(vcfg):  # .enabled() asserts a current config
        enabled = type(rope).enabled()
    return {
        "rope_class": type(rope).__name__,
        "resolved_custom_ops": custom_ops,
        "enabled": enabled,
        "dispatched_to": name,
        "cache_rows": int(rope.cos_sin_cache.shape[0]),
    }


def _run_positions(rope, positions, fn_name: str) -> dict:
    """Rotate a fixed query at each position; compare to independent math."""
    import torch
    out = {}
    fn = getattr(rope, fn_name)
    for pos in positions:
        torch.manual_seed(0)
        q = torch.randn(NUM_TOKENS, HEAD_SIZE, dtype=torch.float32)
        ref = _reference_rotate(q[0], pos, HEAD_SIZE)
        try:
            got, _ = fn(torch.tensor([pos], device="cuda"),
                        q.clone().cuda(),
                        q.clone().cuda())
            torch.cuda.synchronize()
            err = float((got[0].double().cpu() - ref).abs().max())
            out[pos] = {"max_abs_err": err,
                        "verdict": "CORRECT" if err < 1e-2 else "GARBAGE"}
        except Exception as exc:  # device assert / index error
            out[pos] = {"verdict": "RAISED", "error": str(exc)[:200]}
    return out


def probe_cuda() -> dict:
    rope, _, _ = _build_rope(CACHE_LIMIT)
    return {"forward_cuda": _run_positions(rope, IN_BOUNDS + OOB,
                                           "forward_cuda")}


def probe_native() -> dict:
    rope, _, _ = _build_rope(CACHE_LIMIT)
    # one in-bounds control, one OOB: is the native gather checked on CUDA?
    return {"forward_native": _run_positions(rope, [2047, 2048],
                                             "forward_native")}


def probe_control() -> dict:
    """Sanity: a correctly-sized cache must be CORRECT at every position."""
    rope, _, _ = _build_rope(FULL_LIMIT)
    return {"forward_cuda_8192cache": _run_positions(
        rope, IN_BOUNDS + OOB, "forward_cuda")}


PROBES = {"dispatch": probe_dispatch, "cuda": probe_cuda,
          "native": probe_native, "control": probe_control}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=sorted(PROBES))
    args = parser.parse_args()

    if args.probe:  # child mode: run one probe, print JSON
        print(json.dumps(PROBES[args.probe](), indent=1))
        return 0

    # orchestrator: each probe in its own process (device asserts poison
    # the CUDA context of the process they fire in)
    results = {}
    for name in ("dispatch", "control", "cuda", "native"):
        proc = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()),
             "--probe", name],
            capture_output=True, text=True, timeout=600)
        print("=" * 20, "probe:", name, "(exit %d)" % proc.returncode)
        if proc.returncode == 0:
            try:
                results[name] = json.loads(proc.stdout)
                print(json.dumps(results[name], indent=1))
            except json.JSONDecodeError:
                print(proc.stdout[-2000:])
        else:
            results[name] = {"crashed": True}
            print((proc.stderr or proc.stdout)[-1500:])

    print()
    print("#" * 8, "VERDICT", "#" * 8)
    d = results.get("dispatch", {})
    if d.get("dispatched_to"):
        print("Under eager-equivalent config, custom_ops resolves to %s and "
              "RotaryEmbedding dispatches to: %s"
              % (d.get("resolved_custom_ops"), d.get("dispatched_to")))
    else:
        print("dispatch probe unavailable -- rely on the cuda/native "
              "probes: native RAISES at the boundary while real eager "
              "runs never crashed, which excludes forward_native and "
              "proves the forward_cuda dispatch by elimination.")
    cuda = results.get("cuda", {}).get("forward_cuda", {})
    if cuda:
        ib = [str(p) for p in IN_BOUNDS if cuda.get(p, cuda.get(str(p), {})).get("verdict") == "CORRECT"]
        print("forward_cuda in-bounds control: %d/%d CORRECT" % (len(ib), len(IN_BOUNDS)))
        for p in OOB:
            r = cuda.get(p, cuda.get(str(p), {}))
            print("  pos %5d -> %s (max_abs_err=%s)"
                  % (p, r.get("verdict"), r.get("max_abs_err")))
    nat = results.get("native", {}).get("forward_native", {})
    if nat or results.get("native", {}).get("crashed"):
        print("forward_native at pos 2048:",
              json.dumps(nat.get(2048, nat.get("2048", results.get("native"))))[:300])
    print("Paste this whole output into the session -- it goes verbatim "
          "into analysis/vllm_2048_bug_diagnosis.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
