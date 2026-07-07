"""vLLM server adapter.

Builds a ``vllm serve`` command from the config's factors. Pinned engine:
vllm==0.24.0 -- confirmed by direct test to run the full
W4A16 + FP8-KV + EAGLE-3 stack (PREREQ_RESULTS.md Check 6; older 0.10.x
falls back to V0 on fp8 KV, which rejects spec decoding entirely).

- weight quant is carried by the checkpoint itself. AWQ checkpoints get
  ``--quantization awq_marlin`` (plain "awq" is the unoptimized path,
  Check 6); GPTQ checkpoints (the Block-0 YudiZh repro model) rely on
  vLLM's auto-detection, which selects the marlin kernels itself.
- KV-cache quant: ``--kv-cache-dtype fp8``.
- speculative decoding: ``--speculative-config`` with a JSON payload;
  method "eagle" for EAGLE-1-style checkpoints (yuhuili/EAGLE-LLaMA3-*),
  "eagle3" for EAGLE-3 heads. Checkpoints load directly from HF on
  vllm>=0.7.0; no conversion step. (Check 3 resolution.)
"""
from __future__ import annotations

import json
from typing import List

from ..config import RunConfig
from .base import EngineAdapter


class VllmAdapter(EngineAdapter):
    def build_launch_command(self) -> List[str]:
        cfg = self.config
        ea = cfg.engine_args
        cmd: List[str] = [
            "vllm", "serve", cfg.model,
            "--host", ea.host,
            "--port", str(ea.port),
            "--dtype", ea.dtype,
            "--gpu-memory-utilization", str(ea.gpu_memory_utilization),
            "--max-model-len", str(ea.max_model_len),
            "--seed", str(cfg.seed),
        ]

        cmd.append(
            "--enable-prefix-caching" if ea.enable_prefix_caching
            else "--no-enable-prefix-caching"
        )

        if ea.quantization == "awq":
            raise ValueError(
                'quantization="awq" is the unoptimized kernel path; use '
                '"awq_marlin" (PREREQ_RESULTS.md Check 6)'
            )
        if ea.quantization:
            cmd += ["--quantization", ea.quantization]
        elif cfg.factors.weight_quant == "w4a16" and "awq" in cfg.model.lower():
            cmd += ["--quantization", "awq_marlin"]

        kv = cfg.factors.kv_quant
        if kv == "fp8":
            cmd += ["--kv-cache-dtype", "fp8"]
        elif kv != "fp16":
            raise ValueError("kv_quant=%r not supported by the vLLM adapter" % kv)

        spec = cfg.factors.spec_decode
        if spec in ("eagle", "eagle3"):
            spec_payload = {
                "method": spec,
                "model": cfg.draft_model,
                "num_speculative_tokens": ea.num_speculative_tokens,
                "draft_tensor_parallel_size": 1,
            }
            cmd += ["--speculative-config", json.dumps(spec_payload, sort_keys=True)]
        elif spec == "ngram":
            spec_payload = {
                "method": "ngram",
                "num_speculative_tokens": ea.num_speculative_tokens,
                "prompt_lookup_max": 4,
            }
            cmd += ["--speculative-config", json.dumps(spec_payload, sort_keys=True)]

        cmd += list(ea.extra)
        return cmd
