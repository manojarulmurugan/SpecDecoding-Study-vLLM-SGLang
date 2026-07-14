"""Run-cell configuration: schema, validation, deterministic run IDs.

One YAML file describes one run cell (HARNESS_SPEC.md §3). Everything the
harness does is derived from the config, and the config is copied verbatim
into the result record so every number is traceable.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

VALID_BLOCKS = {
    "repro", "serving_baseline", "core_factorial", "k_stress",
    "sglang_seam", "optional",
}
VALID_ENGINES = {"vllm", "sglang"}
VALID_WEIGHT_QUANT = {"fp16", "w4a16", "w8a8", "w4a8"}
VALID_KV_QUANT = {"fp16", "fp8"}  # int8 KV not shipped in vLLM (PREREQ_RESULTS Check 2)
# vLLM speculative_config method strings, verified against docs.vllm.ai.
# EAGLE-1-style checkpoints (e.g. yuhuili/EAGLE-LLaMA3-Instruct-8B) use
# "eagle"; there is no "eagle2" method string -- EAGLE-2 is a drafting-tree
# algorithm on the same checkpoint format, not a different config value.
# See PREREQ_RESULTS.md Check 3 (resolution appended).
VALID_SPEC_DECODE = {"none", "eagle", "eagle3", "ngram"}
VALID_WORKLOADS = {"gsm8k", "humaneval", "rag_shared_prefix", "mt_bench"}
VALID_DECODING = {"greedy", "sampling"}
# a100_40gb / a100_80gb: Colab's High-RAM toggle pins the A100 variant
# (confirmed empirically 2026-07-09; PREREQ_RESULTS Check 1). Cube cells must
# match Phase 2's 80GB card; k_stress targets 40GB where both KV ceilings
# fit in a small concurrency grid.
VALID_GPU_TARGETS = {"a100", "a100_40gb", "a100_80gb", "h100", "l4", "t4", "any"}


class ConfigError(ValueError):
    """Raised when a run config fails validation."""


@dataclass
class Factors:
    weight_quant: str = "fp16"
    kv_quant: str = "fp16"
    spec_decode: str = "none"


@dataclass
class EngineArgs:
    port: int = 8000
    host: str = "127.0.0.1"
    dtype: str = "float16"
    gpu_memory_utilization: float = 0.8
    max_model_len: int = 4096
    # Off by default for controlled cells: GSM8K's 8-shot prefix is shared
    # across every request, so automatic prefix caching would contaminate
    # prefill timing relative to SpecMQuant's per-request full prefill.
    enable_prefix_caching: bool = False
    # None lets vLLM auto-detect quantization from the checkpoint config
    # (correct for the YudiZh GPTQ checkpoints). Set explicitly to override.
    quantization: Optional[str] = None
    num_speculative_tokens: int = 5
    # Extra environment variables for the server process (e.g.
    # VLLM_ATTENTION_BACKEND to pin the attention backend). Recorded
    # verbatim; part of the server signature like every launch input.
    env: Dict[str, str] = field(default_factory=dict)
    # Escape hatch for flags the schema doesn't model; recorded verbatim.
    extra: List[str] = field(default_factory=list)


@dataclass
class RunConfig:
    block: str
    engine: str
    model: str
    factors: Factors
    workload: str
    draft_model: Optional[str] = None
    workload_params: Dict[str, Any] = field(default_factory=dict)
    concurrency: int = 1
    decoding: str = "greedy"
    seed: int = 1234
    repeat_idx: int = 0
    warmup_requests: int = 3
    gpu_target: str = "a100"
    # Poll interval for the emergent-batch-size sampler (PROJECT_SPEC §7.2).
    batch_sample_interval_s: float = 1.0
    engine_args: EngineArgs = field(default_factory=EngineArgs)
    run_id: Optional[str] = None

    def __post_init__(self) -> None:
        self.validate()
        if self.run_id is None:
            self.run_id = self.derive_run_id()

    # -- construction ------------------------------------------------------

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RunConfig":
        raw = dict(raw)
        factors = raw.pop("factors", {})
        engine_args = raw.pop("engine_args", {})
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ConfigError("unknown config fields: %s" % sorted(unknown))
        try:
            factors = Factors(**factors)
            engine_args = EngineArgs(**engine_args)
        except TypeError as exc:
            raise ConfigError(str(exc)) from exc
        return cls(factors=factors, engine_args=engine_args, **raw)

    @classmethod
    def from_yaml(cls, path: "Path | str") -> "RunConfig":
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ConfigError("%s: expected a mapping at top level" % path)
        return cls.from_dict(raw)

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        checks = [
            (self.block, VALID_BLOCKS, "block"),
            (self.engine, VALID_ENGINES, "engine"),
            (self.factors.weight_quant, VALID_WEIGHT_QUANT, "factors.weight_quant"),
            (self.factors.kv_quant, VALID_KV_QUANT, "factors.kv_quant"),
            (self.factors.spec_decode, VALID_SPEC_DECODE, "factors.spec_decode"),
            (self.workload, VALID_WORKLOADS, "workload"),
            (self.decoding, VALID_DECODING, "decoding"),
            (self.gpu_target, VALID_GPU_TARGETS, "gpu_target"),
        ]
        for value, allowed, name in checks:
            if value not in allowed:
                raise ConfigError(
                    "%s=%r not in %s" % (name, value, sorted(allowed))
                )
        spec = self.factors.spec_decode
        if spec in ("eagle", "eagle3") and not self.draft_model:
            raise ConfigError("spec_decode=%s requires draft_model" % spec)
        if spec == "none" and self.draft_model:
            raise ConfigError("draft_model set but spec_decode=none")
        if self.concurrency < 1:
            raise ConfigError("concurrency must be >= 1")
        if self.batch_sample_interval_s <= 0:
            raise ConfigError("batch_sample_interval_s must be > 0")
        if not self.model:
            raise ConfigError("model is required")

    # -- identity ----------------------------------------------------------

    def derive_run_id(self) -> str:
        model_tag = _slug(self.model.split("/")[-1])
        f = self.factors
        return "{block}_{model}_{w}-{k}-{s}_{wl}_c{c}_r{r}".format(
            block=self.block,
            model=model_tag,
            w=f.weight_quant,
            k=f.kv_quant,
            s=f.spec_decode,
            wl=self.workload,
            c=self.concurrency,
            r=self.repeat_idx,
        )

    def server_signature(self) -> str:
        """Stable key for grouping runs that can share one server process.

        Everything that affects the *server launch command* goes in here;
        workload/concurrency/repeat do not.
        """
        payload = {
            "engine": self.engine,
            "model": self.model,
            "factors": asdict(self.factors),
            "draft_model": self.draft_model,
            "seed": self.seed,
            "engine_args": asdict(self.engine_args),
        }
        return json.dumps(payload, sort_keys=True)

    def temperature(self) -> float:
        if self.decoding == "greedy":
            return 0.0
        return float(self.workload_params.get("temperature", 0.7))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def load_configs(paths: List[str]) -> List[RunConfig]:
    """Load many config files, expanding globs, preserving order, deduping."""
    import glob as _glob

    files: List[Path] = []
    for p in paths:
        if any(ch in p for ch in "*?["):
            matches = [Path(x) for x in sorted(_glob.glob(p))]
        else:
            matches = [Path(p)]
        if not matches:
            raise ConfigError("no config files match %r" % p)
        files.extend(matches)
    seen = set()
    configs = []
    for f in files:
        if f in seen:
            continue
        seen.add(f)
        configs.append(RunConfig.from_yaml(f))
    ids = [c.run_id for c in configs]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ConfigError("duplicate run_ids across configs: %s" % sorted(dupes))
    return configs
