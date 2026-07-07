from __future__ import annotations

import json

import pytest

from harness.config import RunConfig
from harness.engines.vllm_adapter import VllmAdapter
from tests.conftest import make_config


def _cmd(**overrides):
    cfg = RunConfig.from_dict(make_config(**overrides))
    return VllmAdapter(cfg).build_launch_command()


def _spec_payload(cmd):
    assert "--speculative-config" in cmd
    return json.loads(cmd[cmd.index("--speculative-config") + 1])


def test_baseline_command():
    cmd = _cmd()
    assert cmd[:3] == ["vllm", "serve", "meta-llama/Meta-Llama-3-8B-Instruct"]
    assert cmd[cmd.index("--port") + 1] == "8000"
    assert cmd[cmd.index("--dtype") + 1] == "float16"
    assert cmd[cmd.index("--seed") + 1] == "1234"
    assert "--no-enable-prefix-caching" in cmd
    assert "--speculative-config" not in cmd
    assert "--kv-cache-dtype" not in cmd
    assert "--quantization" not in cmd  # auto-detect from checkpoint


def test_eagle_command_uses_verified_method_string():
    cmd = _cmd(
        factors={"spec_decode": "eagle"},
        draft_model="yuhuili/EAGLE-LLaMA3-Instruct-8B",
    )
    payload = _spec_payload(cmd)
    # "eagle", not "eagle2": vLLM has no eagle2 method string (PREREQ Check 3)
    assert payload["method"] == "eagle"
    assert payload["model"] == "yuhuili/EAGLE-LLaMA3-Instruct-8B"
    assert payload["num_speculative_tokens"] == 5
    assert payload["draft_tensor_parallel_size"] == 1


def test_eagle3_method_string():
    cmd = _cmd(
        factors={"spec_decode": "eagle3"},
        draft_model="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
        model="meta-llama/Llama-3.1-8B-Instruct",
    )
    assert _spec_payload(cmd)["method"] == "eagle3"


def test_w4a16_carries_no_quant_flag_but_gptq_model():
    cmd = _cmd(model="YudiZh/Meta-Llama-3-8B-Instruct-W4A16-g128",
               factors={"weight_quant": "w4a16"})
    assert cmd[2] == "YudiZh/Meta-Llama-3-8B-Instruct-W4A16-g128"
    assert "--quantization" not in cmd


def test_quantization_override():
    cmd = _cmd(engine_args={"quantization": "gptq_marlin"})
    assert cmd[cmd.index("--quantization") + 1] == "gptq_marlin"


def test_plain_awq_rejected():
    # Check 6: plain "awq" is the unoptimized path; the tested config is
    # awq_marlin. Make the mistake impossible to ship.
    with pytest.raises(ValueError, match="awq_marlin"):
        _cmd(engine_args={"quantization": "awq"})


def test_awq_checkpoint_defaults_to_awq_marlin():
    cmd = _cmd(model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
               factors={"weight_quant": "w4a16"})
    assert cmd[cmd.index("--quantization") + 1] == "awq_marlin"


def test_fp8_kv_cache_flag():
    cmd = _cmd(factors={"kv_quant": "fp8"})
    assert cmd[cmd.index("--kv-cache-dtype") + 1] == "fp8"


def test_prefix_caching_toggle():
    cmd = _cmd(engine_args={"enable_prefix_caching": True})
    assert "--enable-prefix-caching" in cmd
    assert "--no-enable-prefix-caching" not in cmd


def test_extra_args_appended_verbatim():
    cmd = _cmd(engine_args={"extra": ["--swap-space", "4"]})
    assert cmd[-2:] == ["--swap-space", "4"]


def test_all_shipped_repro_configs_build_commands():
    from harness.config import load_configs

    for cfg in load_configs(["configs/repro/repro_*.yaml"]):
        cmd = VllmAdapter(cfg).build_launch_command()
        assert cmd[0] == "vllm"
        if cfg.factors.spec_decode == "eagle":
            assert _spec_payload(cmd)["method"] == "eagle"
