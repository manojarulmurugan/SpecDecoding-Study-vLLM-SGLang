from __future__ import annotations

import pytest

from harness.config import ConfigError, RunConfig, load_configs
from tests.conftest import make_config


def test_round_trip_and_run_id_determinism(tmp_path):
    cfg1 = RunConfig.from_dict(make_config())
    cfg2 = RunConfig.from_dict(make_config())
    assert cfg1.run_id == cfg2.run_id
    assert cfg1.run_id == "repro_meta-llama-3-8b-instruct_fp16-fp16-none_gsm8k_c1_r0"


def test_run_id_varies_with_identity_fields():
    base = RunConfig.from_dict(make_config())
    changed = RunConfig.from_dict(make_config(repeat_idx=1))
    assert base.run_id != changed.run_id
    changed = RunConfig.from_dict(make_config(concurrency=8))
    assert base.run_id != changed.run_id
    changed = RunConfig.from_dict(
        make_config(factors={"spec_decode": "eagle"},
                    draft_model="yuhuili/EAGLE-LLaMA3-Instruct-8B")
    )
    assert base.run_id != changed.run_id


def test_server_signature_groups_correctly():
    a = RunConfig.from_dict(make_config(workload="gsm8k"))
    b = RunConfig.from_dict(make_config(workload="humaneval"))
    c = RunConfig.from_dict(make_config(factors={"kv_quant": "fp8"}))
    assert a.server_signature() == b.server_signature()  # same server, diff workload
    assert a.server_signature() != c.server_signature()  # kv dtype changes launch


def test_eagle_requires_draft_model():
    with pytest.raises(ConfigError, match="requires draft_model"):
        RunConfig.from_dict(make_config(factors={"spec_decode": "eagle"}))


def test_draft_model_without_spec_rejected():
    with pytest.raises(ConfigError, match="spec_decode=none"):
        RunConfig.from_dict(make_config(draft_model="some/draft"))


def test_eagle2_is_not_a_valid_method_string():
    # HARNESS_SPEC originally sketched `spec_decode: eagle2`; vLLM has no
    # such method (PREREQ_RESULTS Check 3 resolution). Make the mistake loud.
    with pytest.raises(ConfigError, match="spec_decode"):
        RunConfig.from_dict(
            make_config(factors={"spec_decode": "eagle2"}, draft_model="x/y")
        )


def test_unknown_field_rejected():
    with pytest.raises(ConfigError, match="unknown config fields"):
        RunConfig.from_dict(make_config(batch_size=32))


def test_greedy_temperature_zero():
    cfg = RunConfig.from_dict(make_config())
    assert cfg.temperature() == 0.0


def test_load_configs_rejects_duplicate_run_ids(tmp_path):
    import yaml

    for name in ("a.yaml", "b.yaml"):
        (tmp_path / name).write_text(yaml.safe_dump(make_config()))
    with pytest.raises(ConfigError, match="duplicate run_ids"):
        load_configs([str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml")])


def test_shipped_repro_configs_are_valid_and_consistent():
    configs = load_configs(["configs/repro/repro_*.yaml"])
    assert len(configs) == 8
    ids = {c.run_id for c in configs}
    assert len(ids) == 8
    for cfg in configs:
        assert cfg.block == "repro"
        assert cfg.concurrency == 1, "Block 0 is single-stream"
        assert cfg.decoding == "greedy"
        assert cfg.factors.kv_quant == "fp16", "Block 0 does not vary KV quant"
        if cfg.factors.weight_quant == "w4a16":
            assert "W4A16" in cfg.model
        else:
            assert cfg.model == "meta-llama/Meta-Llama-3-8B-Instruct"
        if cfg.factors.spec_decode == "eagle":
            assert cfg.draft_model == "yuhuili/EAGLE-LLaMA3-Instruct-8B"
    # exactly two server groups per model (spec on / spec off)
    signatures = {c.server_signature() for c in configs}
    assert len(signatures) == 4
