"""End-to-end pipeline tests against the fake server: the closest thing to
a real run this GPU-less machine can produce."""
from __future__ import annotations

import pytest
import yaml

from harness.config import RunConfig
from harness.engines.base import ServerHandle
from harness.engines.vllm_adapter import VllmAdapter
from harness.results import ResultsStore
from harness.run import execute_run, main as run_main
from harness.sweep import group_by_server, run_sweep
from tests.conftest import make_config


def _config(gsm8k_questions_file, **overrides):
    base = make_config(
        workload_params={"questions_file": gsm8k_questions_file},
        warmup_requests=1,
    )
    base.update(overrides)
    return RunConfig.from_dict(base)


def _external(fake_server):
    return ServerHandle(process=None, base_url=fake_server.base_url, external=True)


def test_execute_run_full_record(fake_server, gsm8k_questions_file, tmp_path):
    cfg = _config(
        gsm8k_questions_file,
        factors={"weight_quant": "fp16", "kv_quant": "fp16", "spec_decode": "eagle"},
        draft_model="yuhuili/EAGLE-LLaMA3-Instruct-8B",
    )
    store = ResultsStore(tmp_path)
    record = execute_run(
        cfg, store, VllmAdapter(cfg), _external(fake_server), log=lambda *_: None
    )
    assert record["status"] == "ok"
    measured = record["measured"]
    assert measured["num_requests"] == 3
    assert measured["num_errors"] == 0
    # fake server: 30 accepted / 10 drafts per request -> tau = 1 + 3 = 4,
    # and the warmup request must NOT pollute the delta
    assert measured["accepted_length_tau"] == pytest.approx(4.0)
    assert measured["acceptance_rate"] == pytest.approx(0.6)
    # canned completion answers "6"; all fixture answers are 6
    assert measured["accuracy"] == 1.0
    assert measured["throughput_tok_s"] > 0
    assert record["config"]["run_id"] == cfg.run_id
    assert record["env"]["engine_version"] == "0.0-fake"
    assert store.is_complete(cfg.run_id)


def test_execute_run_no_spec_has_null_tau(fake_server, gsm8k_questions_file, tmp_path):
    cfg = _config(gsm8k_questions_file)
    store = ResultsStore(tmp_path)
    record = execute_run(
        cfg, store, VllmAdapter(cfg), _external(fake_server), log=lambda *_: None
    )
    assert record["measured"]["accepted_length_tau"] is None


def test_run_main_with_external_server(fake_server, gsm8k_questions_file, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(make_config(
        workload_params={"questions_file": gsm8k_questions_file},
    )))
    rc = run_main([
        str(cfg_path), "--results-dir", str(tmp_path / "results"),
        "--server-url", fake_server.base_url,
    ])
    assert rc == 0
    # second invocation resumes (skips) cleanly
    rc = run_main([
        str(cfg_path), "--results-dir", str(tmp_path / "results"),
        "--server-url", fake_server.base_url,
    ])
    assert rc == 0


def test_group_by_server_orders_and_groups(gsm8k_questions_file):
    a = _config(gsm8k_questions_file, workload="gsm8k")
    b = _config(gsm8k_questions_file, workload="humaneval")
    c = _config(gsm8k_questions_file, model="YudiZh/Meta-Llama-3-8B-Instruct-W4A16-g128",
                factors={"weight_quant": "w4a16"})
    groups = group_by_server([a, b, c])
    assert len(groups) == 2
    assert [len(g) for g in groups.values()] == [2, 1]


def test_sweep_runs_skips_and_resumes(fake_server, gsm8k_questions_file, tmp_path):
    cfg_a = _config(gsm8k_questions_file, repeat_idx=0)
    cfg_b = _config(gsm8k_questions_file, repeat_idx=1)
    store = ResultsStore(tmp_path)
    log_lines = []

    outcome = run_sweep(
        [cfg_a, cfg_b], store, server_url=fake_server.base_url,
        log=log_lines.append,
    )
    assert sorted(outcome["ok"]) == sorted([cfg_a.run_id, cfg_b.run_id])
    assert outcome["skipped"] == []

    outcome = run_sweep(
        [cfg_a, cfg_b], store, server_url=fake_server.base_url,
        log=log_lines.append,
    )
    assert sorted(outcome["skipped"]) == sorted([cfg_a.run_id, cfg_b.run_id])
    assert outcome["ok"] == []


def test_sweep_dry_run_launches_nothing(gsm8k_questions_file, tmp_path):
    cfg = _config(gsm8k_questions_file)
    store = ResultsStore(tmp_path)
    lines = []
    outcome = run_sweep([cfg], store, dry_run=True, log=lines.append)
    assert outcome == {"ok": [], "skipped": [], "failed": []}
    assert any("vllm serve" in l for l in lines)
    assert not store.is_complete(cfg.run_id)


def test_sweep_external_url_rejects_multiple_groups(fake_server, gsm8k_questions_file, tmp_path):
    a = _config(gsm8k_questions_file)
    b = _config(gsm8k_questions_file, factors={"kv_quant": "fp8"})
    with pytest.raises(ValueError, match="distinct server"):
        run_sweep([a, b], ResultsStore(tmp_path), server_url=fake_server.base_url,
                  log=lambda *_: None)
