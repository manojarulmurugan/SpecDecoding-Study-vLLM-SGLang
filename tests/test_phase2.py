"""Phase-2 additions: emergent-batch sampling, goodput, config set,
marginals report."""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from analysis.marginals import collect as marginals_collect, main as marginals_main, render_report
from harness.config import ConfigError, RunConfig, load_configs
from harness.engines.base import ServerHandle
from harness.engines.vllm_adapter import VllmAdapter
from harness.metrics import aggregate_run, summarize_batch_samples
from harness.results import ResultsStore
from harness.run import execute_run
from harness.sampling import MetricsSampler
from harness.sweep import group_by_server
from tests.conftest import make_config


# -- emergent batch-size sampling ----------------------------------------------

def test_sampler_observes_in_flight_requests(fake_server):
    fake_server.chunk_delay_s = 0.03
    from harness.load import run_closed_loop

    sampler = MetricsSampler(fake_server.base_url, interval_s=0.01).start()
    run_closed_loop(
        fake_server.base_url, "m", ["p%d" % i for i in range(12)],
        concurrency=4, max_tokens=8, progress_every=0,
    )
    samples = sampler.stop()
    running = samples["running"]
    assert len(running) >= 3
    assert max(running) >= 2, "sampler never saw concurrent in-flight requests"
    assert max(running) <= 4
    assert fake_server.max_in_flight_seen >= 2
    # capacity gauges sampled alongside (fake: kv usage = 0.1 * in-flight)
    assert samples["kv_cache_usage"]
    assert max(samples["kv_cache_usage"]) <= 0.4 + 1e-9
    assert samples["waiting"] == [0] * len(samples["waiting"])


def test_sampler_survives_dead_endpoint():
    sampler = MetricsSampler("http://127.0.0.1:9", interval_s=0.01,
                             timeout_s=0.2).start()
    time.sleep(0.05)
    assert all(v == [] for v in sampler.stop().values())


def test_summarize_batch_samples():
    s = summarize_batch_samples([1, 2, 3, 10])
    assert s["mean"] == 4.0
    assert s["max"] == 10
    assert s["num_samples"] == 4
    assert summarize_batch_samples([]) is None


def test_execute_run_records_emergent_batch(fake_server, gsm8k_questions_file, tmp_path):
    fake_server.chunk_delay_s = 0.03
    cfg = RunConfig.from_dict(make_config(
        workload_params={"questions_file": gsm8k_questions_file},
        concurrency=3,
        batch_sample_interval_s=0.01,
    ))
    store = ResultsStore(tmp_path)
    record = execute_run(
        cfg, store, VllmAdapter(cfg),
        ServerHandle(process=None, base_url=fake_server.base_url, external=True),
        log=lambda *_: None,
    )
    batch = record["measured"]["emergent_batch_size"]
    assert batch is not None and batch["num_samples"] >= 1
    assert 0 <= batch["max"] <= 3


def test_batch_sample_interval_validated():
    with pytest.raises(ConfigError, match="batch_sample_interval_s"):
        RunConfig.from_dict(make_config(batch_sample_interval_s=0))


# -- goodput -------------------------------------------------------------------

def _req(tokens=10, e2e=2.0):
    return SimpleNamespace(completion_tokens=tokens, ttft_s=0.5, decode_time_s=1.5,
                           e2e_s=e2e, itl_s=[], error=None)


def test_goodput_equals_kept_tokens_per_second():
    measured = aggregate_run([_req(30), _req(50)], wall_time_s=4.0)
    assert measured["goodput_tok_s"] == pytest.approx(20.0)
    assert measured["throughput_tok_s"] == measured["goodput_tok_s"]
    assert measured["spec_rejected_tok_s"] is None


def test_rejected_draft_rate_from_counters():
    spec_stats = {"num_drafts": 100, "num_draft_tokens": 500,
                  "num_accepted_tokens": 300, "accepted_length_tau": 4.0,
                  "acceptance_rate": 0.6}
    measured = aggregate_run([_req(400)], wall_time_s=10.0, spec_stats=spec_stats)
    # 500 drafted - 300 accepted = 200 rejected over 10s
    assert measured["spec_rejected_tok_s"] == pytest.approx(20.0)
    assert measured["goodput_tok_s"] == pytest.approx(40.0)


# -- dataset cycling (num_requests > dataset) ------------------------------------

def test_subsample_tiles_when_oversampled(gsm8k_questions_file):
    from harness.workloads.gsm8k import Gsm8kWorkload

    wl = Gsm8kWorkload(
        {"questions_file": gsm8k_questions_file, "num_requests": 8}, seed=1
    )
    items = wl.build()  # fixture has 3 questions
    assert len(items) == 8
    assert items[0].prompt == items[3].prompt == items[6].prompt


# -- the Phase-2 config set ------------------------------------------------------

def test_full_cube_config_board_shape():
    configs = load_configs(["configs/factorial/cube_*.yaml"])
    # 8 corners x 3 workloads x 4 concurrencies x 3 repeats
    assert len(configs) == 288
    assert len({c.run_id for c in configs}) == 288
    # 8 server groups: repeats and workloads share a launch per corner
    assert len(group_by_server(configs)) == 8
    corners = set()
    for cfg in configs:
        assert cfg.block == "core_factorial"
        assert cfg.decoding == "greedy"
        assert cfg.gpu_target == "a100", "GPU-confound correction: A100 only"
        assert cfg.repeat_idx in (0, 1, 2)
        assert cfg.seed == 1234, "repeats must share the server seed"
        n = cfg.workload_params["num_requests"]
        expected = {1: 64, 8: 160, 32: 320, 64: 512}[cfg.concurrency]
        assert n == expected, "num_requests must scale with concurrency"
        f = cfg.factors
        corners.add((f.weight_quant, f.kv_quant, f.spec_decode))
        if f.spec_decode == "eagle3":
            assert cfg.draft_model == "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
        if f.weight_quant == "w4a16":
            assert "AWQ" in cfg.model
    assert len(corners) == 8, "all 8 corners of the 2^3 cube must be present"


def test_marginal_runbook_globs_select_exactly_the_single_factor_corners():
    globs = [
        "configs/factorial/cube_base_*_r0.yaml",
        "configs/factorial/cube_w_*_r0.yaml",
        "configs/factorial/cube_k_*_r0.yaml",
        "configs/factorial/cube_s_*_r0.yaml",
    ]
    configs = load_configs(globs)
    assert len(configs) == 48
    assert len(group_by_server(configs)) == 4
    for cfg in configs:
        f = cfg.factors
        on = [f.weight_quant != "fp16", f.kv_quant != "fp16", f.spec_decode != "none"]
        assert sum(on) <= 1, "marginal globs must not catch interaction corners"
        assert cfg.repeat_idx == 0


def test_phase2_commands_build():
    import json as _json

    configs = load_configs(["configs/factorial/cube_*.yaml"])
    for cfg in configs:
        cmd = VllmAdapter(cfg).build_launch_command()
        if cfg.factors.kv_quant == "fp8":
            assert cmd[cmd.index("--kv-cache-dtype") + 1] == "fp8"
        if cfg.factors.weight_quant == "w4a16":
            assert cmd[cmd.index("--quantization") + 1] == "awq_marlin"
        if cfg.factors.spec_decode == "eagle3":
            payload = _json.loads(cmd[cmd.index("--speculative-config") + 1])
            assert payload["method"] == "eagle3"


# -- marginals report -------------------------------------------------------------

def _p2_record(workload, conc, w="fp16", k="fp16", s="none", goodput=100.0,
               tau=None, batch=None, status="ok", block="core_factorial"):
    return {
        "run_id": "p2_%s_%s_%s_%s_c%d" % (w, k, s, workload, conc),
        "config": {
            "block": block, "workload": workload, "concurrency": conc,
            "repeat_idx": 0,
            "factors": {"weight_quant": w, "kv_quant": k, "spec_decode": s},
        },
        "env": {}, "status": status,
        "measured": {
            "goodput_tok_s": goodput, "throughput_tok_s": goodput,
            "accepted_length_tau": tau,
            "emergent_batch_size": batch,
        },
    }


def test_marginals_collect_and_render():
    records = [
        _p2_record("gsm8k", 1, goodput=90, batch={"mean": 1.0, "p50": 1, "max": 1}),
        _p2_record("gsm8k", 1, w="w4a16", goodput=160),
        _p2_record("gsm8k", 1, k="fp8", goodput=95),
        _p2_record("gsm8k", 1, s="eagle3", goodput=150, tau=2.6),
        _p2_record("gsm8k", 32, goodput=1500, batch={"mean": 20.5, "p50": 21, "max": 30}),
        _p2_record("gsm8k", 32, s="eagle3", goodput=1400, tau=2.4),
        _p2_record("gsm8k", 8, status="failed"),          # excluded
        _p2_record("gsm8k", 8, w="w4a16", k="fp8"),       # 2-factor: Phase 3, excluded
    ]
    cells = marginals_collect(records)
    assert ("gsm8k", 8, "baseline") not in cells
    assert len([k for k in cells if k[1] == 8]) == 0
    report = render_report(cells)
    assert "1.78x" in report          # 160/90 W speedup at conc=1
    assert "0.93x" in report          # 1400/1500 S at conc=32: erosion visible
    assert "20.5 / 30" in report      # emergent batch mean/max
    assert "2.60" in report           # tau at conc=1


def test_marginals_cli_no_data(tmp_path):
    assert marginals_main([str(tmp_path / "empty")]) == 1


def test_phase3_session_globs():
    """The Phase-3 runbook's three sweep selections must partition cleanly."""
    session_a = load_configs([
        "configs/factorial/cube_ws_*_r0.yaml",
        "configs/factorial/cube_ks_*_r0.yaml",
        "configs/factorial/cube_wks_*_r0.yaml",
        "configs/factorial/cube_wk_*_r0.yaml",
    ])
    assert len(session_a) == 48
    assert len(group_by_server(session_a)) == 4
    for cfg in session_a:
        f = cfg.factors
        on = [f.weight_quant != "fp16", f.kv_quant != "fp16", f.spec_decode != "none"]
        assert sum(on) >= 2, "session A is interaction corners only"
        assert cfg.repeat_idx == 0
    # session-A order follows glob order: ws corners come first
    first = session_a[0].factors
    assert (first.weight_quant, first.kv_quant, first.spec_decode) == (
        "w4a16", "fp16", "eagle3")

    session_b = load_configs(["configs/factorial/cube_*_r1.yaml"])
    session_c = load_configs(["configs/factorial/cube_*_r2.yaml"])
    for session in (session_b, session_c):
        assert len(session) == 96
        assert len(group_by_server(session)) == 8
    # A + B + C + the Phase-2 marginal set = the full 288-cell board
    marginals = load_configs([
        "configs/factorial/cube_base_*_r0.yaml", "configs/factorial/cube_w_*_r0.yaml",
        "configs/factorial/cube_k_*_r0.yaml", "configs/factorial/cube_s_*_r0.yaml",
    ])
    all_ids = {c.run_id for c in session_a + session_b + session_c + marginals}
    board = load_configs(["configs/factorial/cube_*.yaml"])
    assert all_ids == {c.run_id for c in board}
