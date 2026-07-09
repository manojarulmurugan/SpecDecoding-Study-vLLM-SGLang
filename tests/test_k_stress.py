"""K-stress addendum: doc sizing, capacity signals end-to-end, config set,
and the analysis report."""
from __future__ import annotations

import pytest

from analysis.factorial import collect as factorial_collect
from analysis.k_stress import collect as ks_collect, main as ks_main, render_report
from harness.config import RunConfig, load_configs
from harness.engines.base import ServerHandle
from harness.engines.vllm_adapter import VllmAdapter
from harness.results import ResultsStore
from harness.run import execute_run
from harness.sweep import group_by_server
from harness.workloads.rag_shared_prefix import RagSharedPrefixWorkload
from tests.conftest import make_config


# -- long-document sizing --------------------------------------------------------

def test_doc_target_tokens_grows_docs_uniformly():
    wl = RagSharedPrefixWorkload(
        {"synthetic_num_docs": 4, "num_requests": 8, "prefix_overlap": "low",
         "doc_target_tokens": 1300},
        seed=1,
    )
    items = wl.build()
    target_words = int(1300 / 1.3)
    for item in items:
        doc = item.meta["prefix"].split("Document:\n")[1]
        assert len(doc.split()) == target_words, "docs must be uniform length"


def test_doc_sizing_deterministic_and_optional():
    params = {"synthetic_num_docs": 3, "num_requests": 3, "prefix_overlap": "low",
              "doc_target_tokens": 650}
    a = RagSharedPrefixWorkload(dict(params), seed=5).build()
    b = RagSharedPrefixWorkload(dict(params), seed=5).build()
    assert [i.prompt for i in a] == [i.prompt for i in b]
    unsized = RagSharedPrefixWorkload(
        {"synthetic_num_docs": 3, "num_requests": 3, "prefix_overlap": "low"},
        seed=5,
    ).build()
    assert [i.prompt for i in unsized] != [i.prompt for i in a]


def test_sized_docs_keep_byte_identical_prefixes():
    from harness.workloads.rag_shared_prefix import check_shared_prefix_token_ids

    class ByteTokenizer:
        def encode(self, text):
            return list(text.encode("utf-8"))

    wl = RagSharedPrefixWorkload(
        {"synthetic_num_docs": 2, "num_requests": 12, "prefix_overlap": "mid",
         "doc_target_tokens": 1300},
        seed=1,
    )
    assert check_shared_prefix_token_ids(wl.build(), ByteTokenizer()) >= 1


# -- capacity signals end-to-end ---------------------------------------------------

def test_execute_run_records_capacity_signals(fake_server, gsm8k_questions_file, tmp_path):
    fake_server.chunk_delay_s = 0.03
    fake_server.preemptions_per_request = 2
    fake_server.waiting_reported = 3
    cfg = RunConfig.from_dict(make_config(
        workload_params={"questions_file": gsm8k_questions_file,
                         "request_timeout_s": 60},
        concurrency=3,
        batch_sample_interval_s=0.01,
        warmup_requests=1,
    ))
    store = ResultsStore(tmp_path)
    record = execute_run(
        cfg, store, VllmAdapter(cfg),
        ServerHandle(process=None, base_url=fake_server.base_url, external=True),
        log=lambda *_: None,
    )
    m = record["measured"]
    assert m["queue_depth"]["max"] == 3
    assert m["kv_cache_usage"]["max"] <= 0.3 + 1e-9
    # 3 timed requests x 2 preemptions each; the warmup delta is excluded
    assert m["num_preemptions"] == 6
    assert m["prompt_tokens_mean"] == pytest.approx(11.0)  # fake usage value


def test_capacity_signals_null_safe_without_gauges(fake_server, gsm8k_questions_file, tmp_path):
    fake_server.spec_metrics = False
    # strip all vllm gauges: simulate an engine that exposes nothing
    fake_server._metrics_text = lambda: "# nothing here\n"
    cfg = RunConfig.from_dict(make_config(
        workload_params={"questions_file": gsm8k_questions_file},
    ))
    store = ResultsStore(tmp_path)
    record = execute_run(
        cfg, store, VllmAdapter(cfg),
        ServerHandle(process=None, base_url=fake_server.base_url, external=True),
        log=lambda *_: None,
    )
    m = record["measured"]
    assert m["queue_depth"] is None
    assert m["kv_cache_usage"] is None
    assert m["num_preemptions"] is None
    assert record["status"] == "ok"


# -- config set --------------------------------------------------------------------

def test_k_stress_config_set():
    configs = load_configs(["configs/k_stress/kstress_*.yaml"])
    assert len(configs) == 16  # 2 kv x 4 conc x 2 repeats
    assert len({c.run_id for c in configs}) == 16
    assert len(group_by_server(configs)) == 2
    for cfg in configs:
        assert cfg.block == "k_stress"
        assert cfg.factors.weight_quant == "fp16"
        assert cfg.factors.spec_decode == "none"
        assert cfg.workload == "rag_shared_prefix"
        assert cfg.workload_params["prefix_overlap"] == "low"
        assert cfg.workload_params["doc_target_tokens"] == 7400
        assert cfg.workload_params["request_timeout_s"] == 1800
        assert cfg.repeat_idx in (0, 1)
        # pinned 40GB card (High-RAM OFF): both KV ceilings (~16 fp16, ~32
        # fp8) sit inside this grid -- see the generator docstring
        assert cfg.gpu_target == "a100_40gb"
        n = cfg.workload_params["num_requests"]
        assert n == {8: 64, 16: 96, 32: 160, 48: 192}[cfg.concurrency]
        cmd = VllmAdapter(cfg).build_launch_command()
        if cfg.factors.kv_quant == "fp8":
            assert cmd[cmd.index("--kv-cache-dtype") + 1] == "fp8"
        else:
            assert "--kv-cache-dtype" not in cmd


def test_k_stress_records_excluded_from_factorial():
    rec = {
        "run_id": "kstress_x",
        "config": {"block": "k_stress", "workload": "rag_shared_prefix",
                   "concurrency": 32, "repeat_idx": 0,
                   "factors": {"weight_quant": "fp16", "kv_quant": "fp8",
                               "spec_decode": "none"}},
        "env": {}, "status": "ok",
        "measured": {"goodput_tok_s": 100.0},
    }
    assert factorial_collect([rec]) == {}


# -- analysis ----------------------------------------------------------------------

def _ks_record(conc, kv, goodput, batch_mean, kv_usage_max, preemptions,
               repeat=0, prompt_mean=7600.0):
    return {
        "run_id": "kstress_%s_c%d_r%d" % (kv, conc, repeat),
        "config": {"block": "k_stress", "workload": "rag_shared_prefix",
                   "concurrency": conc, "repeat_idx": repeat,
                   "factors": {"weight_quant": "fp16", "kv_quant": kv,
                               "spec_decode": "none"}},
        "env": {}, "status": "ok",
        "measured": {
            "goodput_tok_s": goodput,
            "num_requests": 160, "total_completion_tokens": 160 * 250,
            "prompt_tokens_mean": prompt_mean,
            "emergent_batch_size": {"mean": batch_mean, "p50": batch_mean,
                                    "max": batch_mean + 1, "num_samples": 50},
            "kv_cache_usage": {"mean": kv_usage_max - 0.05, "p50": kv_usage_max,
                               "max": kv_usage_max, "num_samples": 50},
            "queue_depth": {"mean": 1.0, "p50": 0.0, "max": 5.0, "num_samples": 50},
            "num_preemptions": preemptions,
            "ttft_ms": {"p50": 900.0, "p95": 4000.0, "p99": 9000.0, "mean": 1200.0},
        },
    }


def _divergence_records():
    # 40GB-card story: fp16 ceiling ~16, fp8 ceiling ~32
    records = []
    for r in (0, 1):
        records += [
            _ks_record(8, "fp16", 700, 7.8, 0.42, 0, repeat=r),
            _ks_record(8, "fp8", 680, 7.9, 0.21, 0, repeat=r),
            _ks_record(32, "fp16", 900, 16.0, 1.0, 180, repeat=r),
            _ks_record(32, "fp8", 1480, 31.5, 0.95, 0, repeat=r),
        ]
    return records


def test_k_stress_report_shows_divergence():
    cells = ks_collect(_divergence_records())
    report = render_report(cells, pool_tokens=125000)
    assert "1.64x" in report                      # 1480/900 at conc 32
    assert "conc 32: FP16-KV CAPACITY-LIMITED" in report
    assert "conc 8: FP16-KV not capacity-limited" in report
    # predicted plateau: 125000 / (7600 + 250) = ~16
    assert "~16 concurrent requests" in report


def test_k_stress_report_missing_cells_and_cli(tmp_path):
    records = _divergence_records()[:3]  # drop fp8 at conc 32
    report = render_report(ks_collect(records))
    assert "Missing cells" in report

    store = ResultsStore(tmp_path / "results")
    for rec in _divergence_records():
        store.write(rec)
    assert ks_main([str(tmp_path / "results"), "--pool-tokens", "125000"]) == 0
    assert (tmp_path / "results" / "k_stress_report.md").exists()
    assert ks_main([str(tmp_path / "empty")]) == 1
