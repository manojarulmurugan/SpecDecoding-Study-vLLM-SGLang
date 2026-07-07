from __future__ import annotations

from types import SimpleNamespace

import pytest

from harness.metrics import (
    aggregate_run,
    metric_value,
    parse_prometheus_text,
    percentile,
    spec_decode_stats,
    summarize_ms,
)

PROM_TEXT = """\
# HELP vllm:spec_decode_num_drafts_total drafts
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{model_name="m"} 100
vllm:spec_decode_num_draft_tokens_total{model_name="m"} 500
vllm:spec_decode_num_accepted_tokens_total{model_name="m"} 300
vllm:num_requests_running{model_name="m"} 3
"""


def test_parse_prometheus_sums_labels():
    metrics = parse_prometheus_text(
        'foo{a="1"} 2\nfoo{a="2"} 3\nbar 1\n# comment\n'
    )
    assert metrics["foo"] == 5
    assert metrics["bar"] == 1


def test_metric_value_tolerates_total_suffix():
    metrics = parse_prometheus_text(PROM_TEXT)
    assert metric_value(metrics, "vllm:spec_decode_num_drafts") == 100
    assert metric_value(metrics, "vllm:num_requests_running") == 3
    assert metric_value(metrics, "missing") is None


def test_spec_decode_stats_delta_and_tau():
    before = parse_prometheus_text(PROM_TEXT)
    after = parse_prometheus_text(
        'vllm:spec_decode_num_drafts_total{model_name="m"} 300\n'
        'vllm:spec_decode_num_draft_tokens_total{model_name="m"} 1500\n'
        'vllm:spec_decode_num_accepted_tokens_total{model_name="m"} 900\n'
    )
    stats = spec_decode_stats(before, after)
    assert stats["num_drafts"] == 200
    assert stats["num_draft_tokens"] == 1000
    assert stats["num_accepted_tokens"] == 600
    assert stats["acceptance_rate"] == pytest.approx(0.6)
    # tau = 1 (bonus token per verification step) + accepted/drafts
    assert stats["accepted_length_tau"] == pytest.approx(1 + 600 / 200)


def test_spec_decode_stats_absent_counters():
    assert spec_decode_stats({}, {}) is None


def test_percentile_interpolation():
    xs = [1, 2, 3, 4]
    assert percentile(xs, 0) == 1
    assert percentile(xs, 100) == 4
    assert percentile(xs, 50) == pytest.approx(2.5)


def test_summarize_ms_units():
    s = summarize_ms([0.1, 0.2, 0.3])
    assert s["p50"] == pytest.approx(200.0)
    assert summarize_ms([]) is None


def _req(tokens=10, ttft=0.5, decode=1.0, e2e=1.5, error=None):
    return SimpleNamespace(
        completion_tokens=tokens, ttft_s=ttft, decode_time_s=decode,
        e2e_s=e2e, itl_s=[0.1, 0.1], error=error,
    )


def test_aggregate_run_basics():
    results = [_req(), _req(tokens=20, e2e=2.0)]
    measured = aggregate_run(results, wall_time_s=3.0)
    assert measured["num_requests"] == 2
    assert measured["num_errors"] == 0
    assert measured["total_completion_tokens"] == 30
    assert measured["throughput_tok_s"] == pytest.approx(10.0)
    # per-request speeds: 10/1.5 and 20/2.0
    assert measured["request_tok_s_mean"] == pytest.approx((10 / 1.5 + 10.0) / 2)
    assert measured["accepted_length_tau"] is None


def test_aggregate_run_errors_and_spec_stats():
    results = [_req(), _req(error="boom")]
    measured = aggregate_run(
        results, wall_time_s=2.0,
        spec_stats={"accepted_length_tau": 3.5, "acceptance_rate": 0.7,
                    "num_drafts": 42},
    )
    assert measured["num_errors"] == 1
    assert measured["total_completion_tokens"] == 10  # errored request excluded
    assert measured["accepted_length_tau"] == 3.5
    assert measured["spec_num_drafts"] == 42
