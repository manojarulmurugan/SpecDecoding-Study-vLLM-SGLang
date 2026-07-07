from __future__ import annotations

import pytest

from harness.load import run_closed_loop, stream_completion


def test_stream_completion_against_fake_server(fake_server):
    result = stream_completion(
        fake_server.base_url, "fake-model", "hello", index=0, max_tokens=32
    )
    assert result.error is None
    assert result.text == " The answer is 6."
    assert result.finish_reason == "stop"
    # token count must come from usage, not chunk count (spec decode packs
    # multiple tokens per chunk; the fake reports 2 tokens/chunk)
    assert result.completion_tokens == result.n_chunks * 2
    assert result.prompt_tokens == 11
    assert result.ttft_s is not None and result.ttft_s > 0
    assert result.e2e_s >= result.ttft_s
    assert result.decode_time_s == pytest.approx(result.e2e_s - result.ttft_s)
    assert len(result.itl_s) == result.n_chunks - 1


def test_stream_completion_records_error_not_raises():
    result = stream_completion(
        "http://127.0.0.1:9", "m", "p", index=0, max_tokens=4, timeout_s=0.5
    )
    assert result.error is not None
    assert result.completion_tokens in (None, 0)


def test_closed_loop_all_prompts_served(fake_server):
    prompts = ["p%d" % i for i in range(7)]
    out = run_closed_loop(
        fake_server.base_url, "fake-model", prompts,
        concurrency=3, max_tokens=16, progress_every=0,
    )
    assert len(out.results) == 7
    assert all(r.error is None for r in out.results)
    assert sorted(r.index for r in out.results) == list(range(7))
    assert out.wall_time_s > 0
    assert fake_server.request_count == 7


def test_closed_loop_warmup_not_in_results(fake_server):
    out = run_closed_loop(
        fake_server.base_url, "fake-model", ["a", "b"],
        concurrency=1, max_tokens=16, warmup_requests=2, progress_every=0,
    )
    assert len(out.results) == 2
    assert fake_server.request_count == 4  # 2 warmup + 2 timed


def test_payload_shape(fake_server):
    run_closed_loop(
        fake_server.base_url, "fake-model", ["x"], concurrency=1,
        max_tokens=99, temperature=0.0, stop=["```"], seed=42, progress_every=0,
    )
    payload = fake_server.seen_payloads[-1]
    assert payload["model"] == "fake-model"
    assert payload["max_tokens"] == 99
    assert payload["temperature"] == 0.0
    assert payload["stop"] == ["```"]
    assert payload["seed"] == 42
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
