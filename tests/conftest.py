from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests.fake_server import FakeVllmServer  # noqa: E402


@pytest.fixture
def fake_server():
    server = FakeVllmServer()
    server.start()
    yield server
    server.stop()


@pytest.fixture
def gsm8k_questions_file(tmp_path):
    """Tiny GSM8K-format fixture whose answer matches the fake server's
    canned completion (' The answer is 6.')."""
    path = tmp_path / "gsm8k_tiny.jsonl"
    rows = [
        {"question": "What is 2 + 4?", "answer": "2 + 4 = 6.\n#### 6"},
        {"question": "What is 10 - 4?", "answer": "10 - 4 = 6.\n#### 6"},
        {"question": "What is 3 * 2?", "answer": "3 * 2 = 6.\n#### 6"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


def make_config(**overrides):
    """Valid baseline RunConfig dict; tests override what they probe."""
    from copy import deepcopy

    base = {
        "block": "repro",
        "engine": "vllm",
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "factors": {"weight_quant": "fp16", "kv_quant": "fp16", "spec_decode": "none"},
        "workload": "gsm8k",
        "workload_params": {"num_requests": 3, "max_new_tokens": 32},
        "concurrency": 1,
        "decoding": "greedy",
        "seed": 1234,
        "repeat_idx": 0,
        "warmup_requests": 0,
        "gpu_target": "a100",
        "engine_args": {"port": 8000},
    }
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged
