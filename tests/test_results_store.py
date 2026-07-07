from __future__ import annotations

import json

import pytest

from harness.results import ResultsStore


def _record(run_id="r1", status="ok"):
    return {
        "run_id": run_id,
        "config": {"block": "repro"},
        "env": {"git_commit": "abc"},
        "measured": {"throughput_tok_s": 1.0},
        "status": status,
    }


def test_write_load_roundtrip(tmp_path):
    store = ResultsStore(tmp_path)
    path = store.write(_record())
    assert path.exists()
    assert store.load("r1")["measured"]["throughput_tok_s"] == 1.0
    assert store.load("missing") is None


def test_is_complete_requires_ok_status(tmp_path):
    store = ResultsStore(tmp_path)
    store.write(_record("a", status="ok"))
    store.write(_record("b", status="partial"))
    assert store.is_complete("a")
    assert not store.is_complete("b")
    assert not store.is_complete("never-ran")


def test_write_is_atomic_no_tmp_left_behind(tmp_path):
    store = ResultsStore(tmp_path)
    store.write(_record())
    leftovers = list(store.runs_dir.glob("*.tmp"))
    assert leftovers == []


def test_overwrite_same_run_id(tmp_path):
    store = ResultsStore(tmp_path)
    store.write(_record(status="failed"))
    assert not store.is_complete("r1")
    store.write(_record(status="ok"))
    assert store.is_complete("r1")
    assert len(list(store.runs_dir.glob("*.json"))) == 1


def test_missing_keys_rejected(tmp_path):
    store = ResultsStore(tmp_path)
    with pytest.raises(ValueError, match="missing keys"):
        store.write({"run_id": "x"})


def test_export_jsonl(tmp_path):
    store = ResultsStore(tmp_path)
    store.write(_record("a"))
    store.write(_record("b"))
    out = tmp_path / "all.jsonl"
    assert store.export_jsonl(out) == 2
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    assert {json.loads(l)["run_id"] for l in lines} == {"a", "b"}
