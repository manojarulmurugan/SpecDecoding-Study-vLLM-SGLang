"""Server process-lifetime guarantees (Phase-3b Bug A/B hardening).

These use REAL child processes: the zombie bug was precisely the gap
between 'the tracked pid is dead' and 'the process group is dead'.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

import harness.engines.base as base_mod
from harness.config import RunConfig
from harness.engines.base import ServerHandle, gpu_compute_pids
from harness.engines.vllm_adapter import VllmAdapter
from tests.conftest import make_config


def _adapter(**overrides):
    return VllmAdapter(RunConfig.from_dict(make_config(**overrides)))


def _group_alive(pgid) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


# -- Bug B: teardown must kill the WHOLE process group -------------------------

def test_teardown_kills_grandchildren(tmp_path):
    # leader spawns a background child (the EngineCore stand-in), then execs
    # into a long sleep: two live processes, one tracked pid
    proc = subprocess.Popen(
        ["/bin/sh", "-c", "sleep 300 & exec sleep 300"],
        start_new_session=True,
    )
    handle = ServerHandle(process=proc, base_url="http://127.0.0.1:9",
                          pgid=proc.pid)
    assert _group_alive(proc.pid)
    _adapter().teardown(handle, log=lambda *_: None)
    deadline = time.monotonic() + 10
    while _group_alive(proc.pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _group_alive(proc.pid), (
        "process group still alive after teardown -- the Phase-3b zombie"
    )


def test_teardown_survives_already_dead_group():
    proc = subprocess.Popen(["/bin/sh", "-c", "exit 0"], start_new_session=True)
    proc.wait()
    handle = ServerHandle(process=proc, base_url="http://127.0.0.1:9",
                          pgid=proc.pid)
    _adapter().teardown(handle, log=lambda *_: None)  # must not raise


def test_teardown_never_touches_external_servers():
    handle = ServerHandle(process=None, base_url="http://127.0.0.1:9",
                          external=True, pgid=os.getpid())
    _adapter().teardown(handle, log=lambda *_: None)  # must not signal us


# -- Bug B: launch refuses an occupied GPU --------------------------------------

def test_launch_refuses_occupied_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(base_mod, "gpu_compute_pids", lambda: ["31358"])
    with pytest.raises(RuntimeError, match="GPU already held.*31358"):
        _adapter().launch(tmp_path)


def test_gpu_compute_pids_none_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(base_mod.shutil, "which", lambda _: None)
    assert gpu_compute_pids() is None


# -- Bug B: env plumbing ----------------------------------------------------------

def test_launch_passes_env_overrides_and_captures_pgid(tmp_path, monkeypatch):
    monkeypatch.setattr(base_mod, "gpu_compute_pids", lambda: None)
    captured = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured.update(kwargs)
        return FakeProc()

    monkeypatch.setattr(base_mod.subprocess, "Popen", fake_popen)
    adapter = _adapter(engine_args={"env": {"VLLM_ATTENTION_BACKEND": "FLASHINFER"}})
    handle = adapter.launch(tmp_path)
    assert handle.pgid == 4242
    assert captured["start_new_session"] is True
    assert captured["env"]["VLLM_ATTENTION_BACKEND"] == "FLASHINFER"
    assert "PATH" in captured["env"], "overrides must extend os.environ, not replace it"
    # env overrides are part of the server signature (they change the server)
    plain = _adapter()
    assert adapter.config.server_signature() != plain.config.server_signature()
    # and recorded in the launch log for provenance
    log_text = next(tmp_path.glob("server_*.log")).read_text()
    assert "VLLM_ATTENTION_BACKEND" in log_text


def test_launch_without_env_inherits_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(base_mod, "gpu_compute_pids", lambda: None)
    captured = {}

    class FakeProc:
        pid = 1

    monkeypatch.setattr(
        base_mod.subprocess, "Popen",
        lambda cmd, **kw: captured.update(kw) or FakeProc(),
    )
    _adapter().launch(tmp_path)
    assert captured["env"] is None  # inherit parent environment untouched


# -- Bug A: stall watchdog ---------------------------------------------------------

def test_wait_ready_fails_fast_when_log_and_cache_both_frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(base_mod, "hf_cache_size", lambda: 12345)  # frozen
    proc = subprocess.Popen(["/bin/sh", "-c", "sleep 60"], start_new_session=True)
    log_path = tmp_path / "server.log"
    log_path.write_text("+ vllm serve ...\nUsing FlashAttention version 2\n")
    handle = ServerHandle(process=proc, base_url="http://127.0.0.1:9",
                          log_path=log_path, pgid=proc.pid)
    try:
        with pytest.raises(RuntimeError, match="STALLED"):
            _adapter().wait_ready(handle, timeout_s=30, poll_s=0.05,
                                  stall_timeout_s=0.5, log=lambda *_: None)
    finally:
        _adapter().teardown(handle, log=lambda *_: None)


def test_wait_ready_tolerates_silent_log_during_active_download(tmp_path, monkeypatch):
    """The false-positive scenario from the 2026-07-12 bug report: tqdm is
    silent on non-tty stdout, so a cold weight download produces a frozen
    log -- but the HF cache grows. A growing cache must keep the watchdog
    quiet; the wait then ends in the ordinary TimeoutError, never STALLED."""
    ticks = iter(range(10_000))
    monkeypatch.setattr(base_mod, "hf_cache_size", lambda: next(ticks))
    proc = subprocess.Popen(["/bin/sh", "-c", "sleep 60"], start_new_session=True)
    log_path = tmp_path / "server.log"
    log_path.write_text("Loading model from scratch...\n")  # then silence
    handle = ServerHandle(process=proc, base_url="http://127.0.0.1:9",
                          log_path=log_path, pgid=proc.pid)
    try:
        with pytest.raises(TimeoutError, match="not ready"):
            _adapter().wait_ready(handle, timeout_s=2, poll_s=0.05,
                                  stall_timeout_s=0.3, log=lambda *_: None)
    finally:
        _adapter().teardown(handle, log=lambda *_: None)


def test_hf_cache_size_walks_the_hub_cache(tmp_path, monkeypatch):
    from harness.engines.base import hf_cache_size

    hub = tmp_path / "hub" / "models--x" / "blobs"
    hub.mkdir(parents=True)
    (hub / "abc123").write_bytes(b"x" * 1000)
    (hub / "def456.incomplete").write_bytes(b"y" * 500)  # in-flight download
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    assert hf_cache_size() == 1500
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "missing"))
    assert hf_cache_size() == 0


def test_wait_ready_reports_process_exit(tmp_path):
    proc = subprocess.Popen(["/bin/sh", "-c", "exit 3"], start_new_session=True)
    proc.wait()
    log_path = tmp_path / "server.log"
    log_path.write_text("boom\n")
    handle = ServerHandle(process=proc, base_url="http://127.0.0.1:9",
                          log_path=log_path, pgid=proc.pid)
    with pytest.raises(RuntimeError, match="exited with code 3"):
        _adapter().wait_ready(handle, timeout_s=5, poll_s=0.05,
                              log=lambda *_: None)


# -- Bug A postmortem: backend recording ---------------------------------------------

def test_detect_attention_backend(tmp_path):
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "INFO 07-11 [gpu_model_runner.py] Using FlashAttention version 2\n"
        "INFO loading weights...\n"
    )
    handle = ServerHandle(process=None, base_url="x", log_path=log_path)
    backend = _adapter().detect_attention_backend(handle)
    assert backend is not None and "FlashAttention" in backend

    log_path.write_text("nothing relevant here\n")
    assert _adapter().detect_attention_backend(handle) is None
    assert _adapter().detect_attention_backend(
        ServerHandle(process=None, base_url="x")) is None


def test_run_record_carries_attention_backend(fake_server, gsm8k_questions_file, tmp_path):
    from harness.results import ResultsStore
    from harness.run import execute_run

    cfg = RunConfig.from_dict(make_config(
        workload_params={"questions_file": gsm8k_questions_file},
    ))
    record = execute_run(
        cfg, ResultsStore(tmp_path), VllmAdapter(cfg),
        ServerHandle(process=None, base_url=fake_server.base_url, external=True),
        log=lambda *_: None,
    )
    assert "attention_backend" in record["env"]  # None here (no log), but recorded
