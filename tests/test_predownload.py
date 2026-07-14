from __future__ import annotations

import subprocess

import pytest

import scripts.predownload as pd


@pytest.fixture(autouse=True)
def fake_cli(monkeypatch):
    monkeypatch.setattr(pd, "_hf_cli", lambda: ["hf"])


def test_success_first_attempt(monkeypatch):
    calls = []

    def fake_run(cmd, env=None, timeout=None):
        calls.append((cmd, env, timeout))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(pd.subprocess, "run", fake_run)
    rc = pd.predownload(["org/model"], attempts=3, timeout_s=60,
                        log=lambda *_: None, sleep=lambda _: None)
    assert rc == 0
    assert len(calls) == 1
    cmd, env, timeout = calls[0]
    assert cmd == ["hf", "download", "org/model"]
    assert env["HF_HUB_DISABLE_XET"] == "1"
    assert timeout == 60


def test_retry_with_backoff_then_success(monkeypatch):
    outcomes = iter([1, 1, 0])  # two failures, then success
    sleeps = []
    monkeypatch.setattr(
        pd.subprocess, "run",
        lambda cmd, env=None, timeout=None:
            subprocess.CompletedProcess(cmd, next(outcomes)),
    )
    rc = pd.predownload(["org/model"], attempts=3, backoff_s=10,
                        log=lambda *_: None, sleep=sleeps.append)
    assert rc == 0
    assert sleeps == [10, 20], "linear backoff: backoff_s * attempt"


def test_timeout_counts_as_failure(monkeypatch):
    def fake_run(cmd, env=None, timeout=None):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(pd.subprocess, "run", fake_run)
    lines = []
    rc = pd.predownload(["org/model"], attempts=2, backoff_s=1,
                        log=lines.append, sleep=lambda _: None)
    assert rc == 1
    assert any("TIMEOUT" in l for l in lines)
    # the loud-failure guidance names the known CDN incident signature
    assert any("SignatureError" in l for l in lines)
    assert any("curl" in l for l in lines)


def test_partial_failure_still_reports_all(monkeypatch):
    monkeypatch.setattr(
        pd.subprocess, "run",
        lambda cmd, env=None, timeout=None:
            subprocess.CompletedProcess(cmd, 1 if "bad/repo" in cmd else 0),
    )
    lines = []
    rc = pd.predownload(["good/repo", "bad/repo"], attempts=2,
                        log=lines.append, sleep=lambda _: None)
    assert rc == 1
    assert any("bad/repo" in l and "FAILED after" in l for l in lines)


def test_cli_main_defaults(monkeypatch):
    seen = []
    monkeypatch.setattr(
        pd.subprocess, "run",
        lambda cmd, env=None, timeout=None:
            seen.append(cmd) or subprocess.CompletedProcess(cmd, 0),
    )
    assert pd.main([]) == 0
    assert len(seen) == 3  # the three default checkpoints
    assert any("meta-llama/Llama-3.1-8B-Instruct" in c for c in seen)
