from __future__ import annotations

import json
import subprocess
from pathlib import Path

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
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

    monkeypatch.setattr(pd.subprocess, "run", fake_run)
    lines = []
    rc = pd.predownload(["org/model"], attempts=2, backoff_s=1,
                        log=lines.append, sleep=lambda _: None)
    assert rc == 1
    assert any("TIMEOUT" in l for l in lines)
    # hf exhaustion hands off to the curl fallback, which also fails here
    assert any("curl fallback" in l and "FAILED" in l for l in lines)
    # the loud-failure guidance names the known CDN incident signature
    assert any("SignatureError" in l for l in lines)
    assert any("curl" in l for l in lines)


def test_partial_failure_still_reports_all(monkeypatch):
    monkeypatch.setattr(
        pd.subprocess, "run",
        lambda cmd, **kwargs:
            subprocess.CompletedProcess(
                cmd, 1 if any("bad/repo" in c for c in cmd) else 0),
    )
    lines = []
    rc = pd.predownload(["good/repo", "bad/repo"], attempts=2,
                        log=lines.append, sleep=lambda _: None)
    assert rc == 1
    assert any("bad/repo" in l and "FAILED after" in l for l in lines)
    # the fallback was tried for the failing repo before giving up
    assert any("curl fallback FAILED for bad/repo" in l for l in lines)


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


# ---------------------------------------------------------------------------
# curl fallback (2026-07-14 Xet CDN incident: hf-client route 403s with
# "SignatureError: invalid key pair id"; browser-UA resolve URLs work)
# ---------------------------------------------------------------------------

def _fake_tree():
    return [
        {"type": "file", "path": "config.json", "oid": "gitoid-config",
         "size": 12},
        {"type": "file", "path": "model-00001.safetensors",
         "oid": "gitoid-model", "lfs": {"oid": "sha256-model"}, "size": 5},
        {"type": "directory", "path": "original", "oid": "x", "size": 0},
        {"type": "file", "path": "original/consolidated.pth",
         "oid": "gitoid-pth", "lfs": {"oid": "sha256-pth"}, "size": 3},
    ]


_FAKE_CONTENT = {
    "config.json": b"x" * 12,
    "model-00001.safetensors": b"m" * 5,
    "original/consolidated.pth": b"p" * 3,
}


def _fake_curl_run(cmd, **kwargs):
    """Stand-in for subprocess.run(curl ...): serves the hub API JSON and
    writes fake blob bytes for resolve URLs."""
    url = cmd[-1]
    if "/revision/main" in url:
        out = json.dumps({"sha": "deadbeefcafe"}).encode()
        return subprocess.CompletedProcess(cmd, 0, stdout=out)
    if "/tree/main" in url:
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps(_fake_tree()).encode())
    assert "/resolve/main/" in url
    rel = url.split("/resolve/main/")[1]
    target = Path(cmd[cmd.index("-o") + 1])
    target.write_bytes(_FAKE_CONTENT[rel])
    return subprocess.CompletedProcess(cmd, 0)


@pytest.fixture
def hub_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("HF_TOKEN", "tok123")
    return tmp_path / "hub"


def test_plan_fallback_files_etag_is_lfs_sha_else_git_oid():
    plan = pd.plan_fallback_files(_fake_tree())
    assert [f["etag"] for f in plan] == [
        "gitoid-config", "sha256-model", "sha256-pth"]
    assert [f["size"] for f in plan] == [12, 5, 3]


def test_plan_fallback_files_pagination_guard():
    tree = [{"type": "file", "path": "f%d" % i, "oid": "o", "size": 1}
            for i in range(1000)]
    with pytest.raises(RuntimeError, match="pagination"):
        pd.plan_fallback_files(tree)


def test_curl_fallback_builds_standard_cache_layout(hub_cache):
    ok = pd.curl_fallback("org/model", log=lambda *_: None, run=_fake_curl_run)
    assert ok
    repo = hub_cache / "models--org--model"
    assert (repo / "refs" / "main").read_text() == "deadbeefcafe"
    snap = repo / "snapshots" / "deadbeefcafe"
    cfg = snap / "config.json"
    assert cfg.is_symlink()
    assert cfg.read_bytes() == b"x" * 12
    assert cfg.resolve() == (repo / "blobs" / "gitoid-config").resolve()
    nested = snap / "original" / "consolidated.pth"
    assert nested.resolve() == (repo / "blobs" / "sha256-pth").resolve()
    assert not list((repo / "blobs").glob("*.part")), "no leftover partials"


def test_curl_fallback_requests_use_browser_ua_and_token(hub_cache):
    seen = []

    def spy_run(cmd, **kwargs):
        seen.append(cmd)
        return _fake_curl_run(cmd, **kwargs)

    assert pd.curl_fallback("org/model", log=lambda *_: None, run=spy_run)
    for cmd in seen:
        assert "Mozilla/5.0" in cmd, "browser UA is load-bearing (edge routing)"
        assert "Authorization: Bearer tok123" in cmd


def test_curl_fallback_size_mismatch_fails_loudly(hub_cache):
    def bad_run(cmd, **kwargs):
        if "/resolve/main/" in cmd[-1]:
            Path(cmd[cmd.index("-o") + 1]).write_bytes(b"short")
            return subprocess.CompletedProcess(cmd, 0)
        return _fake_curl_run(cmd, **kwargs)

    lines = []
    assert not pd.curl_fallback("org/model", log=lines.append, run=bad_run)
    assert any("size mismatch" in l for l in lines)


def test_curl_fallback_skips_complete_blobs(hub_cache):
    blobs = hub_cache / "models--org--model" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "sha256-model").write_bytes(b"m" * 5)  # already complete

    def run_no_redownload(cmd, **kwargs):
        assert "model-00001.safetensors" not in cmd[-1], \
            "complete blob must not be re-downloaded"
        return _fake_curl_run(cmd, **kwargs)

    assert pd.curl_fallback("org/model", log=lambda *_: None,
                            run=run_no_redownload)


def test_predownload_falls_back_after_hf_exhausted(monkeypatch):
    fallback_calls = []
    monkeypatch.setattr(
        pd.subprocess, "run",
        lambda cmd, env=None, timeout=None: subprocess.CompletedProcess(cmd, 1),
    )
    monkeypatch.setattr(
        pd, "curl_fallback",
        lambda repo, timeout_s=0, log=print: fallback_calls.append(repo) or True,
    )
    rc = pd.predownload(["org/model"], attempts=2,
                        log=lambda *_: None, sleep=lambda _: None)
    assert rc == 0, "fallback success rescues the repo"
    assert fallback_calls == ["org/model"]


def test_curl_only_skips_hf_cli(monkeypatch):
    def no_hf(cmd, **kwargs):
        raise AssertionError("hf CLI must not run under --curl-only")

    monkeypatch.setattr(pd.subprocess, "run", no_hf)
    monkeypatch.setattr(pd, "curl_fallback",
                        lambda repo, timeout_s=0, log=print: True)
    assert pd.main(["org/model", "--curl-only"]) == 0
