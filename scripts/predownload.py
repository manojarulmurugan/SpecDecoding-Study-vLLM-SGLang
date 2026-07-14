"""Pre-download HF checkpoints with retries, hard timeouts, loud failures.

Why this exists (2026-07-14 incident, evidence in
colab/archive_phase3b_xet_debug_20260714.ipynb): HF's Xet CDN served
presigned URLs with a broken signing key ("403 SignatureError: invalid key
pair id") across unrelated repos — hub-side infrastructure, not our config.
`pip uninstall hf-xet` did not help (the URLs are hub-issued) and
HF_HUB_DISABLE_XET has a known-ineffective history (huggingface_hub#3266);
it is still set here as belt-and-braces. A bare `!hf download` in a
notebook can sit silent indefinitely on this failure; this wrapper bounds
each attempt, retries with backoff (CDN incidents are transient), and
fails loudly with the diagnosis.

Curl fallback (added 14h into the same incident, once "transient" was
falsified): the hub routes hf-client requests to the us.gcp.cdn.hf.co xet
bridge (the edge with the broken signing key), but plain
`/<repo>/resolve/main/<file>` requests with a browser user-agent are
302'd to the healthy AWS cas-bridge edge (verified serving bytes while
the GCP edge still 403'd; etag mapping x-linked-etag == lfs sha256 for
LFS files, git blob oid otherwise, verified 2026-07-14). When the hf CLI
exhausts its attempts, curl_fallback() rebuilds the standard HF cache
layout (blobs/<etag> + snapshots/<sha>/ symlinks + refs/main) from those
URLs, so vLLM's hub lookups find complete blobs and never hit the broken
CDN route. TLS plus an exact size check against the tree listing guards
integrity; *.part files make interrupted downloads resumable.

Usage (from repo root, inside the vLLM venv):
    python scripts/predownload.py [repo ...] [--curl-only]
Defaults to the three checkpoints the Phase-3 runbooks need.
--curl-only skips the hf CLI entirely (use when hf is known-broken).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path
from typing import List, Optional

DEFAULT_REPOS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
]

HF_ENDPOINT = "https://huggingface.co"

# Load-bearing: with an hf-client UA the hub issues xet-bridge URLs on the
# GCP edge; with a browser UA, resolve URLs 302 to the healthy cas-bridge.
BROWSER_UA = "Mozilla/5.0"

FAILURE_GUIDANCE = """\
All download attempts failed, INCLUDING the curl/resolve-URL fallback
(which bypasses the xet-bridge CDN route that broke on 2026-07-14 with
'403 SignatureError: invalid key pair id'). That combination means either
no egress to huggingface.co at all (check: curl -sI https://huggingface.co)
or missing auth for gated repos (HF_TOKEN unset and no ~/.cache/huggingface
token). Fix those, or wait and re-run this cell -- the sweep cannot start
without a complete model cache."""


def _hf_cli() -> List[str]:
    """The `hf` CLI next to this interpreter (venv), else on PATH, else the
    legacy `huggingface-cli`."""
    sibling = Path(sys.executable).with_name("hf")
    if sibling.exists():
        return [str(sibling)]
    for name in ("hf", "huggingface-cli"):
        found = shutil.which(name)
        if found:
            return [found]
    raise RuntimeError("no hf/huggingface-cli executable found")


def _hub_cache_dir() -> Path:
    """Same resolution order huggingface_hub uses for the hub cache."""
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        val = os.environ.get(var)
        if val:
            return Path(val)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _hf_token() -> Optional[str]:
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        val = os.environ.get(var)
        if val:
            return val.strip()
    token_file = Path(
        os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface")
    ) / "token"
    if token_file.exists():
        return token_file.read_text().strip() or None
    return None


def _curl_cmd(url: str, token: Optional[str], extra: List[str]) -> List[str]:
    cmd = ["curl", "-sS", "-f", "-L", "--connect-timeout", "30",
           "--retry", "3", "--retry-delay", "5", "-A", BROWSER_UA]
    if token:
        cmd += ["-H", "Authorization: Bearer " + token]
    return cmd + extra + [url]


def _curl_json(url: str, token: Optional[str], timeout_s: float, run):
    proc = run(_curl_cmd(url, token, []), capture_output=True, timeout=timeout_s)
    if proc.returncode != 0:
        err = (proc.stderr or b"")
        if isinstance(err, bytes):
            err = err.decode(errors="replace")
        raise RuntimeError("curl exit %s for %s: %s" % (proc.returncode, url, err[-300:]))
    return json.loads(proc.stdout)


def plan_fallback_files(tree: list) -> List[dict]:
    """File entries as {path, etag, size}. Blob name under blobs/ is the
    etag the hub reports on resolve: LFS sha256 for LFS files, git blob
    oid otherwise (both verified against x-linked-etag headers)."""
    files = [e for e in tree if e.get("type") == "file"]
    if len(files) >= 1000:
        raise RuntimeError(
            "tree listing hit the pagination limit (%d entries); "
            "this fallback does not paginate" % len(files))
    return [{
        "path": e["path"],
        "etag": (e.get("lfs") or {}).get("oid") or e["oid"],
        "size": int(e["size"]),
    } for e in files]


def curl_fallback(repo: str, timeout_s: float = 1800.0, log=print,
                  run=subprocess.run) -> bool:
    """Rebuild the HF cache entry for `repo` via plain resolve URLs on the
    healthy cas-bridge edge. Returns True iff every file is present with
    the exact size the hub's tree listing reports."""
    try:
        token = _hf_token()
        api = "%s/api/models/%s" % (HF_ENDPOINT, repo)
        sha = _curl_json(api + "/revision/main", token, timeout_s, run)["sha"]
        tree = _curl_json(api + "/tree/main?recursive=true", token, timeout_s, run)
        files = plan_fallback_files(tree)
        repo_dir = _hub_cache_dir() / ("models--" + repo.replace("/", "--"))
        blobs_dir = repo_dir / "blobs"
        snap_dir = repo_dir / "snapshots" / sha
        blobs_dir.mkdir(parents=True, exist_ok=True)
        for n, f in enumerate(files, 1):
            blob = blobs_dir / f["etag"]
            part = blobs_dir / (f["etag"] + ".part")
            if blob.exists() and blob.stat().st_size == f["size"]:
                log("[predownload]   (%d/%d) cached  %s" % (n, len(files), f["path"]))
            else:
                if part.exists() and part.stat().st_size > f["size"]:
                    part.unlink()  # oversize partial cannot be resumed
                log("[predownload]   (%d/%d) curl    %s (%.1f MB)"
                    % (n, len(files), f["path"], f["size"] / 1e6))
                url = "%s/%s/resolve/main/%s" % (
                    HF_ENDPOINT, repo, urllib.parse.quote(f["path"]))
                proc = run(_curl_cmd(url, token, ["-C", "-", "-o", str(part)]),
                           timeout=timeout_s)
                if proc.returncode != 0:
                    raise RuntimeError("curl exit %s for %s" % (proc.returncode, f["path"]))
                got = part.stat().st_size
                if got != f["size"]:
                    raise RuntimeError("size mismatch for %s: got %d, want %d"
                                       % (f["path"], got, f["size"]))
                part.replace(blob)
            link = snap_dir / f["path"]
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(os.path.relpath(blob, link.parent))
        refs_dir = repo_dir / "refs"
        refs_dir.mkdir(parents=True, exist_ok=True)
        (refs_dir / "main").write_text(sha)
        log("[predownload] curl fallback OK: %s @ %s (%d files)"
            % (repo, sha[:12], len(files)))
        return True
    except Exception as exc:  # loud but per-repo: predownload() aggregates
        log("[predownload] curl fallback FAILED for %s: %s" % (repo, exc))
        return False


def download_once(repo: str, timeout_s: float, log=print) -> bool:
    env = dict(os.environ)
    env["HF_HUB_DISABLE_XET"] = "1"  # belt-and-braces; see module docstring
    cmd = _hf_cli() + ["download", repo]
    log("[predownload] %s (timeout %ds): %s" % (repo, timeout_s, " ".join(cmd)))
    try:
        proc = subprocess.run(cmd, env=env, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        log("[predownload] TIMEOUT after %ds: %s" % (timeout_s, repo))
        return False
    if proc.returncode != 0:
        log("[predownload] FAILED (exit %s): %s" % (proc.returncode, repo))
        return False
    return True


def predownload(
    repos: List[str],
    attempts: int = 3,
    timeout_s: float = 1800.0,
    backoff_s: float = 120.0,
    log=print,
    sleep=time.sleep,
    curl_only: bool = False,
) -> int:
    """0 if every repo landed; 1 otherwise (with guidance printed)."""
    failed: List[str] = []
    for repo in repos:
        ok = False
        if not curl_only:
            for attempt in range(1, attempts + 1):
                if download_once(repo, timeout_s, log=log):
                    ok = True
                    break
                if attempt < attempts:
                    delay = backoff_s * attempt
                    log("[predownload] attempt %d/%d failed for %s; retrying in %.0fs"
                        % (attempt, attempts, repo, delay))
                    sleep(delay)
        if not ok:
            if not curl_only:
                log("[predownload] hf CLI exhausted for %s -- switching to curl "
                    "fallback (resolve URLs via the healthy cas-bridge edge)" % repo)
            ok = curl_fallback(repo, timeout_s=timeout_s, log=log)
        if not ok:
            failed.append(repo)
    if failed:
        log("[predownload] FAILED after %d attempt(s) each: %s" % (attempts, failed))
        log(FAILURE_GUIDANCE)
        return 1
    log("[predownload] all %d checkpoint(s) present in the HF cache" % len(repos))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repos", nargs="*", default=None)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--backoff-s", type=float, default=120.0)
    parser.add_argument("--curl-only", action="store_true",
                        help="skip the hf CLI and go straight to the "
                             "resolve-URL curl fallback")
    args = parser.parse_args(argv)
    return predownload(
        args.repos or DEFAULT_REPOS,
        attempts=args.attempts, timeout_s=args.timeout_s,
        backoff_s=args.backoff_s, curl_only=args.curl_only,
    )


if __name__ == "__main__":
    sys.exit(main())
