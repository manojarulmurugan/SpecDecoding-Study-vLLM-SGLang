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
fails loudly with the diagnosis and the known-working manual fallback.

Usage (from repo root, inside the vLLM venv):
    python scripts/predownload.py [repo ...]
Defaults to the three checkpoints the Phase-3 runbooks need.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

DEFAULT_REPOS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
]

FAILURE_GUIDANCE = """\
All download attempts failed. If errors show '403 ... SignatureError:
invalid key pair id' from a xet-bridge/CDN URL, this is Hugging Face-side
CDN trouble (seen 2026-07-14 across unrelated repos): wait a few hours and
re-run this cell -- do NOT debug locally, client-side fixes don't help.
Manual fallback proven to work during that incident: plain resolve URLs
with a browser user-agent, e.g.
  curl -L -A "Mozilla/5.0" -H "Authorization: Bearer $HF_TOKEN" \\
    -o <file> https://huggingface.co/<repo>/resolve/main/<file>
(routes via the healthy cas-bridge edge)."""


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
) -> int:
    """0 if every repo landed; 1 otherwise (with guidance printed)."""
    failed: List[str] = []
    for repo in repos:
        ok = False
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
    args = parser.parse_args(argv)
    return predownload(
        args.repos or DEFAULT_REPOS,
        attempts=args.attempts, timeout_s=args.timeout_s,
        backoff_s=args.backoff_s,
    )


if __name__ == "__main__":
    sys.exit(main())
