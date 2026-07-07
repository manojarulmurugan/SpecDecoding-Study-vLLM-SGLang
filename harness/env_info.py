"""Environment capture for result records: git commit, GPU, engine version."""
from __future__ import annotations

import datetime
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


def _run(cmd) -> Optional[str]:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def git_commit(repo_root: "Path | str | None" = None) -> Optional[str]:
    cmd = ["git"]
    if repo_root:
        cmd += ["-C", str(repo_root)]
    cmd += ["rev-parse", "HEAD"]
    return _run(cmd)


def gpu_info() -> Dict[str, Optional[str]]:
    name_driver = _run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
    )
    gpu_name = driver = cuda = None
    if name_driver:
        first = name_driver.splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        if len(parts) >= 2:
            gpu_name, driver = parts[0], parts[1]
    banner = _run(["nvidia-smi"])
    if banner:
        m = re.search(r"CUDA Version:\s*([\d.]+)", banner)
        if m:
            cuda = m.group(1)
    return {"gpu_name": gpu_name, "driver": driver, "cuda": cuda}


def collect_env(
    engine_version: Optional[str] = None, repo_root: "Path | str | None" = None
) -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "git_commit": git_commit(repo_root),
        "engine_version": engine_version,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    env.update(gpu_info())
    return env
