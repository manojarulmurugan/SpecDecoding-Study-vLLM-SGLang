"""Engine adapter contract (HARNESS_SPEC.md §5).

Command construction is separated from process launch so the full launch
command for every config can be validated in unit tests on a GPU-less
machine.

Process-lifetime guarantees (hardened after the Phase-3b zombie incident,
2026-07-11): servers are launched as their own process GROUP, teardown
signals the whole group (vLLM V1 spawns a separate EngineCore child that a
plain terminate() orphans, leaving ~16GB of GPU memory held), and launch()
refuses to start a server while nvidia-smi still reports compute processes
-- so a failed teardown surfaces as one precise error, never a cascade of
generic "Engine core initialization failed" groups.
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests

from ..config import RunConfig
from ..metrics import parse_prometheus_text


def hf_cache_size() -> int:
    """Total bytes under the HF hub cache.

    Empirically verified (2026-07-12): huggingface_hub's tqdm progress bars
    auto-disable when stdout is not a tty, so a server whose output is
    redirected to a log file writes NOTHING during a cold weight download --
    but in-flight downloads land in the cache as blobs/*.incomplete, so
    cache size grows continuously. This is the watchdog's second activity
    signal: a genuine wedge (Phase-3b Bug A) had log AND cache both frozen.
    """
    root = (
        os.environ.get("HF_HUB_CACHE")
        or os.path.join(
            os.environ.get("HF_HOME")
            or os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
            "hub",
        )
    )
    total = 0
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            try:
                total += os.stat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return total


def gpu_compute_pids() -> Optional[List[str]]:
    """PIDs currently holding GPU compute contexts, or None when nvidia-smi
    is unavailable (GPU-less dev machines)."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


@dataclass
class ServerHandle:
    process: Optional[subprocess.Popen]
    base_url: str
    log_path: Optional[Path] = None
    external: bool = False  # server not managed by us; never killed
    # Process-group id (== leader pid under start_new_session). Captured at
    # launch so teardown can signal the WHOLE group even after the leader
    # has been reaped.
    pgid: Optional[int] = None


class EngineAdapter(ABC):
    def __init__(self, config: RunConfig):
        self.config = config

    # -- to implement per engine -------------------------------------------

    @abstractmethod
    def build_launch_command(self) -> List[str]:
        """The exact argv used to start the server. Pure; unit-testable."""

    # -- shared machinery ----------------------------------------------------

    def base_url(self) -> str:
        ea = self.config.engine_args
        return "http://%s:%d" % (ea.host, ea.port)

    def launch(self, log_dir: "Path | str") -> ServerHandle:
        holders = gpu_compute_pids()
        if holders:
            raise RuntimeError(
                "refusing to launch: GPU already held by compute process(es) "
                "pid=%s -- a previous server was not torn down (or another "
                "job is running). Kill them (kill -9 <pid>) or restart the "
                "runtime before continuing." % ",".join(holders)
            )
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / ("server_%s.log" % time.strftime("%Y%m%d_%H%M%S"))
        cmd = self.build_launch_command()
        env = None
        if self.config.engine_args.env:
            env = dict(os.environ)
            env.update(self.config.engine_args.env)
        with open(log_path, "w") as fh:
            fh.write("+ %s\n" % " ".join(cmd))
            if self.config.engine_args.env:
                fh.write("+ env overrides: %s\n" % self.config.engine_args.env)
            fh.flush()
            process = subprocess.Popen(
                cmd, stdout=fh, stderr=subprocess.STDOUT,
                start_new_session=True, env=env,
            )
        return ServerHandle(
            process=process, base_url=self.base_url(), log_path=log_path,
            pgid=process.pid,  # session leader: pgid == pid
        )

    def wait_ready(self, handle: ServerHandle, timeout_s: float = 2400.0,
                   poll_s: float = 5.0, stall_timeout_s: float = 600.0,
                   log=print) -> None:
        """Poll /health until the server answers.

        Failure modes, distinguished deliberately:
        - process exit -> immediate failure with log tail;
        - STALL: process alive, no /health, AND *both* activity signals
          frozen for ``stall_timeout_s``: the server log AND the HF cache
          size. The second signal is essential: tqdm auto-disables on
          non-tty stdout, so a cold ~16GB weight download writes NOTHING to
          the redirected log while the cache grows for many minutes
          (empirically verified 2026-07-12, after the watchdog's first
          version false-positive-killed a cold-cache launch). The genuine
          Phase-3b Bug-A wedge had log and cache both static at idle power.
        """
        deadline = time.monotonic() + timeout_s
        last_fingerprint = None
        last_activity = time.monotonic()
        while time.monotonic() < deadline:
            if handle.process is not None and handle.process.poll() is not None:
                raise RuntimeError(
                    "server exited with code %s before becoming ready%s"
                    % (handle.process.returncode, self._log_tail(handle))
                )
            try:
                r = requests.get(handle.base_url + "/health", timeout=5)
                if r.status_code == 200:
                    return
            except requests.RequestException:
                pass
            if handle.log_path is not None:
                try:
                    log_size = Path(handle.log_path).stat().st_size
                except OSError:
                    log_size = -1
                fingerprint = (log_size, hf_cache_size())
                if fingerprint != last_fingerprint:
                    last_fingerprint = fingerprint
                    last_activity = time.monotonic()
                elif time.monotonic() - last_activity > stall_timeout_s:
                    raise RuntimeError(
                        "server STALLED: alive but no /health, and log + HF "
                        "cache both unchanged for %.0fs (an active download "
                        "would grow the cache)%s"
                        % (stall_timeout_s, self._log_tail(handle))
                    )
            time.sleep(poll_s)
        raise TimeoutError(
            "server not ready after %.0fs%s" % (timeout_s, self._log_tail(handle))
        )

    def teardown(self, handle: ServerHandle, log=print) -> None:
        """Kill the server and its WHOLE process group, then verify the GPU
        is actually released. vLLM V1 runs EngineCore as a separate child
        process; signaling only the tracked pid orphans it with the model
        weights still resident (Phase-3b Bug B)."""
        if handle.external or handle.process is None:
            return

        def _signal_group(sig) -> None:
            if handle.pgid is None:
                return
            try:
                os.killpg(handle.pgid, sig)
            except (ProcessLookupError, PermissionError, OSError):
                pass

        _signal_group(signal.SIGTERM)
        try:
            handle.process.terminate()
        except OSError:
            pass
        try:
            handle.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            _signal_group(signal.SIGKILL)
            handle.process.kill()
            try:
                handle.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        # The leader can die while children linger; always final-sweep the
        # group with SIGKILL (no-op if everything is already gone).
        _signal_group(signal.SIGKILL)

        # Verify release: give the driver a few seconds to reap contexts.
        for _ in range(10):
            holders = gpu_compute_pids()
            if not holders:  # empty list or None (no nvidia-smi)
                return
            time.sleep(2)
        log(
            "[teardown] WARNING: GPU still held by pid(s) %s after group "
            "kill -- kill them manually or restart the runtime before the "
            "next launch (launch() will refuse until the GPU is free)."
            % ",".join(holders)
        )

    def scrape_metrics(self, handle: ServerHandle) -> Dict[str, float]:
        try:
            r = requests.get(handle.base_url + "/metrics", timeout=15)
            r.raise_for_status()
        except requests.RequestException:
            return {}
        return parse_prometheus_text(r.text)

    def server_version(self, handle: ServerHandle) -> Optional[str]:
        try:
            r = requests.get(handle.base_url + "/version", timeout=10)
            r.raise_for_status()
            return r.json().get("version")
        except (requests.RequestException, ValueError):
            return None

    _BACKEND_PATTERNS = re.compile(
        r"(Using \S*[Ff]lash\S* version \S+|Using \S*[Ff]lash\S*|"
        r"[Aa]ttention backend[^\n]*)"
    )

    def detect_attention_backend(self, handle: ServerHandle) -> Optional[str]:
        """Scrape the attention-backend selection line from the server log.

        Recorded per run so backend differences (FP8-KV historically selects
        FlashInfer while FP16-KV picks FlashAttention) are data, not
        after-the-fact guesswork -- Phase-3b Bug A postmortem.
        """
        if not handle.log_path or not Path(handle.log_path).exists():
            return None
        try:
            text = Path(handle.log_path).read_text(errors="replace")
        except OSError:
            return None
        matches = self._BACKEND_PATTERNS.findall(text)
        return matches[0].strip() if matches else None

    @staticmethod
    def _log_tail(handle: ServerHandle, lines: int = 30) -> str:
        if not handle.log_path or not Path(handle.log_path).exists():
            return ""
        tail = Path(handle.log_path).read_text(errors="replace").splitlines()[-lines:]
        return "\nserver log tail (%s):\n%s" % (handle.log_path, "\n".join(tail))
