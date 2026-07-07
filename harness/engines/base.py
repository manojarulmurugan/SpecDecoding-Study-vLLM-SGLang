"""Engine adapter contract (HARNESS_SPEC.md §5).

Command construction is separated from process launch so the full launch
command for every config can be validated in unit tests on a GPU-less
machine.
"""
from __future__ import annotations

import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests

from ..config import RunConfig
from ..metrics import parse_prometheus_text


@dataclass
class ServerHandle:
    process: Optional[subprocess.Popen]
    base_url: str
    log_path: Optional[Path] = None
    external: bool = False  # server not managed by us; never killed


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
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / ("server_%s.log" % time.strftime("%Y%m%d_%H%M%S"))
        cmd = self.build_launch_command()
        with open(log_path, "w") as fh:
            fh.write("+ %s\n" % " ".join(cmd))
            fh.flush()
            process = subprocess.Popen(
                cmd, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True
            )
        return ServerHandle(process=process, base_url=self.base_url(), log_path=log_path)

    def wait_ready(self, handle: ServerHandle, timeout_s: float = 2400.0,
                   poll_s: float = 5.0, log=print) -> None:
        """Poll /health until the server answers. Generous timeout: first
        launch includes the model download."""
        deadline = time.monotonic() + timeout_s
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
            time.sleep(poll_s)
        raise TimeoutError(
            "server not ready after %.0fs%s" % (timeout_s, self._log_tail(handle))
        )

    def teardown(self, handle: ServerHandle) -> None:
        if handle.external or handle.process is None:
            return
        handle.process.terminate()
        try:
            handle.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            handle.process.kill()
            handle.process.wait(timeout=30)

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

    @staticmethod
    def _log_tail(handle: ServerHandle, lines: int = 30) -> str:
        if not handle.log_path or not Path(handle.log_path).exists():
            return ""
        tail = Path(handle.log_path).read_text(errors="replace").splitlines()[-lines:]
        return "\nserver log tail (%s):\n%s" % (handle.log_path, "\n".join(tail))
