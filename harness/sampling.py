"""In-flight sampling of the engine's emergent batch size.

PROJECT_SPEC §7.2: batch size in vLLM is scheduler-dependent — the harness
sets offered load (concurrency) and MEASURES the running batch. This sampler
polls the server's /metrics endpoint on a background thread during the timed
window and records the ``vllm:num_requests_running`` gauge.
"""
from __future__ import annotations

import threading
from typing import List, Optional

import requests

from .metrics import metric_value, parse_prometheus_text

RUNNING_GAUGE = "vllm:num_requests_running"


class BatchSizeSampler:
    def __init__(
        self,
        base_url: str,
        interval_s: float = 1.0,
        metric_name: str = RUNNING_GAUGE,
        timeout_s: float = 10.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.interval_s = interval_s
        self.metric_name = metric_name
        self.timeout_s = timeout_s
        self.samples: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _loop(self) -> None:
        session = requests.Session()
        try:
            while not self._stop.is_set():
                try:
                    resp = session.get(self.base_url + "/metrics", timeout=self.timeout_s)
                    resp.raise_for_status()
                    value = metric_value(parse_prometheus_text(resp.text), self.metric_name)
                    if value is not None:
                        self.samples.append(value)
                except requests.RequestException:
                    pass  # transient scrape failure: skip the sample, keep going
                self._stop.wait(self.interval_s)
        finally:
            session.close()

    def start(self) -> "BatchSizeSampler":
        if self._thread is not None:
            raise RuntimeError("sampler already started")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> List[float]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(10.0, self.timeout_s + self.interval_s))
        return list(self.samples)
