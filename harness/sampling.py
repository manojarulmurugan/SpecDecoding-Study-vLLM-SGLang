"""In-flight sampling of engine gauges during the timed window.

PROJECT_SPEC §7.2: batch size in vLLM is scheduler-dependent — the harness
sets offered load (concurrency) and MEASURES the running batch. The sampler
polls /metrics on a background thread and records:

- running batch          (vllm:num_requests_running)
- queue depth            (vllm:num_requests_waiting)
- KV-cache usage frac    (vllm:kv_cache_usage_perc; 1.0 = pool saturated)

Gauge names are candidate lists: verified against current vLLM docs, but a
rename in the pinned engine version degrades to a missing column, never a
crashed run (the K-stress addendum design review, 2026-07-09).
"""
from __future__ import annotations

import threading
from typing import Dict, List, Optional, Sequence, Tuple

import requests

from .metrics import metric_value, parse_prometheus_text

GAUGE_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "running": ("vllm:num_requests_running",),
    "waiting": ("vllm:num_requests_waiting",),
    "kv_cache_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
}

# Counter (not gauge): delta'd across the timed window in run.py, like the
# spec-decode counters. metric_value() tolerates a `_total` suffix.
PREEMPTIONS_COUNTER = "vllm:num_preemptions"


class MetricsSampler:
    def __init__(
        self,
        base_url: str,
        interval_s: float = 1.0,
        gauges: Optional[Dict[str, Sequence[str]]] = None,
        timeout_s: float = 10.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.interval_s = interval_s
        self.gauges = dict(gauges or GAUGE_CANDIDATES)
        self.timeout_s = timeout_s
        self.samples: Dict[str, List[float]] = {name: [] for name in self.gauges}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _loop(self) -> None:
        session = requests.Session()
        try:
            while not self._stop.is_set():
                try:
                    resp = session.get(self.base_url + "/metrics", timeout=self.timeout_s)
                    resp.raise_for_status()
                    metrics = parse_prometheus_text(resp.text)
                    for name, candidates in self.gauges.items():
                        for candidate in candidates:
                            value = metric_value(metrics, candidate)
                            if value is not None:
                                self.samples[name].append(value)
                                break
                except requests.RequestException:
                    pass  # transient scrape failure: skip the sample, keep going
                self._stop.wait(self.interval_s)
        finally:
            session.close()

    def start(self) -> "MetricsSampler":
        if self._thread is not None:
            raise RuntimeError("sampler already started")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> Dict[str, List[float]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(10.0, self.timeout_s + self.interval_s))
        return {name: list(values) for name, values in self.samples.items()}
