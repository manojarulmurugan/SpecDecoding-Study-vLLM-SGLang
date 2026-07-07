"""Closed-loop load driver against an OpenAI-compatible completions endpoint.

Design (HARNESS_SPEC.md §6): the independent variable is offered load -- a
fixed pool of N concurrent clients. Batch size inside the engine is emergent
and never set. Requests stream (SSE) so we can time TTFT and inter-chunk
latency; token counts come from the final ``usage`` chunk because with
speculative decoding one streamed chunk can carry multiple tokens.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import requests

DEFAULT_REQUEST_TIMEOUT_S = 900.0


@dataclass
class RequestResult:
    index: int
    text: str = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    n_chunks: int = 0
    ttft_s: Optional[float] = None
    decode_time_s: Optional[float] = None  # first chunk -> last chunk
    e2e_s: Optional[float] = None          # send -> last chunk
    itl_s: List[float] = field(default_factory=list)
    finish_reason: Optional[str] = None
    error: Optional[str] = None


def stream_completion(
    base_url: str,
    model: str,
    prompt: str,
    index: int,
    max_tokens: int,
    temperature: float = 0.0,
    stop: Optional[List[str]] = None,
    seed: Optional[int] = None,
    timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    session: Optional[requests.Session] = None,
) -> RequestResult:
    result = RequestResult(index=index)
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if stop:
        payload["stop"] = stop
    if seed is not None:
        payload["seed"] = seed

    http = session or requests
    send_ts = time.monotonic()
    prev_chunk_ts: Optional[float] = None
    last_chunk_ts: Optional[float] = None
    try:
        with http.post(
            base_url.rstrip("/") + "/v1/completions",
            json=payload,
            stream=True,
            timeout=timeout_s,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage")
                if usage:
                    result.prompt_tokens = usage.get("prompt_tokens")
                    result.completion_tokens = usage.get("completion_tokens")
                choices = chunk.get("choices") or []
                if not choices:
                    continue  # usage-only chunk
                now = time.monotonic()
                result.n_chunks += 1
                if result.ttft_s is None:
                    result.ttft_s = now - send_ts
                if prev_chunk_ts is not None:
                    result.itl_s.append(now - prev_chunk_ts)
                prev_chunk_ts = now
                last_chunk_ts = now
                result.text += choices[0].get("text") or ""
                if choices[0].get("finish_reason"):
                    result.finish_reason = choices[0]["finish_reason"]
    except Exception as exc:  # recorded, not raised: one bad request != dead run
        result.error = "%s: %s" % (type(exc).__name__, exc)
        return result

    if last_chunk_ts is not None:
        result.e2e_s = last_chunk_ts - send_ts
        if result.ttft_s is not None:
            result.decode_time_s = last_chunk_ts - (send_ts + result.ttft_s)
    if result.completion_tokens is None:
        # Fallback when the server doesn't honor include_usage. Chunk count
        # UNDER-counts with speculative decoding (multi-token chunks); the
        # aggregator treats this as best-effort.
        result.completion_tokens = result.n_chunks
    return result


@dataclass
class LoadRunResult:
    results: List[RequestResult]
    wall_time_s: float


def run_closed_loop(
    base_url: str,
    model: str,
    prompts: Sequence[str],
    concurrency: int,
    max_tokens: int,
    temperature: float = 0.0,
    stop: Optional[List[str]] = None,
    seed: Optional[int] = None,
    warmup_requests: int = 0,
    timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    progress_every: int = 25,
    log=print,
) -> LoadRunResult:
    """Drive all prompts through a pool of ``concurrency`` workers.

    Warmup requests (re-using the first prompts) run before the timed
    window and are discarded; the caller should scrape engine metrics
    *after* warmup so counter deltas cover only the timed window.
    """
    _local = threading.local()
    _all_sessions: List[requests.Session] = []
    _lock = threading.Lock()
    done = [0]

    def _session() -> requests.Session:
        if not hasattr(_local, "s"):
            _local.s = requests.Session()
            with _lock:
                _all_sessions.append(_local.s)
        return _local.s

    def _one(args) -> RequestResult:
        idx, prompt = args
        res = stream_completion(
            base_url, model, prompt, idx, max_tokens,
            temperature=temperature, stop=stop, seed=seed,
            timeout_s=timeout_s, session=_session(),
        )
        with _lock:
            done[0] += 1
            if progress_every and done[0] % progress_every == 0:
                log("  [load] %d/%d requests done" % (done[0], len(prompts)))
        return res

    for w in range(warmup_requests):
        stream_completion(
            base_url, model, prompts[w % len(prompts)], -1, max_tokens,
            temperature=temperature, stop=stop, seed=seed, timeout_s=timeout_s,
        )

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(_one, enumerate(prompts)))
    wall = time.monotonic() - start

    for s in _all_sessions:
        s.close()
    return LoadRunResult(results=results, wall_time_s=wall)
