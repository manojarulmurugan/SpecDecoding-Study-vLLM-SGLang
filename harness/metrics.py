"""Metric definitions and aggregation (LITERATURE.md §7, HARNESS_SPEC.md §8).

Pure functions only -- everything here is unit-testable without a GPU.

Speculative-decoding stats come from vLLM's Prometheus counters
(vllm:spec_decode_num_drafts / _num_draft_tokens / _num_accepted_tokens,
defined in vllm/v1/spec_decode/metrics.py). Mean accepted length per
verification step:  tau = 1 + accepted_tokens/drafts  (the +1 is the bonus
token the target emits on every step), matching vLLM's own logged
"mean acceptance length".
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

# -- prometheus text parsing -------------------------------------------------

_METRIC_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([^\s]+)")


def parse_prometheus_text(text: str) -> Dict[str, float]:
    """Parse Prometheus exposition text into {metric_name: summed value}.

    Values are summed across label sets (we only ever run one model per
    server, so the sum is the per-model value).
    """
    out: Dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _METRIC_LINE.match(line)
        if not m:
            continue
        name, _, raw = m.groups()
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isnan(value):
            continue
        out[name] = out.get(name, 0.0) + value
    return out


def metric_value(metrics: Dict[str, float], base_name: str) -> Optional[float]:
    """Look up a counter tolerating the client-library ``_total`` suffix."""
    for candidate in (base_name, base_name + "_total"):
        if candidate in metrics:
            return metrics[candidate]
    return None


SPEC_COUNTERS = {
    "num_drafts": "vllm:spec_decode_num_drafts",
    "num_draft_tokens": "vllm:spec_decode_num_draft_tokens",
    "num_accepted_tokens": "vllm:spec_decode_num_accepted_tokens",
}


def spec_decode_stats(
    before: Dict[str, float], after: Dict[str, float]
) -> Optional[Dict[str, float]]:
    """Delta the spec-decode counters across a timed window.

    Returns None when the counters are absent (spec decoding off, or an
    engine that does not expose them).
    """
    deltas = {}
    for key, name in SPEC_COUNTERS.items():
        b, a = metric_value(before, name), metric_value(after, name)
        if a is None:
            return None
        deltas[key] = a - (b or 0.0)
    drafts = deltas["num_drafts"]
    draft_tokens = deltas["num_draft_tokens"]
    accepted = deltas["num_accepted_tokens"]
    return {
        "num_drafts": drafts,
        "num_draft_tokens": draft_tokens,
        "num_accepted_tokens": accepted,
        "acceptance_rate": (accepted / draft_tokens) if draft_tokens else None,
        "accepted_length_tau": (1.0 + accepted / drafts) if drafts else None,
    }


# -- latency aggregation -----------------------------------------------------

def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile, p in [0, 100]."""
    if not values:
        raise ValueError("percentile of empty sequence")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = (p / 100.0) * (len(xs) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return xs[lo]
    frac = rank - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def summarize_ms(values_s: Sequence[float]) -> Optional[Dict[str, float]]:
    """p50/p95/p99 summary of a list of second-valued latencies, in ms."""
    values = [v for v in values_s if v is not None]
    if not values:
        return None
    return {
        "p50": percentile(values, 50) * 1000.0,
        "p95": percentile(values, 95) * 1000.0,
        "p99": percentile(values, 99) * 1000.0,
        "mean": sum(values) / len(values) * 1000.0,
    }


def aggregate_run(
    request_results: Iterable[Any],
    wall_time_s: float,
    spec_stats: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Build the ``measured`` block of a result record (HARNESS_SPEC.md §4).

    ``request_results`` are load.RequestResult objects (duck-typed here so
    tests can pass simple namespaces).

    Notes on definitions:
    - throughput_tok_s: total completion tokens / wall time of the timed
      window. With spec decoding, completion tokens are already only the
      *kept* tokens, so under greedy decoding this equals goodput.
    - request_tok_s_mean: mean per-request completion tokens/sec over each
      request's full lifetime (send -> last token). This is the closest
      analog of SpecMQuant's per-question "generate_speed" and is the
      quantity the Block-0 speedup ratios are computed from.
    - ITL is measured between streamed chunks. With spec decoding a chunk
      may carry several tokens, so chunk ITL is a scheduler-step time, not
      a per-token time; per-token pacing is tokens/sec above.
    """
    results = list(request_results)
    ok = [r for r in results if not getattr(r, "error", None)]
    errors = [r for r in results if getattr(r, "error", None)]

    completion_tokens = sum(r.completion_tokens or 0 for r in ok)
    itl_all: List[float] = []
    for r in ok:
        itl_all.extend(getattr(r, "itl_s", []) or [])

    per_request_speed = [
        (r.completion_tokens / r.e2e_s)
        for r in ok
        if r.completion_tokens and r.e2e_s
    ]
    per_request_decode_speed = [
        (r.completion_tokens / r.decode_time_s)
        for r in ok
        if r.completion_tokens and r.decode_time_s
    ]

    measured: Dict[str, Any] = {
        "num_requests": len(results),
        "num_errors": len(errors),
        "wall_time_s": wall_time_s,
        "total_completion_tokens": completion_tokens,
        "ttft_ms": summarize_ms([r.ttft_s for r in ok]),
        "itl_ms": summarize_ms(itl_all),
        "e2e_latency_ms": summarize_ms([r.e2e_s for r in ok]),
        "throughput_tok_s": (completion_tokens / wall_time_s) if wall_time_s else None,
        "request_tok_s_mean": _mean(per_request_speed),
        "request_decode_tok_s_mean": _mean(per_request_decode_speed),
        "accepted_length_tau": None,
        "acceptance_rate": None,
        "spec_num_drafts": None,
        "accuracy": None,  # filled by the caller after correctness scoring
    }
    if spec_stats:
        measured["accepted_length_tau"] = spec_stats.get("accepted_length_tau")
        measured["acceptance_rate"] = spec_stats.get("acceptance_rate")
        measured["spec_num_drafts"] = spec_stats.get("num_drafts")
    return measured


def _mean(values: Sequence[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)
