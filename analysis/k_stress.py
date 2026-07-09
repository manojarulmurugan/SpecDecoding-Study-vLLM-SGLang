"""K-stress addendum analysis: FP8-KV as capacity relief.

Companion to the Phase-3 K main-effect story. Phase 2 measured K ~flat in an
admission-limited regime (KV demand ~18% of pool); these cells create
capacity pressure (unique ~7.4k-token contexts) and ask whether FP16-KV and
FP8-KV diverge in what they can sustain.

Signals, per concurrency level (FP16-KV vs FP8-KV):
- goodput + the FP8/FP16 ratio (the user-visible outcome)
- emergent batch mean/max: FP16 should plateau at ~pool/context while FP8
  keeps tracking the offered concurrency
- kv_cache_usage max: ~1.0 = the ceiling was actually HIT (mechanism proof)
- num_preemptions: >0 = capacity thrash (eviction + recompute)
- queue depth and TTFT p95: where the pain lands for users

With --pool-tokens N (read "GPU KV cache size: N tokens" from the FP16
server log), the report adds the predicted FP16 plateau = pool / measured
mean context, next to the measured one.

Usage:
    python -m analysis.k_stress results [--pool-tokens N] [--out PATH]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from harness.results import ResultsStore


def collect(records: List[Dict[str, Any]]) -> Dict[Tuple[int, str], List[Dict[str, Any]]]:
    """-> {(concurrency, kv_quant): [measured, ...]} across repeats."""
    out: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        cfg = rec.get("config", {})
        if cfg.get("block") != "k_stress" or rec.get("status") != "ok":
            continue
        key = (int(cfg["concurrency"]), cfg["factors"]["kv_quant"])
        out[key].append(rec["measured"])
    return dict(out)


def _mean(values: List[Optional[float]]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _agg(measures: List[Dict[str, Any]], key: str, sub: Optional[str] = None) -> Optional[float]:
    values = []
    for m in measures:
        v = m.get(key)
        if sub is not None:
            v = (v or {}).get(sub)
        values.append(v)
    return _mean(values)


def render_report(
    cells: Dict[Tuple[int, str], List[Dict[str, Any]]],
    pool_tokens: Optional[int] = None,
) -> str:
    concurrencies = sorted({k[0] for k in cells})
    lines = ["# K-stress addendum: FP8-KV under capacity pressure", ""]
    lines.append(
        "| conc | goodput fp16 | goodput fp8 (ratio) | batch fp16 mean/max "
        "| batch fp8 mean/max | kv-usage max fp16/fp8 | preemptions fp16/fp8 "
        "| queue-p50 fp16/fp8 | ttft-p95 s fp16/fp8 |"
    )
    lines.append("|" + "---|" * 9)
    missing = []
    context_tokens = None
    for conc in concurrencies:
        fp16 = cells.get((conc, "fp16"), [])
        fp8 = cells.get((conc, "fp8"), [])
        if not fp16 or not fp8:
            missing.append("conc %d: missing %s" % (
                conc, "fp16" if not fp16 else "fp8"))
            continue
        g16, g8 = _agg(fp16, "goodput_tok_s"), _agg(fp8, "goodput_tok_s")
        prompt_mean = _agg(fp16, "prompt_tokens_mean")
        completion_mean = None
        n_req = _agg(fp16, "num_requests")
        total_completion = _agg(fp16, "total_completion_tokens")
        if n_req and total_completion:
            completion_mean = total_completion / n_req
        if prompt_mean:
            context_tokens = prompt_mean + (completion_mean or 0)

        def pair(key, sub=None, fmt="%.1f"):
            a, b = _agg(fp16, key, sub), _agg(fp8, key, sub)
            return "%s / %s" % (
                fmt % a if a is not None else "—",
                fmt % b if b is not None else "—",
            )

        lines.append(
            "| %d | %.0f | %.0f (%.2fx) | %s / %s | %s / %s | %s | %s | %s | %s |"
            % (
                conc, g16 or 0, g8 or 0,
                (g8 / g16) if g16 and g8 else 0,
                "%.1f" % (_agg(fp16, "emergent_batch_size", "mean") or 0),
                "%.0f" % (_agg(fp16, "emergent_batch_size", "max") or 0),
                "%.1f" % (_agg(fp8, "emergent_batch_size", "mean") or 0),
                "%.0f" % (_agg(fp8, "emergent_batch_size", "max") or 0),
                pair("kv_cache_usage", "max", "%.2f"),
                pair("num_preemptions", None, "%.0f"),
                pair("queue_depth", "p50", "%.1f"),
                # ttft stored in ms; report seconds
                "%.1f / %.1f" % (
                    (_agg(fp16, "ttft_ms", "p95") or 0) / 1000.0,
                    (_agg(fp8, "ttft_ms", "p95") or 0) / 1000.0,
                ),
            )
        )
    lines.append("")

    if pool_tokens and context_tokens:
        predicted = pool_tokens / context_tokens
        lines.append(
            "**Predicted FP16-KV plateau** (pool %d tokens / measured mean "
            "context %.0f tokens): **~%.0f concurrent requests**; FP8-KV "
            "doubles it (~%.0f). Compare against the measured batch columns."
            % (pool_tokens, context_tokens, predicted, predicted * 2)
        )
        lines.append("")
    verdicts = []
    for conc in concurrencies:
        fp16 = cells.get((conc, "fp16"), [])
        if not fp16:
            continue
        usage = _agg(fp16, "kv_cache_usage", "max")
        preempt = _agg(fp16, "num_preemptions")
        limited = (usage is not None and usage >= 0.97) or (preempt or 0) > 0
        verdicts.append(
            "- conc %d: FP16-KV %s (kv-usage max %s, preemptions %s)"
            % (conc,
               "CAPACITY-LIMITED" if limited else "not capacity-limited",
               "%.2f" % usage if usage is not None else "n/a",
               "%.0f" % preempt if preempt is not None else "n/a"))
    if verdicts:
        lines.append("## Regime per concurrency (FP16-KV)")
        lines += verdicts
        lines.append("")
    if missing:
        lines.append("## Missing cells")
        lines += ["- %s" % m for m in missing]
        lines.append("")
    if not cells:
        lines.append("NO DATA: no completed k_stress records found.")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir")
    parser.add_argument("--pool-tokens", type=int, default=None,
                        help='the "GPU KV cache size: N tokens" line from '
                             "the FP16-KV server log")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    store = ResultsStore(args.results_dir)
    cells = collect(store.load_all())
    report = render_report(cells, pool_tokens=args.pool_tokens)
    print(report)
    out = args.out or str(Path(args.results_dir) / "k_stress_report.md")
    Path(out).write_text(report)
    print("[k_stress] report written to %s" % out)
    return 0 if cells else 1


if __name__ == "__main__":
    sys.exit(main())
