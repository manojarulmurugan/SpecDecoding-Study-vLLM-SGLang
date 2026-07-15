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
    """Capacity cells: {(concurrency, kv_quant): [measured, ...]} across
    repeats. Spec-on cells belong to the KS probe (collect_ks_probe), not
    here -- mixing them would contaminate the capacity comparison."""
    out: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        cfg = rec.get("config", {})
        if cfg.get("block") != "k_stress" or rec.get("status") != "ok":
            continue
        if cfg["factors"].get("spec_decode", "none") != "none":
            continue
        if cfg["factors"].get("weight_quant", "fp16") != "fp16":
            continue  # W corners get their own comparison, not this table
        key = (int(cfg["concurrency"]), cfg["factors"]["kv_quant"])
        out[key].append(rec["measured"])
    return dict(out)


def collect_w_corners(records: List[Dict[str, Any]]) -> Dict[Tuple[int, str], List[Dict[str, Any]]]:
    """AWQ capacity cells: {(concurrency, kv_quant): [measured, ...]}."""
    out: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        cfg = rec.get("config", {})
        if cfg.get("block") != "k_stress" or rec.get("status") != "ok":
            continue
        if cfg["factors"].get("spec_decode", "none") != "none":
            continue
        if cfg["factors"].get("weight_quant") != "w4a16":
            continue
        out[(int(cfg["concurrency"]), cfg["factors"]["kv_quant"])].append(rec["measured"])
    return dict(out)


def collect_ks_probe(records: List[Dict[str, Any]]) -> Dict[Tuple[int, str], List[Dict[str, Any]]]:
    """Long-context K-under-S cells: {(concurrency, kv_quant): [measured]}."""
    out: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        cfg = rec.get("config", {})
        if cfg.get("block") != "k_stress" or rec.get("status") != "ok":
            continue
        if cfg["factors"].get("spec_decode", "none") == "none":
            continue
        out[(int(cfg["concurrency"]), cfg["factors"]["kv_quant"])].append(rec["measured"])
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


def render_w_corner_section(
    w_cells: Dict[Tuple[int, str], List[Dict[str, Any]]]
) -> List[str]:
    if not w_cells:
        return []
    lines = ["## W capacity channel (AWQ weights, no spec)", ""]
    lines.append(
        "AWQ frees ~10.4GB of weight memory -> larger KV pool -> higher "
        "sustainable concurrency at the SAME kv dtype. Compare batch/kv-usage "
        "against the FP16-weights table above."
    )
    lines.append("")
    lines.append("| conc | goodput fp16kv | goodput fp8kv | batch fp16kv mean/max "
                 "| batch fp8kv mean/max | kv-usage max | preemptions |")
    lines.append("|" + "---|" * 7)
    for conc in sorted({k[0] for k in w_cells}):
        f16 = w_cells.get((conc, "fp16"), [])
        f8 = w_cells.get((conc, "fp8"), [])
        if not f16 or not f8:
            continue
        lines.append(
            "| %d | %.0f | %.0f | %.1f / %.0f | %.1f / %.0f | %s / %s | %s / %s |"
            % (
                conc,
                _agg(f16, "goodput_tok_s") or 0, _agg(f8, "goodput_tok_s") or 0,
                _agg(f16, "emergent_batch_size", "mean") or 0,
                _agg(f16, "emergent_batch_size", "max") or 0,
                _agg(f8, "emergent_batch_size", "mean") or 0,
                _agg(f8, "emergent_batch_size", "max") or 0,
                "%.2f" % (_agg(f16, "kv_cache_usage", "max") or 0),
                "%.2f" % (_agg(f8, "kv_cache_usage", "max") or 0),
                "%.0f" % (_agg(f16, "num_preemptions") or 0),
                "%.0f" % (_agg(f8, "num_preemptions") or 0),
            )
        )
    lines.append("")
    return lines


def render_ks_probe_section(
    probe: Dict[Tuple[int, str], List[Dict[str, Any]]],
    capacity: Dict[Tuple[int, str], List[Dict[str, Any]]],
) -> List[str]:
    if not probe:
        return []
    lines = ["## KS long-context probe (EAGLE-3 on, ~7.4k-token contexts)", ""]
    lines.append(
        "K-toggle-under-S at long context, same hardware/kernels as the "
        "factorial's short-context KS. Short-context reference (factorial "
        "@ c1, ~1k contexts): K-under-S ~x0.63, K-solo x0.94, tau invariant. "
        "If the long-context ratio here is materially higher than x0.63, "
        "context length buys back bandwidth credit; if it matches, the "
        "emulation tax dominates regardless of context."
    )
    lines.append("")
    lines.append(
        "**EAGER-MODE CAVEAT (2026-07-15 crash fix):** these probe cells "
        "run with --enforce-eager (vLLM 0.24.0's compiled eagle_head "
        "kernels device-assert at this context length; PREREQ 2026-07-15), "
        "while every other cell in the project runs compiled. Within-row "
        "ratios (fp8kv/fp16kv, both eager) and tau are clean; absolute "
        "probe tok/s vs any compiled cell -- including the K-solo column "
        "here and the factorial's short-context KS goodput -- is NOT "
        "like-for-like. Compare RATIOS across regimes, never raw goodput."
    )
    lines.append("")
    lines.append("| conc | goodput S+fp16kv | S+fp8kv (K-under-S ratio) | "
                 "tau fp16kv/fp8kv | K-solo ratio same conc (long ctx) |")
    lines.append("|" + "---|" * 5)
    for conc in sorted({k[0] for k in probe}):
        f16 = probe.get((conc, "fp16"), [])
        f8 = probe.get((conc, "fp8"), [])
        if not f16 or not f8:
            continue
        g16, g8 = _agg(f16, "goodput_tok_s"), _agg(f8, "goodput_tok_s")
        solo16 = _agg(capacity.get((conc, "fp16"), []), "goodput_tok_s")
        solo8 = _agg(capacity.get((conc, "fp8"), []), "goodput_tok_s")
        solo = ("x%.2f" % (solo8 / solo16)) if solo16 and solo8 else "—"
        lines.append(
            "| %d | %.0f | %.0f (x%.2f) | %.2f / %.2f | %s |"
            % (
                conc, g16 or 0, g8 or 0,
                (g8 / g16) if g16 and g8 else 0,
                _agg(f16, "accepted_length_tau") or 0,
                _agg(f8, "accepted_length_tau") or 0,
                solo,
            )
        )
    lines.append("")
    return lines


def render_report(
    cells: Dict[Tuple[int, str], List[Dict[str, Any]]],
    pool_tokens: Optional[int] = None,
    w_cells: Optional[Dict[Tuple[int, str], List[Dict[str, Any]]]] = None,
    probe_cells: Optional[Dict[Tuple[int, str], List[Dict[str, Any]]]] = None,
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
    lines += render_w_corner_section(w_cells or {})
    lines += render_ks_probe_section(probe_cells or {}, cells)
    if missing:
        lines.append("## Missing cells")
        lines += ["- %s" % m for m in missing]
        lines.append("")
    if not cells and not (w_cells or probe_cells):
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
    records = store.load_all()
    cells = collect(records)
    report = render_report(
        cells, pool_tokens=args.pool_tokens,
        w_cells=collect_w_corners(records),
        probe_cells=collect_ks_probe(records),
    )
    print(report)
    out = args.out or str(Path(args.results_dir) / "k_stress_report.md")
    Path(out).write_text(report)
    print("[k_stress] report written to %s" % out)
    return 0 if (cells or collect_w_corners(records) or collect_ks_probe(records)) else 1


if __name__ == "__main__":
    sys.exit(main())
