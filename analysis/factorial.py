"""Core 2^3 factorial analysis (HARNESS_SPEC.md §9, as revised).

Everything is computed on **log(goodput)**: "clean compounding" is a
multiplicative claim, so multiplicative compounding = additive in logs, and
the standard factorial contrasts in log space are the actual test of it.

Definitions (per workload, per concurrency level -- ~12 mini-factorials,
protected against cherry-picking by the pre-stated hypothesis directions in
EXPERIMENT_MATRIX §5):

- Corner coding: x_W, x_K, x_S in {-1,+1}; y = log(goodput).
- Effect_j = (1/4) * sum_i x_{j,i} * y_i over the 8 corner means -- the
  standard 2^3 contrast (difference of means at +1 vs -1). exp(effect) is
  the multiplicative speedup factor attributable to that term.
- Interference gap (the headline quantity): naive compounding predicts
  full-stack log-gain = sum of the three single-factor log-gains over
  baseline. gap = naive - measured (both vs the baseline corner).
  gap > 0  => sub-additive (interference: the stack delivers less than the
  product of individual wins); gap ~ 0 => clean compounding;
  gap < 0  => super-additive. exp(gap) = the factor by which the naive
  product overestimates the measured combined speedup.
  (Note: the gap is a corner-difference quantity; algebraically it equals
  2*(e_WK + e_WS + e_KS) - 2*e_WKS in effect terms, so it aggregates ALL
  interaction structure, not just the three-way contrast, which is also
  reported separately.)

- Spread: the point estimate uses per-corner means across repeats; each
  repeat index with a complete cube also yields its own effect estimate,
  and the report shows min..max across those per-repeat estimates. An
  effect whose spread straddles 0 is not distinguishable from noise here.

Usage:
    python -m analysis.factorial results [--metric goodput_tok_s] [--out PATH]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from harness.results import ResultsStore

Corner = Tuple[int, int, int]  # (x_W, x_K, x_S) each in {-1, +1}

EFFECT_NAMES = ["W", "K", "S", "WK", "WS", "KS", "WKS"]
ALL_CORNERS: List[Corner] = [
    (w, k, s) for w in (-1, 1) for k in (-1, 1) for s in (-1, 1)
]
BASELINE: Corner = (-1, -1, -1)
FULL_STACK: Corner = (1, 1, 1)
SINGLES: Dict[str, Corner] = {"W": (1, -1, -1), "K": (-1, 1, -1), "S": (-1, -1, 1)}


def corner_of(factors: Dict[str, str]) -> Corner:
    return (
        1 if factors.get("weight_quant") != "fp16" else -1,
        1 if factors.get("kv_quant") != "fp16" else -1,
        1 if factors.get("spec_decode", "none") != "none" else -1,
    )


def _sign(corner: Corner, name: str) -> int:
    w, k, s = corner
    lookup = {"W": w, "K": k, "S": s}
    prod = 1
    for ch in name:
        prod *= lookup[ch]
    return prod


def effects_from_cube(y: Dict[Corner, float]) -> Dict[str, float]:
    """Standard 2^3 contrasts from a complete cube of log-metric values."""
    missing = [c for c in ALL_CORNERS if c not in y]
    if missing:
        raise ValueError("incomplete cube, missing corners: %s" % missing)
    out = {"mean": sum(y.values()) / 8.0}
    for name in EFFECT_NAMES:
        out[name] = sum(_sign(c, name) * y[c] for c in ALL_CORNERS) / 4.0
    return out


def interference_gap(y: Dict[Corner, float]) -> Dict[str, float]:
    """naive (sum of single log-gains) minus measured full-stack log-gain."""
    base = y[BASELINE]
    naive = sum(y[SINGLES[n]] - base for n in ("W", "K", "S"))
    measured = y[FULL_STACK] - base
    gap = naive - measured
    return {
        "naive_log_gain": naive,
        "measured_log_gain": measured,
        "gap_log": gap,
        "naive_speedup": math.exp(naive),
        "measured_speedup": math.exp(measured),
        "overestimate_factor": math.exp(gap),
    }


# -- data plumbing -------------------------------------------------------------


def collect(
    records: List[Dict[str, Any]], metric: str = "goodput_tok_s"
) -> Dict[Tuple[str, int], Dict[Corner, Dict[int, float]]]:
    """-> {(workload, concurrency): {corner: {repeat_idx: log(metric)}}}"""
    out: Dict[Tuple[str, int], Dict[Corner, Dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for rec in records:
        cfg = rec.get("config", {})
        if cfg.get("block") not in ("core_factorial", "serving_baseline"):
            continue
        if rec.get("status") != "ok":
            continue
        value = rec.get("measured", {}).get(metric)
        if value is None:
            value = rec.get("measured", {}).get("throughput_tok_s")
        if not value or value <= 0:
            continue
        key = (cfg["workload"], int(cfg["concurrency"]))
        corner = corner_of(cfg["factors"])
        out[key][corner][int(cfg.get("repeat_idx", 0))] = math.log(value)
    return {k: dict(v) for k, v in out.items()}


def analyze_cell(
    cube: Dict[Corner, Dict[int, float]]
) -> Optional[Dict[str, Any]]:
    """One (workload, concurrency) mini-factorial. None if cube incomplete."""
    if any(c not in cube or not cube[c] for c in ALL_CORNERS):
        return None
    mean_y = {c: sum(reps.values()) / len(reps) for c, reps in cube.items()}
    point = effects_from_cube(mean_y)
    gap = interference_gap(mean_y)

    # per-repeat estimates: only repeat indices present at ALL corners
    complete_repeats = sorted(
        set.intersection(*(set(cube[c]) for c in ALL_CORNERS))
    )
    per_repeat = {
        r: effects_from_cube({c: cube[c][r] for c in ALL_CORNERS})
        for r in complete_repeats
    }
    per_repeat_gaps = {
        r: interference_gap({c: cube[c][r] for c in ALL_CORNERS})["gap_log"]
        for r in complete_repeats
    }
    spread = {}
    for name in EFFECT_NAMES:
        values = [per_repeat[r][name] for r in complete_repeats]
        spread[name] = (min(values), max(values)) if len(values) >= 2 else None
    gap_values = list(per_repeat_gaps.values())
    return {
        "effects_log": {n: point[n] for n in EFFECT_NAMES},
        "effects_ratio": {n: math.exp(point[n]) for n in EFFECT_NAMES},
        "spread_log": spread,
        "gap": gap,
        "gap_spread_log": (min(gap_values), max(gap_values))
        if len(gap_values) >= 2 else None,
        "complete_repeats": complete_repeats,
        "repeats_per_corner": {str(c): sorted(cube[c]) for c in ALL_CORNERS},
    }


# -- reporting -----------------------------------------------------------------


def _fmt_effect(value: float, spread: Optional[Tuple[float, float]]) -> str:
    text = "%+.3f (x%.2f)" % (value, math.exp(value))
    if spread:
        text += " [%+.3f..%+.3f]" % spread
        if spread[0] <= 0 <= spread[1]:
            text += " ~0?"
    return text


def render_report(
    cells: Dict[Tuple[str, int], Optional[Dict[str, Any]]], metric: str
) -> str:
    lines = ["# Core 2^3 factorial: log-space effects on %s" % metric, ""]
    lines.append(
        "Effect columns: log-effect (multiplicative factor) "
        "[min..max across complete repeats]; '~0?' = spread straddles zero. "
        "Gap > 0 = sub-additive interference (see analysis/factorial.py docstring)."
    )
    lines.append("")
    incomplete = []
    for (workload, conc) in sorted(cells):
        cell = cells[(workload, conc)]
        if cell is None:
            incomplete.append("%s @ conc=%d" % (workload, conc))
            continue
        lines.append("## %s @ concurrency %d" % (workload, conc))
        lines.append("")
        lines.append("| effect | estimate |")
        lines.append("|---|---|")
        for name in EFFECT_NAMES:
            lines.append(
                "| %s | %s |"
                % (name, _fmt_effect(cell["effects_log"][name],
                                     cell["spread_log"][name]))
            )
        gap = cell["gap"]
        gap_line = (
            "**Interference gap**: naive x%.2f vs measured x%.2f -> "
            "gap %+.3f log (naive overestimates by x%.2f)"
            % (gap["naive_speedup"], gap["measured_speedup"],
               gap["gap_log"], gap["overestimate_factor"])
        )
        if cell["gap_spread_log"]:
            gap_line += " [%+.3f..%+.3f across repeats]" % cell["gap_spread_log"]
        lines.append("")
        lines.append(gap_line)
        lines.append(
            "Complete repeats: %s" % (cell["complete_repeats"] or "none (point only)")
        )
        lines.append("")
    if incomplete:
        lines.append("## Incomplete cubes (skipped)")
        lines += ["- %s" % i for i in incomplete]
        lines.append("")
    if not cells:
        lines.append("NO DATA: no completed core_factorial records found.")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir")
    parser.add_argument("--metric", default="goodput_tok_s")
    parser.add_argument("--out", default=None)
    parser.add_argument("--json-out", default=None,
                        help="also dump machine-readable effects JSON")
    args = parser.parse_args(argv)

    store = ResultsStore(args.results_dir)
    data = collect(store.load_all(), metric=args.metric)
    cells = {key: analyze_cell(cube) for key, cube in data.items()}
    report = render_report(cells, args.metric)
    print(report)
    out = args.out or str(Path(args.results_dir) / "factorial_report.md")
    Path(out).write_text(report)
    json_out = args.json_out or str(Path(args.results_dir) / "factorial_effects.json")
    Path(json_out).write_text(json.dumps(
        {"%s@c%d" % k: v for k, v in cells.items()}, indent=2, sort_keys=True
    ))
    print("[factorial] report -> %s ; effects json -> %s" % (out, json_out))
    complete = [c for c in cells.values() if c]
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
