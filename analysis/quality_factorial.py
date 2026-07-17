"""Quality-side 2^3 factorial: accuracy effects, computed like the speed side.

Same replicated-contrast machinery as analysis/factorial.py, but on RAW
accuracy (percentage points, additive) instead of log(goodput)
(multiplicative): a quantization step either costs correctness points or it
doesn't — there is no compounding story to test in logs. Effects are the
standard 2^3 contrasts x100 (points); each complete repeat yields its own
estimate, and an effect whose per-repeat spread straddles 0 is noise.

The companion claim to the speed side's interference finding: speed
interferes (gap x1.3-3.0), quality does NOT compound — the full stack's
accuracy delta ~= the sum of the mains, with all interaction terms ~0.

Only GSM8K (exact match) and HumanEval (unit tests) carry accuracy;
rag_shared_prefix has no ground truth by design and is skipped.

Usage:  python3 -m analysis.quality_factorial phase3_results
Writes <results_dir>/quality_effects.json and prints a markdown report.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from analysis.factorial import Corner, corner_of, effects_from_cube

EFFECT_NAMES = ("W", "K", "S", "WK", "WS", "KS", "WKS")


def collect_accuracy(
    records: List[Dict[str, Any]],
) -> Dict[Tuple[str, int], Dict[Corner, Dict[int, float]]]:
    """-> {(workload, concurrency): {corner: {repeat_idx: accuracy}}}"""
    out: Dict[Tuple[str, int], Dict[Corner, Dict[int, float]]] = (
        defaultdict(lambda: defaultdict(dict)))
    for rec in records:
        cfg = rec.get("config", {})
        if cfg.get("block") != "core_factorial" or rec.get("status") != "ok":
            continue
        acc = rec.get("measured", {}).get("accuracy")
        if acc is None:  # rag_shared_prefix: no ground truth by design
            continue
        key = (cfg["workload"], int(cfg["concurrency"]))
        out[key][corner_of(cfg["factors"])][int(cfg.get("repeat_idx", 0))] = acc
    return {k: dict(v) for k, v in out.items()}


def analyze_cell(cube: Dict[Corner, Dict[int, float]]) -> Dict[str, Any]:
    """Effects in accuracy POINTS (x100), pooled + per-complete-repeat."""
    complete = sorted(set.intersection(
        *[set(reps) for reps in cube.values()])) if len(cube) == 8 else []
    if not complete:
        return {"error": "incomplete cube", "corners": len(cube)}
    mean_y = {c: sum(r.values()) / len(r) for c, r in cube.items()}
    point = {k: v * 100 for k, v in effects_from_cube(mean_y).items()}
    per_rep = {
        k: [effects_from_cube({c: cube[c][r] for c in cube})[k] * 100
            for r in complete]
        for k in EFFECT_NAMES}
    base = mean_y[corner_of(
        {"weight_quant": "fp16", "kv_quant": "fp16", "spec_decode": "none"})]
    full = mean_y[corner_of(
        {"weight_quant": "w4a16", "kv_quant": "fp8", "spec_decode": "eagle3"})]
    # Additive-compounding check: does the full stack cost more points than
    # the three mains would predict? (speed's interference-gap analog)
    naive_delta = point["W"] + point["K"] + point["S"]
    measured_delta = (full - base) * 100
    return {
        "complete_repeats": complete,
        "effects_pts": {k: point[k] for k in EFFECT_NAMES},
        "effects_pts_range": {k: [min(v), max(v)] for k, v in per_rep.items()},
        "baseline_acc": base,
        "fullstack_acc": full,
        "compounding": {
            "naive_delta_pts": naive_delta,
            "measured_delta_pts": measured_delta,
            "excess_pts": measured_delta - naive_delta,
        },
    }


def analyze(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "%s@c%d" % key: analyze_cell(cube)
        for key, cube in sorted(collect_accuracy(records).items())
    }


def render_report(results: Dict[str, Any]) -> str:
    lines = ["# Quality-side 2^3 factorial: accuracy effects (points)", ""]
    lines.append(
        "Positive = that factor RAISES accuracy. '~0?' = per-repeat spread "
        "straddles zero (noise). `excess` = full-stack delta minus the sum "
        "of the three mains: ~0 everywhere means quality costs do NOT "
        "compound, the additive analog of the speed side's interference "
        "test.")
    lines.append("")
    for cell, r in results.items():
        lines.append("## %s" % cell)
        if "error" in r:
            lines.append("(%s)" % r["error"])
            continue
        lines.append("")
        lines.append("| effect | points | per-repeat range |")
        lines.append("|---|---|---|")
        for k in EFFECT_NAMES:
            lo, hi = r["effects_pts_range"][k]
            flag = " ~0?" if lo < 0 < hi else ""
            lines.append("| %s | %+.1f%s | [%+.1f..%+.1f] |"
                         % (k, r["effects_pts"][k], flag, lo, hi))
        c = r["compounding"]
        lines.append("")
        lines.append(
            "baseline %.1f%% -> full stack %.1f%% | naive (sum of mains) "
            "%+.1f pts, measured %+.1f pts, excess %+.1f pts"
            % (r["baseline_acc"] * 100, r["fullstack_acc"] * 100,
               c["naive_delta_pts"], c["measured_delta_pts"],
               c["excess_pts"]))
        lines.append("")
    return "\n".join(lines)


def load_records(results_dir: pathlib.Path) -> List[Dict[str, Any]]:
    return [json.loads(f.read_text())
            for f in sorted((results_dir / "runs").glob("*.json"))]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results_dir", type=pathlib.Path)
    args = p.parse_args(argv)
    results = analyze(load_records(args.results_dir))
    if not results:
        print("no core_factorial records with accuracy found")
        return 1
    out = args.results_dir / "quality_effects.json"
    out.write_text(json.dumps(results, indent=1, sort_keys=True))
    print(render_report(results))
    print("[quality_factorial] effects written to %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
