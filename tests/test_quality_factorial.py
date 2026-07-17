"""Quality-side factorial: contrast math on synthetic cubes + invariants
that must hold on the committed phase3_results records."""
from __future__ import annotations

import pytest

from analysis.quality_factorial import (
    analyze, analyze_cell, collect_accuracy, load_records)
from pathlib import Path


def _rec(workload, conc, rep, w, k, s, acc, block="core_factorial",
         status="ok"):
    return {
        "run_id": "t", "status": status, "env": {},
        "config": {"block": block, "workload": workload, "concurrency": conc,
                   "repeat_idx": rep,
                   "factors": {"weight_quant": w, "kv_quant": k,
                               "spec_decode": s}},
        "measured": {"accuracy": acc},
    }


def _cube(workload="gsm8k", conc=1, w_cost=0.10):
    """Synthetic cube: base 0.8, W costs w_cost, K/S free, no interactions."""
    recs = []
    for w in ("fp16", "w4a16"):
        for k in ("fp16", "fp8"):
            for s in ("none", "eagle3"):
                acc = 0.8 - (w_cost if w == "w4a16" else 0.0)
                for rep in (0, 1):
                    recs.append(_rec(workload, conc, rep, w, k, s, acc))
    return recs


def test_pure_w_cost_recovered_exactly():
    cells = analyze(_cube(w_cost=0.10))
    r = cells["gsm8k@c1"]
    assert r["effects_pts"]["W"] == pytest.approx(-10.0)
    for other in ("K", "S", "WK", "WS", "KS", "WKS"):
        assert r["effects_pts"][other] == pytest.approx(0.0)
    assert r["compounding"]["excess_pts"] == pytest.approx(0.0)


def test_rag_and_non_factorial_records_skipped():
    recs = _cube() + [
        _rec("rag_shared_prefix", 1, 0, "fp16", "fp16", "none", None),
        _rec("gsm8k", 1, 0, "fp16", "fp16", "none", 0.5, block="k_stress"),
        _rec("gsm8k", 1, 0, "fp16", "fp16", "none", 0.1, status="failed"),
    ]
    cubes = collect_accuracy(recs)
    assert set(cubes) == {("gsm8k", 1)}
    assert len(cubes[("gsm8k", 1)]) == 8


def test_incomplete_cube_reports_error():
    recs = _cube()[:-2]  # drop one corner entirely
    result = analyze_cell(collect_accuracy(recs)[("gsm8k", 1)])
    assert result.get("error") == "incomplete cube"


def test_committed_phase3_records_match_known_findings():
    """Invariants of the real dataset (committed, so stable in CI):
    W costs points everywhere; WK is positive on HumanEval in all 4 cells;
    quality does not compound (|excess| <= 1 pt)."""
    results = analyze(load_records(Path("phase3_results")))
    cells = {k: v for k, v in results.items() if "error" not in v}
    assert len(cells) == 8, "gsm8k + humaneval x 4 concurrencies"
    for name, r in cells.items():
        assert r["effects_pts"]["W"] < -2.5, name
        assert abs(r["compounding"]["excess_pts"]) <= 1.0, name
        assert abs(r["effects_pts"]["S"]) <= 1.0, "greedy S is quality-free"
        if name.startswith("humaneval"):
            lo, hi = r["effects_pts_range"]["WK"]
            assert lo > 0, "WK positive, not noise, on %s" % name
