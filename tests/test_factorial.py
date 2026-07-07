"""Verify the 2^3 factorial statistics against synthetic cubes with KNOWN
injected effects, before any real data exists.

Ground-truth model: goodput = 100 * 1.8^w * 1.05^k * 1.6^s (w,k,s in {0,1}),
optionally distorted with interference factors on specific corners.
"""
from __future__ import annotations

import math

import pytest
import yaml

from analysis.factorial import (
    ALL_CORNERS,
    analyze_cell,
    collect,
    corner_of,
    effects_from_cube,
    interference_gap,
    main as factorial_main,
    render_report,
)
from harness.results import ResultsStore

W_FACTOR, K_FACTOR, S_FACTOR = 1.8, 1.05, 1.6


def goodput(w, k, s, ws_interference=1.0, wks_interference=1.0, scale=1.0):
    value = 100.0 * (W_FACTOR ** w) * (K_FACTOR ** k) * (S_FACTOR ** s) * scale
    if w and s:
        value *= ws_interference
    if w and k and s:
        value *= wks_interference
    return value


def make_record(workload, conc, w, k, s, value, repeat=0, status="ok",
                block="core_factorial"):
    return {
        "run_id": "cube_%d%d%d_%s_c%d_r%d" % (w, k, s, workload, conc, repeat),
        "config": {
            "block": block, "workload": workload, "concurrency": conc,
            "repeat_idx": repeat,
            "factors": {
                "weight_quant": "w4a16" if w else "fp16",
                "kv_quant": "fp8" if k else "fp16",
                "spec_decode": "eagle3" if s else "none",
            },
        },
        "env": {}, "status": status,
        "measured": {"goodput_tok_s": value, "throughput_tok_s": value},
    }


def full_cube_records(workload="gsm8k", conc=8, repeats=(0,), **kwargs):
    records = []
    for w in (0, 1):
        for k in (0, 1):
            for s in (0, 1):
                for r in repeats:
                    records.append(make_record(
                        workload, conc, w, k, s,
                        goodput(w, k, s, **kwargs), repeat=r,
                    ))
    return records


# -- the contrasts themselves ---------------------------------------------------

def test_corner_coding():
    assert corner_of({"weight_quant": "fp16", "kv_quant": "fp16",
                      "spec_decode": "none"}) == (-1, -1, -1)
    assert corner_of({"weight_quant": "w4a16", "kv_quant": "fp8",
                      "spec_decode": "eagle3"}) == (1, 1, 1)
    # Block-0-style 'eagle' also counts as S on
    assert corner_of({"weight_quant": "fp16", "kv_quant": "fp16",
                      "spec_decode": "eagle"})[2] == 1


def test_clean_compounding_recovers_exact_effects_and_zero_gap():
    y = {}
    for c in ALL_CORNERS:
        w, k, s = [(v + 1) // 2 for v in c]
        y[c] = math.log(goodput(w, k, s))
    effects = effects_from_cube(y)
    assert effects["W"] == pytest.approx(math.log(W_FACTOR))
    assert effects["K"] == pytest.approx(math.log(K_FACTOR))
    assert effects["S"] == pytest.approx(math.log(S_FACTOR))
    for name in ("WK", "WS", "KS", "WKS"):
        assert abs(effects[name]) < 1e-12, "%s should vanish under clean compounding" % name
    gap = interference_gap(y)
    assert abs(gap["gap_log"]) < 1e-12
    assert gap["overestimate_factor"] == pytest.approx(1.0)
    assert gap["naive_speedup"] == pytest.approx(W_FACTOR * K_FACTOR * S_FACTOR)


def test_sub_additive_interference_detected_with_known_gap():
    y = {}
    for c in ALL_CORNERS:
        w, k, s = [(v + 1) // 2 for v in c]
        y[c] = math.log(goodput(w, k, s, ws_interference=0.85,
                                wks_interference=0.9))
    effects = effects_from_cube(y)
    # singles are untouched, so main effects keep their clean values...
    # (up to the interference terms' contribution to the contrasts)
    gap = interference_gap(y)
    # naive uses only baseline + singles (uncontaminated corners);
    # measured full stack carries both interference factors
    expected_gap = -(math.log(0.85) + math.log(0.9))
    assert gap["gap_log"] == pytest.approx(expected_gap)
    assert gap["overestimate_factor"] == pytest.approx(1 / (0.85 * 0.9))
    assert gap["gap_log"] > 0, "sub-additive case must have positive gap"
    # the pairwise WS contrast picks up the w&s interference: negative
    assert effects["WS"] < -0.05
    assert effects["WKS"] != 0


def test_super_additive_gap_is_negative():
    y = {}
    for c in ALL_CORNERS:
        w, k, s = [(v + 1) // 2 for v in c]
        y[c] = math.log(goodput(w, k, s, wks_interference=1.25))
    assert interference_gap(y)["gap_log"] == pytest.approx(-math.log(1.25))


def test_incomplete_cube_raises():
    y = {c: 1.0 for c in ALL_CORNERS[:-1]}
    with pytest.raises(ValueError, match="missing corners"):
        effects_from_cube(y)


# -- repeats and spread -----------------------------------------------------------

def test_spread_across_repeats_brackets_point_estimate():
    # repeat r inflates the W-on corners by (1 + 0.02r): the W effect (and
    # only its dependents) varies across repeats
    records = []
    for r in (0, 1, 2):
        for w in (0, 1):
            for k in (0, 1):
                for s in (0, 1):
                    scale = (1 + 0.02 * r) if w else 1.0
                    records.append(make_record(
                        "gsm8k", 8, w, k, s, goodput(w, k, s, scale=scale), repeat=r,
                    ))
    cell = analyze_cell(collect(records)[("gsm8k", 8)])
    assert cell["complete_repeats"] == [0, 1, 2]
    lo, hi = cell["spread_log"]["W"]
    assert lo == pytest.approx(math.log(W_FACTOR))
    assert hi == pytest.approx(math.log(W_FACTOR * 1.04))
    assert lo <= cell["effects_log"]["W"] <= hi
    # K unaffected by the repeat noise: zero-width spread at the true value
    klo, khi = cell["spread_log"]["K"]
    assert klo == pytest.approx(khi) == pytest.approx(math.log(K_FACTOR))
    assert cell["gap_spread_log"] is not None


def test_partial_repeats_fall_back_to_point_only():
    records = full_cube_records(repeats=(0, 1))
    # drop repeat 1 at one corner: only repeat 0 is complete
    records = [r for r in records
               if not (r["config"]["repeat_idx"] == 1
                       and r["config"]["factors"]["weight_quant"] == "w4a16"
                       and r["config"]["factors"]["kv_quant"] == "fp8"
                       and r["config"]["factors"]["spec_decode"] == "eagle3")]
    cell = analyze_cell(collect(records)[("gsm8k", 8)])
    assert cell["complete_repeats"] == [0]
    assert cell["spread_log"]["W"] is None


def test_missing_corner_marks_cell_incomplete():
    records = full_cube_records()
    records = [r for r in records
               if r["config"]["factors"]["spec_decode"] == "none"
               or r["config"]["factors"]["weight_quant"] == "fp16"]
    data = collect(records)
    assert analyze_cell(data[("gsm8k", 8)]) is None
    report = render_report({("gsm8k", 8): None}, "goodput_tok_s")
    assert "Incomplete cubes" in report


def test_collect_filters_status_and_block():
    records = full_cube_records()
    records[0]["status"] = "failed"
    records[1]["config"]["block"] = "repro"
    data = collect(records)
    cube = data[("gsm8k", 8)]
    assert len(cube) == 6  # two corners dropped
    # zero/None metric values are skipped, not log(0)-crashed
    records = full_cube_records()
    records[0]["measured"]["goodput_tok_s"] = 0
    records[0]["measured"]["throughput_tok_s"] = None
    assert len(collect(records)[("gsm8k", 8)]) == 7


def test_report_flags_spread_straddling_zero():
    # K effect noise straddles zero across repeats -> '~0?' marker
    records = []
    for r, k_scale in ((0, 0.99), (1, 1.01)):
        for w in (0, 1):
            for k in (0, 1):
                for s in (0, 1):
                    value = 100.0 * (W_FACTOR ** w) * ((1.0 * k_scale) ** k) * (S_FACTOR ** s)
                    records.append(make_record("gsm8k", 8, w, k, s, value, repeat=r))
    cell = analyze_cell(collect(records)[("gsm8k", 8)])
    report = render_report({("gsm8k", 8): cell}, "goodput_tok_s")
    assert "~0?" in report


def test_cli_end_to_end(tmp_path):
    store = ResultsStore(tmp_path / "results")
    for rec in full_cube_records(repeats=(0, 1, 2), ws_interference=0.85):
        store.write(rec)
    rc = factorial_main([str(tmp_path / "results")])
    assert rc == 0
    report = (tmp_path / "results" / "factorial_report.md").read_text()
    assert "gsm8k @ concurrency 8" in report
    assert "Interference gap" in report
    assert (tmp_path / "results" / "factorial_effects.json").exists()
    assert factorial_main([str(tmp_path / "empty")]) == 1
