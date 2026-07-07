from __future__ import annotations

import pytest
import yaml

from analysis.repro_gate import collect, evaluate, main as gate_main, render_report
from harness.results import ResultsStore

TARGETS = {
    "reference": {
        "fp16_eagle_rel_speedup": 2.3,
        "w4a16_eagle_rel_speedup": 1.3,
        "w4a16_quant_only_speedup": 2.1,
    },
    "gate": {
        "min_fp16_rel_speedup": 1.2,
        "min_relative_erosion": 0.15,
        "magnitude_rel_tolerance": 0.30,
    },
}


def _record(workload, weight_quant, spec, speed, tau=None, acc=0.8):
    return {
        "run_id": "repro_%s_%s_%s" % (weight_quant, spec, workload),
        "config": {
            "block": "repro",
            "workload": workload,
            "factors": {
                "weight_quant": weight_quant, "kv_quant": "fp16",
                "spec_decode": spec,
            },
        },
        "env": {},
        "measured": {
            "request_tok_s_mean": speed,
            "accepted_length_tau": tau,
            "accuracy": acc,
        },
        "status": "ok",
    }


def _records_eroding():
    # fp16: 30 -> 66 tok/s (2.2x); w4a16: 60 -> 78 (1.3x): erosion reproduced
    return [
        _record("gsm8k", "fp16", "none", 30.0),
        _record("gsm8k", "fp16", "eagle", 66.0, tau=3.1),
        _record("gsm8k", "w4a16", "none", 60.0),
        _record("gsm8k", "w4a16", "eagle", 78.0, tau=3.0),
    ]


def test_gate_passes_on_erosion():
    result = evaluate(collect(_records_eroding()), TARGETS)
    assert result["failures"] == []
    row = result["rows"][0]
    assert row["rel_speedup_fp16"] == pytest.approx(2.2)
    assert row["rel_speedup_w4a16"] == pytest.approx(1.3)
    assert row["quant_only_speedup"] == pytest.approx(2.0)
    report = render_report(result)
    assert "GATE: PASS" in report


def test_gate_fails_without_erosion():
    records = _records_eroding()
    records[3]["measured"]["request_tok_s_mean"] = 130.0  # w4a16 eagle 2.17x
    result = evaluate(collect(records), TARGETS)
    assert any("NOT reproduced" in f for f in result["failures"])
    assert "GATE: FAIL" in render_report(result)


def test_gate_fails_when_eagle_never_helps():
    records = _records_eroding()
    records[1]["measured"]["request_tok_s_mean"] = 33.0  # fp16 eagle only 1.1x
    result = evaluate(collect(records), TARGETS)
    assert any("isn't paying off" in f for f in result["failures"])


def test_gate_fails_on_missing_cells():
    result = evaluate(collect(_records_eroding()[:3]), TARGETS)
    assert any("missing cells" in f for f in result["failures"])


def test_magnitude_deviation_warns_not_fails():
    records = _records_eroding()
    # fp16 eagle only 1.6x: erosion vs w4a16's 1.3x still >15% (direction ok)
    # but magnitude deviates >30% from the 2.3x reference -> warning only
    records[1]["measured"]["request_tok_s_mean"] = 48.0
    result = evaluate(collect(records), TARGETS)
    assert result["failures"] == []
    assert any("fp16_eagle_rel_speedup" in w for w in result["warnings"])


def test_accuracy_drift_under_greedy_spec_warns():
    records = _records_eroding()
    records[1]["measured"]["accuracy"] = 0.5  # eagle cell accuracy moved a lot
    result = evaluate(collect(records), TARGETS)
    assert any("output-preserving" in w for w in result["warnings"])


def test_non_repro_and_failed_records_ignored():
    records = _records_eroding()
    records[0]["config"]["block"] = "core_factorial"
    records[1]["status"] = "failed"
    table = collect(records)
    assert "fp16|none" not in table.get("gsm8k", {})
    assert "fp16|eagle" not in table.get("gsm8k", {})


def test_gate_main_cli(tmp_path):
    store = ResultsStore(tmp_path / "results")
    for rec in _records_eroding():
        store.write(rec)
    targets_path = tmp_path / "targets.yaml"
    targets_path.write_text(yaml.safe_dump(TARGETS))
    rc = gate_main([str(tmp_path / "results"), "--targets", str(targets_path)])
    assert rc == 0
    assert (tmp_path / "results" / "repro_gate_report.md").exists()

    # empty store -> nonzero exit
    empty = tmp_path / "empty"
    rc = gate_main([str(empty), "--targets", str(targets_path)])
    assert rc == 1
