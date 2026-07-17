"""stack_advisor: every recommendation must be backed by a real finding id,
and the headline scenarios must produce the verdicts the data supports."""
from __future__ import annotations

from analysis.stack_advisor import FINDINGS, main, recommend


def _by_lever(recs):
    return {r.lever.split(" ")[0]: r for r in recs}


def test_every_finding_has_claim_and_source():
    for fid, f in FINDINGS.items():
        assert f["claim"] and f["source"], fid


def test_all_recommendations_carry_valid_provenance():
    scenarios = [
        dict(context_tokens=512, concurrency=1),
        dict(context_tokens=512, concurrency=64, workload="code"),
        dict(context_tokens=7400, concurrency=8),
        dict(context_tokens=7400, concurrency=48),
        dict(context_tokens=512, concurrency=1, quality_sensitive=True),
        dict(context_tokens=512, concurrency=1, native_fp8=True),
    ]
    for kw in scenarios:
        for r in recommend(**kw):
            assert r.findings, "%s in %s has no provenance" % (r.lever, kw)
            for fid in r.findings:
                assert fid in FINDINGS, fid


def test_long_context_turns_s_off():
    recs = _by_lever(recommend(context_tokens=7400, concurrency=1))
    assert recs["S"].verdict == "OFF"
    assert "D2-S-long" in recs["S"].findings


def test_batch1_short_context_is_s_territory():
    recs = _by_lever(recommend(context_tokens=512, concurrency=1))
    assert recs["S"].verdict == "ON"
    # and K must NOT ride along with S at batch 1 on emulating hardware
    assert recs["K"].verdict == "OFF"
    assert "F3-KS" in recs["K"].findings


def test_quality_sensitive_rejects_w_only():
    recs = _by_lever(recommend(context_tokens=512, concurrency=1,
                               quality_sensitive=True))
    assert recs["W"].verdict == "OFF"
    assert "QUAL-W" in recs["W"].findings
    assert recs["S"].verdict == "ON", "S is quality-free under greedy"


def test_capacity_pressure_turns_k_and_w_on():
    # 48 x (7400+256) / 142896 pool = 2.6x demand: deep capacity regime
    recs = _by_lever(recommend(context_tokens=7400, concurrency=48))
    assert recs["K"].verdict == "ON"
    assert "3b-K-cap" in recs["K"].findings
    assert recs["W"].verdict == "ON", "W as admission-ceiling lever"
    assert "3b-W-cap" in recs["W"].findings


def test_saturated_without_pressure_turns_w_off():
    # conc 64 but tiny contexts: compute-bound, no KV pressure
    recs = _by_lever(recommend(context_tokens=256, concurrency=64))
    assert recs["W"].verdict == "OFF"


def test_code_workload_keeps_s_on_when_saturated():
    recs = _by_lever(recommend(context_tokens=512, concurrency=64,
                               workload="code"))
    assert recs["S"].verdict == "ON"
    recs = _by_lever(recommend(context_tokens=512, concurrency=64,
                               workload="rag"))
    assert recs["S"].verdict == "OFF"


def test_native_fp8_flags_extrapolation():
    recs = _by_lever(recommend(context_tokens=512, concurrency=1,
                               native_fp8=True))
    assert recs["K"].verdict == "CONDITIONAL"
    assert any("EXTRAPOLATION" in c for c in recs["K"].caveats)


def test_cli_renders_with_stack_rules(capsys):
    assert main(["--context-tokens", "7400", "--concurrency", "8"]) == 0
    out = capsys.readouterr().out
    assert "S (EAGLE-3) -> OFF" in out
    assert "NEVER multiply" in out
    assert "Scope:" in out
    assert main(["--list-findings"]) == 0
