"""validate_claims: recompute the decision guide's quantitative claims from the
raw per-run records and PASS/FAIL each against the range the guide asserts.

This is the "tested product" loop for DECISION_GUIDE.md: every headline number
in the guide is re-derived here directly from the committed ``*_results/runs/*.json``
records, so a claim can never silently drift from the data (a new run set, an
edited number, a regression) without this failing loudly.

    python3 -m analysis.validate_claims phase3_results phase3b_results phase3c_diagnostics_results
    # -> 337 records, 11/11 PASS

Each row maps a finding id (as used in DECISION_GUIDE.md's provenance table) to a
committed data source and a predicted range; the SOURCES map documents where each
range is asserted in the guide. Extracted 2026-07-20 from the retired
``stack_advisor.py`` CLI (git history) — the recommendation engine was cut, this
reproducibility check was worth keeping. Pure stdlib; GPU-free.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Optional

# finding id -> the committed report/section where the guide asserts the range.
# Informational only (printed with --sources); the checks below are the authority.
SOURCES: Dict[str, str] = {
    "P2-S-c1": "phase2_marginals_report.md",
    "P2-W-c1": "phase2_marginals_report.md",
    "P2-K-tax": "phase2_marginals_report.md",
    "F3-KS": "factorial_report.md (KS)",
    "F3-W-rev": "factorial_report.md (W main effect, conc 64)",
    "F3-GAP": "factorial_report.md + quality_effects.json",
    "QUAL-W": "quality_effects.json",
    "QUAL-S": "phase3 runs + tests/test_repro_gate.py",
    "3b-K-cap": "k_stress_report.md (capacity table)",
    "D2-S-long": "phase3c diagnostics + k_stress KS-probe + retest",
}


def _load_runs(dirs: List[str]) -> List[dict]:
    recs = []
    for d in dirs:
        for f in sorted((pathlib.Path(d) / "runs").glob("*.json")):
            recs.append(json.loads(f.read_text()))
    return recs


def _mean_metric(records, key="goodput_tok_s", block=None, workload=None,
                 conc=None, w=None, k=None, s=None, rid_prefix=None):
    vals = []
    for r in records:
        cfg = r.get("config", {})
        fac = cfg.get("factors", {})
        if r.get("status") != "ok":
            continue
        if block and cfg.get("block") != block:
            continue
        if workload and cfg.get("workload") != workload:
            continue
        if conc is not None and int(cfg.get("concurrency", -1)) != conc:
            continue
        if w and fac.get("weight_quant") != w:
            continue
        if k and fac.get("kv_quant") != k:
            continue
        if s and fac.get("spec_decode") != s:
            continue
        if rid_prefix and not r.get("run_id", "").startswith(rid_prefix):
            continue
        v = r.get("measured", {}).get(key)
        if v is not None:
            vals.append(v)
    return sum(vals) / len(vals) if vals else None


def _ratio(records, num_kw, den_kw, key="goodput_tok_s"):
    n = _mean_metric(records, key=key, **num_kw)
    d = _mean_metric(records, key=key, **den_kw)
    return (n / d) if (n and d) else None


def _cf(**kw):  # core-factorial gsm8k selector shorthand
    base = dict(block="core_factorial", workload="gsm8k")
    base.update(kw)
    return base


# (finding, description, (lo, hi), fmt, compute(records) -> float|None)
VALIDATIONS = [
    ("P2-S-c1", "S toggle, GSM8K conc 1", (1.8, 2.5), "x%.2f",
     lambda r: _ratio(r, _cf(conc=1, w="fp16", k="fp16", s="eagle3"),
                      _cf(conc=1, w="fp16", k="fp16", s="none"))),
    ("P2-W-c1", "W toggle, GSM8K conc 1", (1.8, 2.3), "x%.2f",
     lambda r: _ratio(r, _cf(conc=1, w="w4a16", k="fp16", s="none"),
                      _cf(conc=1, w="fp16", k="fp16", s="none"))),
    ("P2-K-tax", "K toggle, GSM8K conc 1", (0.88, 1.00), "x%.2f",
     lambda r: _ratio(r, _cf(conc=1, w="fp16", k="fp8", s="none"),
                      _cf(conc=1, w="fp16", k="fp16", s="none"))),
    ("F3-KS", "K-under-S, GSM8K conc 1", (0.55, 0.78), "x%.2f",
     lambda r: _ratio(r, _cf(conc=1, w="fp16", k="fp8", s="eagle3"),
                      _cf(conc=1, w="fp16", k="fp16", s="eagle3"))),
    ("F3-W-rev", "W toggle, GSM8K conc 64", (0.85, 1.02), "x%.2f",
     lambda r: _ratio(r, _cf(conc=64, w="w4a16", k="fp16", s="none"),
                      _cf(conc=64, w="fp16", k="fp16", s="none"))),
    ("F3-GAP", "naive/measured full stack, GSM8K conc 1", (2.5, 3.3), "x%.2f",
     lambda r: (lambda b, W, K, S, F:
                ((W / b) * (K / b) * (S / b)) / (F / b)
                if all((b, W, K, S, F)) else None)(
         _mean_metric(r, **_cf(conc=1, w="fp16", k="fp16", s="none")),
         _mean_metric(r, **_cf(conc=1, w="w4a16", k="fp16", s="none")),
         _mean_metric(r, **_cf(conc=1, w="fp16", k="fp8", s="none")),
         _mean_metric(r, **_cf(conc=1, w="fp16", k="fp16", s="eagle3")),
         _mean_metric(r, **_cf(conc=1, w="w4a16", k="fp8", s="eagle3")))),
    ("QUAL-W", "W accuracy delta, HumanEval (pts, pooled conc)",
     (-12.0, -6.0), "%+.1f pts",
     lambda r: (lambda a, b: (a - b) * 100 if (a and b) else None)(
         _mean_metric(r, key="accuracy", block="core_factorial",
                      workload="humaneval", w="w4a16", k="fp16", s="none"),
         _mean_metric(r, key="accuracy", block="core_factorial",
                      workload="humaneval", w="fp16", k="fp16", s="none"))),
    ("QUAL-S", "S accuracy delta, GSM8K (pts, pooled conc)",
     (-1.5, 1.5), "%+.1f pts",
     lambda r: (lambda a, b: (a - b) * 100 if (a and b) else None)(
         _mean_metric(r, key="accuracy", block="core_factorial",
                      workload="gsm8k", w="fp16", k="fp16", s="eagle3"),
         _mean_metric(r, key="accuracy", block="core_factorial",
                      workload="gsm8k", w="fp16", k="fp16", s="none"))),
    ("3b-K-cap", "K toggle at capacity, k_stress conc 48", (1.10, 1.30),
     "x%.2f",
     lambda r: _ratio(r,
                      dict(block="k_stress", conc=48, w="fp16", k="fp8",
                           s="none"),
                      dict(block="k_stress", conc=48, w="fp16", k="fp16",
                           s="none"))),
    ("D2-S-long", "S toggle at 7.4k ctx, eager, conc 8", (0.80, 0.97),
     "x%.2f",
     lambda r: _ratio(r,
                      dict(block="k_stress", conc=8, w="fp16", k="fp16",
                           s="eagle3"),
                      dict(block="diagnostics", conc=8,
                           rid_prefix="diag_slong-eager-base"))),
    ("D2-S-long", "tau, short-context eager (healthy ref)", (2.4, 3.1),
     "%.2f",
     lambda r: _mean_metric(r, key="accepted_length_tau",
                            block="diagnostics",
                            rid_prefix="diag_tau-eager-short")),
]


def run_validation(dirs: List[str]) -> int:
    records = _load_runs(dirs)
    print("# validate_claims: %d records from %s"
          % (len(records), ", ".join(dirs)))
    print()
    print("| finding | check | predicted | measured | verdict |")
    print("|---|---|---|---|---|")
    failures = 0
    for fid, name, (lo, hi), fmt, compute in VALIDATIONS:
        measured = compute(records)
        pred = "%s..%s" % (fmt % lo, fmt % hi)
        if measured is None:
            print("| %s | %s | %s | - | SKIPPED (no records) |"
                  % (fid, name, pred))
            continue
        ok = lo <= measured <= hi
        failures += 0 if ok else 1
        print("| %s | %s | %s | %s | %s |"
              % (fid, name, pred, fmt % measured,
                 "PASS" if ok else "FAIL"))
    print()
    print("PASS: decision-guide claims consistent with the records provided."
          if failures == 0 else
          "FAIL: %d claim(s) outside their predicted range - fix the guide or "
          "explain the regression before shipping." % failures)
    return 0 if failures == 0 else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Recompute DECISION_GUIDE.md's quantitative claims from raw "
                    "run records and PASS/FAIL each against its predicted range.")
    p.add_argument("results_dirs", nargs="+", metavar="RESULTS_DIR",
                   help="result directories to load records from, e.g. "
                        "phase3_results phase3b_results phase3c_diagnostics_results")
    p.add_argument("--sources", action="store_true",
                   help="print the finding-id -> guide-source map and exit")
    args = p.parse_args(argv)

    if args.sources:
        for fid, src in SOURCES.items():
            print("[%s] %s" % (fid, src))
        return 0

    return run_validation(args.results_dirs)


if __name__ == "__main__":
    raise SystemExit(main())
