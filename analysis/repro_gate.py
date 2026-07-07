"""Block-0 reproduction-gate analysis (EXPERIMENT_MATRIX.md §7 Layer 1).

Reads the repro-block result records and answers: does EAGLE's relative
speedup shrink under W4A16 the way SpecMQuant reports?

Gate design (see configs/repro/reference_targets.yaml for provenance):
- PRIMARY (pass/fail): the *direction* of the interaction. For each
  workload, rel_speedup = (spec-on request tok/s) / (spec-off request tok/s)
  within the same weight precision. Gate passes when
      rel_speedup(fp16) > min_fp16_rel      (EAGLE actually helps at FP16)
      rel_speedup(w4a16) < rel_speedup(fp16) * (1 - min_erosion)
- SECONDARY (warn only): magnitude vs the paper's Figure-1(a) numbers.
  SpecMQuant's engine is bespoke C/CUDA with tree drafting (size 60);
  vLLM's EAGLE is chain drafting in a production engine, so magnitudes are
  expected to differ. Cross-engine magnitude mismatch is a caveat to
  document, not a harness bug -- hence warn, not fail.

Usage:
    python -m analysis.repro_gate results --targets configs/repro/reference_targets.yaml
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from harness.results import ResultsStore

SPEED_KEY = "request_tok_s_mean"  # per-request tokens/s, SpecMQuant's ratio basis


def collect(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """-> {workload: {"fp16|none": measured, "fp16|eagle": ..., ...}}"""
    table: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for rec in records:
        cfg = rec.get("config", {})
        if cfg.get("block") != "repro":
            continue
        if rec.get("status") != "ok":
            continue
        f = cfg["factors"]
        key = "%s|%s" % (f["weight_quant"], f["spec_decode"])
        table[cfg["workload"]][key] = rec["measured"]
    return dict(table)


def evaluate(
    table: Dict[str, Dict[str, Dict[str, Any]]],
    targets: Dict[str, Any],
) -> Dict[str, Any]:
    gate = targets.get("gate", {})
    min_fp16_rel = float(gate.get("min_fp16_rel_speedup", 1.2))
    min_erosion = float(gate.get("min_relative_erosion", 0.15))
    magnitude_tol = float(gate.get("magnitude_rel_tolerance", 0.30))
    refs = targets.get("reference", {})

    rows = []
    failures = []
    warnings = []
    for workload in sorted(table):
        cells = table[workload]
        needed = ["fp16|none", "fp16|eagle", "w4a16|none", "w4a16|eagle"]
        missing = [k for k in needed if k not in cells]
        if missing:
            failures.append("%s: missing cells %s" % (workload, missing))
            continue
        speed = {k: cells[k][SPEED_KEY] for k in needed}
        rel_fp16 = speed["fp16|eagle"] / speed["fp16|none"]
        rel_w4a16 = speed["w4a16|eagle"] / speed["w4a16|none"]
        quant_only = speed["w4a16|none"] / speed["fp16|none"]
        row = {
            "workload": workload,
            "rel_speedup_fp16": rel_fp16,
            "rel_speedup_w4a16": rel_w4a16,
            "quant_only_speedup": quant_only,
            "tau_fp16": cells["fp16|eagle"].get("accepted_length_tau"),
            "tau_w4a16": cells["w4a16|eagle"].get("accepted_length_tau"),
            "acc_fp16_base": cells["fp16|none"].get("accuracy"),
            "acc_fp16_eagle": cells["fp16|eagle"].get("accuracy"),
            "acc_w4a16_base": cells["w4a16|none"].get("accuracy"),
            "acc_w4a16_eagle": cells["w4a16|eagle"].get("accuracy"),
        }
        rows.append(row)

        if rel_fp16 <= min_fp16_rel:
            failures.append(
                "%s: EAGLE rel speedup on FP16 is %.2fx (<= %.2fx) -- spec "
                "decoding isn't paying off at all; check the eagle config"
                % (workload, rel_fp16, min_fp16_rel)
            )
        if rel_w4a16 >= rel_fp16 * (1 - min_erosion):
            failures.append(
                "%s: no W4A16 erosion: rel speedup %.2fx (fp16) vs %.2fx "
                "(w4a16); SpecMQuant direction NOT reproduced"
                % (workload, rel_fp16, rel_w4a16)
            )
        for name, got in (("fp16_eagle_rel_speedup", rel_fp16),
                          ("w4a16_eagle_rel_speedup", rel_w4a16),
                          ("w4a16_quant_only_speedup", quant_only)):
            ref = refs.get(name)
            if ref and abs(got - ref) / ref > magnitude_tol:
                warnings.append(
                    "%s: %s = %.2fx vs reference %.2fx (>±%.0f%%) -- "
                    "expected across engines; document the gap"
                    % (workload, name, got, ref, magnitude_tol * 100)
                )
        # Greedy spec decoding is output-preserving: accuracy drift between
        # spec-on and spec-off at the same precision signals a harness bug.
        for prec in ("fp16", "w4a16"):
            a0 = row["acc_%s_base" % prec]
            a1 = row["acc_%s_eagle" % prec]
            if a0 is not None and a1 is not None and abs(a0 - a1) > 0.05:
                warnings.append(
                    "%s: accuracy moved %.3f -> %.3f when enabling EAGLE at "
                    "%s -- greedy spec decode should be output-preserving; "
                    "inspect outputs before trusting speed numbers"
                    % (workload, a0, a1, prec)
                )
    return {"rows": rows, "failures": failures, "warnings": warnings}


def render_report(result: Dict[str, Any]) -> str:
    lines = ["# Block-0 reproduction gate", ""]
    lines.append(
        "| workload | EAGLE rel speedup (FP16) | EAGLE rel speedup (W4A16) | "
        "quant-only speedup | tau FP16 | tau W4A16 |"
    )
    lines.append("|---|---|---|---|---|---|")
    for r in result["rows"]:
        lines.append(
            "| %s | %.2fx | %.2fx | %.2fx | %s | %s |"
            % (
                r["workload"], r["rel_speedup_fp16"], r["rel_speedup_w4a16"],
                r["quant_only_speedup"],
                _fmt(r["tau_fp16"]), _fmt(r["tau_w4a16"]),
            )
        )
    lines.append("")
    if result["failures"]:
        lines.append("## GATE: FAIL")
        lines += ["- %s" % f for f in result["failures"]]
    elif result["rows"]:
        lines.append("## GATE: PASS (direction reproduced)")
    else:
        lines.append("## GATE: NO DATA")
    if result["warnings"]:
        lines.append("")
        lines.append("## Warnings (document, don't panic)")
        lines += ["- %s" % w for w in result["warnings"]]
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Optional[float]) -> str:
    return "%.2f" % value if isinstance(value, float) else "n/a"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir")
    parser.add_argument("--targets", default="configs/repro/reference_targets.yaml")
    parser.add_argument("--out", default=None, help="write markdown report here")
    args = parser.parse_args(argv)

    with open(args.targets) as fh:
        targets = yaml.safe_load(fh)
    store = ResultsStore(args.results_dir)
    table = collect(store.load_all())
    result = evaluate(table, targets)
    report = render_report(result)
    print(report)
    out = args.out or str(Path(args.results_dir) / "repro_gate_report.md")
    Path(out).write_text(report)
    print("[repro_gate] report written to %s" % out)
    if result["failures"] or not result["rows"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
