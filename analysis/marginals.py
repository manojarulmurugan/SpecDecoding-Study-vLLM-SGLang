"""Phase-2 analysis: serving baseline + single-optimization marginals.

Per workload, tabulates goodput across concurrency for each configuration
(baseline / W4A16 / FP8-KV / EAGLE-3), speedup vs the FP16 baseline at the
same (workload, concurrency), the measured emergent batch size, and tau for
spec cells. This is the input to Phase 2's headline plot: how each
optimization's benefit moves with offered load.

The full 2^3 interaction analysis (analysis/factorial.py) lands in Phase 3
once the remaining four corners are run; this report only claims marginals.

Usage:
    python -m analysis.marginals results --out results/phase2_marginals_report.md
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from harness.results import ResultsStore

CONFIG_LABELS = {
    ("fp16", "fp16", "none"): "baseline",
    ("w4a16", "fp16", "none"): "W (w4a16)",
    ("fp16", "fp8", "none"): "K (fp8-kv)",
    ("fp16", "fp16", "eagle3"): "S (eagle3)",
    ("fp16", "fp16", "eagle"): "S (eagle)",
}
BASELINE = "baseline"


def collect(records: List[Dict[str, Any]]) -> Dict[Tuple[str, int, str], Dict[str, Any]]:
    """-> {(workload, concurrency, label): measured} for ok factorial cells."""
    out: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for rec in records:
        cfg = rec.get("config", {})
        if cfg.get("block") not in ("core_factorial", "serving_baseline"):
            continue
        if rec.get("status") != "ok":
            continue
        f = cfg["factors"]
        label = CONFIG_LABELS.get(
            (f["weight_quant"], f["kv_quant"], f["spec_decode"])
        )
        if label is None:
            continue  # multi-factor corners belong to the Phase-3 analysis
        key = (cfg["workload"], int(cfg["concurrency"]), label)
        if key in out and cfg.get("repeat_idx", 0) != 0:
            continue  # marginals report uses repeat 0; spreads come in Phase 3
        out[key] = rec["measured"]
    return out


def render_report(cells: Dict[Tuple[str, int, str], Dict[str, Any]]) -> str:
    workloads = sorted({k[0] for k in cells})
    concurrencies = sorted({k[1] for k in cells})
    labels = [l for l in ("baseline", "W (w4a16)", "K (fp8-kv)", "S (eagle3)", "S (eagle)")
              if any(k[2] == l for k in cells)]

    lines = ["# Phase-2 marginals: goodput vs concurrency", ""]
    missing: List[str] = []
    for wl in workloads:
        lines.append("## %s" % wl)
        lines.append("")
        header = "| conc | " + " | ".join(
            "%s goodput (xbase)" % l for l in labels
        ) + " | emergent batch (mean/max, baseline) | tau (S) |"
        lines.append(header)
        lines.append("|" + "---|" * (len(labels) + 3))
        for conc in concurrencies:
            base = cells.get((wl, conc, BASELINE))
            row = ["| %d " % conc]
            for label in labels:
                m = cells.get((wl, conc, label))
                if m is None:
                    row.append("| — ")
                    if label == BASELINE:
                        missing.append("%s c%d %s" % (wl, conc, label))
                    continue
                goodput = m.get("goodput_tok_s") or m.get("throughput_tok_s")
                cell = "%.0f" % goodput if goodput else "?"
                if base and label != BASELINE and goodput:
                    base_goodput = base.get("goodput_tok_s") or base.get("throughput_tok_s")
                    if base_goodput:
                        cell += " (%.2fx)" % (goodput / base_goodput)
                row.append("| %s " % cell)
            batch = (base or {}).get("emergent_batch_size")
            row.append(
                "| %.1f / %.0f " % (batch["mean"], batch["max"]) if batch else "| — "
            )
            taus = [
                cells[(wl, conc, l)].get("accepted_length_tau")
                for l in labels
                if l.startswith("S") and (wl, conc, l) in cells
            ]
            tau = next((t for t in taus if t), None)
            row.append("| %.2f |" % tau if tau else "| — |")
            lines.append("".join(row))
        lines.append("")
    if missing:
        lines.append("## Missing baseline cells (speedups above are absolute-only)")
        lines += ["- %s" % m for m in missing]
        lines.append("")
    if not workloads:
        lines.append("NO DATA: no completed core_factorial/serving_baseline records found.")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    store = ResultsStore(args.results_dir)
    cells = collect(store.load_all())
    report = render_report(cells)
    print(report)
    out = args.out or str(Path(args.results_dir) / "phase2_marginals_report.md")
    Path(out).write_text(report)
    print("[marginals] report written to %s" % out)
    return 0 if cells else 1


if __name__ == "__main__":
    sys.exit(main())
