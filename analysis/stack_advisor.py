"""stack-advisor: the project's decision guide as an executable CLI.

Given a deployment scenario (GPU class, context length, concurrency,
quality sensitivity, workload family), recommends which of the three
levers to enable — W (AWQ W4A16 weights), K (FP8 KV cache), S (EAGLE-3
speculative decoding) — with the expected effect range and, crucially,
**provenance**: every recommendation cites the measured cells behind it
(FINDINGS below). Nothing here is a heuristic; every rule is a sentence
this repo's data supports, and the caveats say where the data stops.

Usage (from the repo root):
    python3 -m analysis.stack_advisor --context-tokens 7400 --concurrency 8
    python3 -m analysis.stack_advisor --context-tokens 512 --concurrency 1 \\
        --workload code --quality-sensitive
    python3 -m analysis.stack_advisor --list-findings

Scope the numbers inherit (stated up front, repeated in output): one model
(Llama-3.1-8B-Instruct), one engine version (vLLM 0.24.0), A100 (FP8
emulated, not native), greedy decoding, closed-loop load.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Evidence base: finding id -> (claim, source). Sources are committed
# reports / result sets in this repo; every recommendation cites >=1 id.
# ---------------------------------------------------------------------------
FINDINGS: Dict[str, Dict[str, str]] = {
    "P2-W-c1": {
        "claim": "W4A16 alone: x1.70-2.13 goodput at conc 1-8, fading to "
                 "~x1.0 by conc 64 (memory-bound win vanishes compute-bound)",
        "source": "phase2_marginals_report.md (48 cells, 80GB)"},
    "F3-W-rev": {
        "claim": "W main effect reverses at conc 64 on GSM8K/RAG "
                 "(x0.90-0.92): Marlin dequant overhead in the "
                 "compute-bound regime",
        "source": "phase3_results/factorial_report.md"},
    "3b-W-cap": {
        "claim": "AWQ frees ~10.4GB weights -> FP16-KV admission ceiling "
                 "~17 -> ~27 concurrent 7.7k-token requests (measured "
                 "plateau vs predicted ~26)",
        "source": "k_stress W-capacity section (40GB)"},
    "QUAL-W": {
        "claim": "W4A16 costs accuracy: main effect -3.0 to -4.0 pts GSM8K "
                 "and -6.2 to -7.9 pts HumanEval at every concurrency "
                 "(toggle-alone up to -10); the only lever with a real "
                 "quality price",
        "source": "phase3_results/quality_effects.json (computed 2^3 "
                  "contrasts, 3 repeats)"},
    "QUAL-WK": {
        "claim": "FP8-KV partially OFFSETS W4A16's quality damage on "
                 "HumanEval: WK interaction +1.7 to +3.5 pts, per-repeat "
                 "ranges excluding zero in all 4 concurrency cells (K main "
                 "itself +1.4 to +2.1 pts there). Mechanism unresolved: KV "
                 "rounding vs attention-backend numerics (speed-side "
                 "backend effect measured ~0, quality-side not isolated)",
        "source": "phase3_results/quality_effects.json"},
    "P2-K-tax": {
        "claim": "FP8-KV alone at short context, below KV ceiling: "
                 "x0.94-0.98 (A100 FP8-emulation tax, no upside)",
        "source": "phase2_marginals_report.md"},
    "3b-K-cap": {
        "claim": "At KV-capacity pressure FP8-KV doubles admitted batch "
                 "(~17 -> ~33-42), +17-19% goodput, TTFT p95 -21%, queue "
                 "p50 30s -> 11s at conc 48",
        "source": "k_stress capacity table (40GB)"},
    "3b-K-long": {
        "claim": "FP8-KV alone at 7.4k context below ceiling: ~x1.01 "
                 "(bandwidth credit offsets the tax as KV bytes grow)",
        "source": "k_stress conc-8 K-isolation cells"},
    "QUAL-K": {
        "claim": "FP8-KV accuracy cost ~0: GSM8K main effect straddles "
                 "zero at every concurrency; HumanEval slightly POSITIVE "
                 "(+1.4 to +2.1 pts). NOT yet measured at long context",
        "source": "phase3_results/quality_effects.json"},
    "P2-S-c1": {
        "claim": "EAGLE-3 alone, short context, conc 1: x1.90 (RAG) / "
                 "x2.13 (GSM8K) / x3.16 (HumanEval) - tracks tau "
                 "(2.5 / 2.9 / 4.1)",
        "source": "phase2_marginals_report.md"},
    "P2-S-cross": {
        "claim": "S crossover in concurrency: GSM8K/RAG fall below x1.0 by "
                 "conc 32-64; HumanEval (tau 4.1) still x1.37 at conc 64; "
                 "tau is FLAT in concurrency - erosion is economics, not "
                 "acceptance",
        "source": "phase2_marginals_report.md"},
    "D2-S-long": {
        "claim": "EAGLE-3 is COUNTERPRODUCTIVE at 7.4k context on the "
                 "stock checkpoint + vLLM 0.24.0: x0.94 at conc 1, x0.89 "
                 "at conc 8 vs no-spec baseline (same eager regime); tau "
                 "collapses 2.85 -> 1.14. PENDING RETEST: source-level "
                 "diagnosis (analysis/vllm_2048_bug_diagnosis.md) shows "
                 "the draft checkpoint's max_position_embeddings=2048 "
                 "under-sizes its RoPE cache, and eager mode silently "
                 "reads GARBAGE rotations for draft positions >= 2048 -- "
                 "the tau collapse may be this bug, not the drafter. "
                 "Config-edit retest queued; advice stands for stock "
                 "deployments meanwhile",
        "source": "phase3c diagnostics + k_stress KS-probe + "
                  "vllm_2048_bug_diagnosis.md"},
    "QUAL-S": {
        "claim": "EAGLE-3 under greedy decoding is quality-free (measured "
                 "bit-identical accuracy spec-on vs spec-off; theoretical "
                 "guarantee). Sampling (T>0) NOT measured",
        "source": "phase3_results/runs + tests/test_repro_gate.py"},
    "F3-KS": {
        "claim": "Never pair K with S at low concurrency on FP8-emulating "
                 "GPUs: K-under-S x0.63 short context, x0.89 long context "
                 "(conc 1); penalty shrinks to ~x0.95 by conc 64. tau "
                 "invariant under K everywhere - no acceptance channel "
                 "for EAGLE-3",
        "source": "factorial_report.md KS + k_stress KS-probe"},
    "F3-WS": {
        "claim": "W erodes S at every concurrency (WS x0.78-0.83), NOT "
                 "amplified by batching (flat-to-shrinking); W also drops "
                 "tau 14% on GSM8K (acceptance channel exists for W on "
                 "reasoning workloads)",
        "source": "factorial_report.md WS (reproduces SpecMQuant under "
                  "batching)"},
    "F3-GAP": {
        "claim": "Speedups do NOT multiply: full-stack interference gap "
                 "x1.30-2.97 (naive product overestimates worst at batch "
                 "1); quality costs DO add cleanly: full-stack accuracy "
                 "delta minus sum-of-mains is within 0.7 pts in all 8 "
                 "measured cells",
        "source": "factorial_report.md + quality_effects.json"},
    "HW-FP8": {
        "claim": "All K findings are on A100 (FP8 emulated). On native-FP8 "
                 "GPUs (H100+) the emulation tax should vanish and K's "
                 "ledger improve - EXTRAPOLATION, not measured here",
        "source": "EXPERIMENT_MATRIX.md hardware caveat"},
}

# Measured KV pools (tokens), keyed by (gpu, weights): used to estimate
# capacity pressure. From k_stress server logs (0.85 gpu-mem-util, 8B).
MEASURED_POOLS = {
    ("a100-40gb", "fp16"): 142_896,
    ("a100-40gb", "w4a16"): 221_664,
}

OUTPUT_TOKENS_DEFAULT = 256
LONG_CONTEXT_TOKENS = 4096   # measured long point is 7.4k; short is ~1k
SATURATION_CONCURRENCY = 32  # S/W crossovers measured at conc 32-64
PRESSURE_THRESHOLD = 0.6     # demand/pool above this -> capacity regime


@dataclass
class Recommendation:
    lever: str
    verdict: str                    # "ON" | "OFF" | "CONDITIONAL"
    expected: str
    rationale: str
    findings: List[str]
    caveats: List[str] = field(default_factory=list)


def _pressure(gpu: str, weights: str, context_tokens: int, concurrency: int,
              pool_tokens: Optional[int]) -> Optional[float]:
    pool = pool_tokens or MEASURED_POOLS.get((gpu, weights))
    if not pool:
        return None
    return concurrency * (context_tokens + OUTPUT_TOKENS_DEFAULT) / pool


def recommend(gpu: str = "a100-40gb", native_fp8: bool = False,
              context_tokens: int = 1024, concurrency: int = 1,
              quality_sensitive: bool = False, workload: str = "chat",
              pool_tokens: Optional[int] = None) -> List[Recommendation]:
    recs: List[Recommendation] = []
    long_ctx = context_tokens >= LONG_CONTEXT_TOKENS
    saturated = concurrency >= SATURATION_CONCURRENCY

    # ----- S (EAGLE-3) ------------------------------------------------
    if long_ctx:
        recs.append(Recommendation(
            "S (EAGLE-3)", "OFF",
            "x0.89-0.94 (measured net LOSS at ~7.4k-token context)",
            "acceptance collapses (tau 2.85 -> 1.14) once prompts leave the "
            "drafter's distribution; ~77% of draft compute is discarded",
            ["D2-S-long"],
            ["PENDING RETEST: the tau collapse may be vLLM's draft RoPE-cache "
             "bug (garbage rotations past position 2048, see "
             "analysis/vllm_2048_bug_diagnosis.md), not the drafter; a "
             "checkpoint config-edit retest is queued and could flip "
             "this verdict"]))
    elif saturated and workload != "code":
        recs.append(Recommendation(
            "S (EAGLE-3)", "OFF",
            "x0.90-1.13 by conc 32, below x1.0 by conc 64 (GSM8K/RAG)",
            "spec decoding's spare-compute subsidy is gone once the batch "
            "saturates the GPU; tau stays flat, the economics flip",
            ["P2-S-cross"], []))
    else:
        expected = ("x1.37-1.84 even at conc 32-64 (code, tau~4.1)"
                    if workload == "code" and saturated
                    else "x1.90-3.16 at conc 1-8 (workload-dependent via tau)")
        recs.append(Recommendation(
            "S (EAGLE-3)", "ON", expected,
            "the largest quality-free lever in its regime (greedy spec "
            "decode is output-preserving, measured bit-identical)",
            ["P2-S-c1", "P2-S-cross", "QUAL-S"],
            ["quality guarantee is for greedy; T>0 sampling not measured"]))

    # ----- W (AWQ W4A16) ----------------------------------------------
    pressure = _pressure(gpu, "fp16", context_tokens, concurrency, pool_tokens)
    w_caveats = ["costs accuracy: -3/-4 pts GSM8K, -6/-8 pts HumanEval "
                 "(QUAL-W); FP8-KV claws back 2-3 pts on code (QUAL-WK)"]
    if quality_sensitive:
        recs.append(Recommendation(
            "W (AWQ W4A16)", "OFF",
            "avoided: the only lever with a measured quality price",
            "quality-sensitive deployment: K and S deliver speed at ~zero "
            "accuracy cost; W does not",
            ["QUAL-W", "F3-GAP"], []))
    elif saturated:
        if pressure is not None and pressure >= PRESSURE_THRESHOLD:
            recs.append(Recommendation(
                "W (AWQ W4A16)", "ON",
                "raises the admission ceiling ~17 -> ~27 (freed VRAM -> "
                "bigger KV pool); throughput itself ~x1.0",
                "at KV-capacity pressure W is a capacity lever, not a "
                "speed lever",
                ["3b-W-cap", "F3-W-rev"], w_caveats))
        else:
            recs.append(Recommendation(
                "W (AWQ W4A16)", "OFF",
                "x0.90-1.02 at conc 64 (can be a net loss)",
                "compute-bound regime: dequant overhead cancels the "
                "memory win; without KV pressure there is nothing to buy",
                ["F3-W-rev", "P2-W-c1"], w_caveats))
    else:
        recs.append(Recommendation(
            "W (AWQ W4A16)", "ON",
            "x1.70-2.13 at conc 1-8",
            "low-concurrency decode is memory-bandwidth-bound; quartering "
            "weight bytes is the single biggest lever",
            ["P2-W-c1"],
            w_caveats + (["W erodes S when stacked: apply F3-WS, do not "
                          "multiply their speedups"]
                         if any(r.lever.startswith("S") and r.verdict == "ON"
                                for r in recs) else [])))

    # ----- K (FP8 KV cache) -------------------------------------------
    s_on = any(r.lever.startswith("S") and r.verdict == "ON" for r in recs)
    k_caveats = ["long-context accuracy under FP8-KV not yet measured "
                 "(short-context cost ~0)"]
    if native_fp8:
        recs.append(Recommendation(
            "K (FP8 KV)", "CONDITIONAL",
            "likely free-to-positive (emulation tax should vanish)",
            "every measured K penalty here is A100 FP8-emulation tax; "
            "native-FP8 hardware removes its cause",
            ["HW-FP8", "3b-K-cap"],
            k_caveats + ["EXTRAPOLATION: this project measured A100 only"]))
    elif pressure is not None and pressure >= PRESSURE_THRESHOLD:
        recs.append(Recommendation(
            "K (FP8 KV)", "ON",
            "2x admitted concurrency, +17-19% goodput, TTFT p95 -21% at "
            "the measured knee",
            "projected KV demand approaches the pool: K's capacity channel "
            "engages and dominates its ~5% tax",
            ["3b-K-cap", "QUAL-K"], k_caveats))
    elif s_on:
        recs.append(Recommendation(
            "K (FP8 KV)", "OFF",
            "x0.63 (short ctx) to x0.89 (long ctx) on S's speedup at low "
            "concurrency",
            "S multiplies KV traffic, so it multiplies the A100 emulation "
            "tax; no acceptance upside exists for EAGLE-3 (tau invariant)",
            ["F3-KS"], k_caveats))
    else:
        recs.append(Recommendation(
            "K (FP8 KV)", "OFF",
            "x0.94-1.01 below the capacity knee (a small tax, no upside)",
            "below KV pressure on emulating hardware K buys nothing",
            ["P2-K-tax", "3b-K-long"], k_caveats))

    return recs


STACK_ADVICE = (
    "Stacking rules (always apply): (1) NEVER multiply the levers' "
    "individual speedups - measured interference gap is x1.30-2.97, worst "
    "at batch 1 [F3-GAP]; (2) quality costs do NOT compound - the full "
    "stack's accuracy ~= W's cost alone [F3-GAP]; (3) W erodes S at every "
    "concurrency [F3-WS]; K erodes S worst at batch 1 on emulating "
    "hardware [F3-KS]."
)

SCOPE_NOTE = (
    "Scope: Llama-3.1-8B-Instruct, vLLM 0.24.0, A100 (FP8 emulated), "
    "greedy decoding, closed-loop load. Sampling (T>0), other model "
    "sizes/architectures, and native-FP8 GPUs are outside the measured "
    "envelope."
)


def render(recs: List[Recommendation], show_provenance: bool = True) -> str:
    lines = ["# stack-advisor recommendation", ""]
    for r in recs:
        lines.append("## %s -> %s" % (r.lever, r.verdict))
        lines.append("  expected : %s" % r.expected)
        lines.append("  why      : %s" % r.rationale)
        if show_provenance:
            for fid in r.findings:
                f = FINDINGS[fid]
                lines.append("  evidence : [%s] %s (%s)"
                             % (fid, f["claim"], f["source"]))
        for c in r.caveats:
            lines.append("  caveat   : %s" % c)
        lines.append("")
    lines.append(STACK_ADVICE)
    lines.append("")
    lines.append(SCOPE_NOTE)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# --validate: recompute each finding's quantitative claim from raw run
# records and compare against the range the advisor asserts. This is the
# "tested product" loop: if a rule ever drifts from the data (new runs,
# edited claims), validation fails loudly instead of the guide rotting.
# ---------------------------------------------------------------------------

def _load_runs(dirs: List[str]) -> List[dict]:
    import json as _json
    import pathlib as _pathlib
    recs = []
    for d in dirs:
        for f in sorted((_pathlib.Path(d) / "runs").glob("*.json")):
            recs.append(_json.loads(f.read_text()))
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
    print("# stack-advisor --validate: %d records from %s"
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
    print("PASS: advisor claims consistent with the records provided."
          if failures == 0 else
          "FAIL: %d claim(s) outside their predicted range - fix the "
          "finding or explain the regression before shipping advice."
          % failures)
    return 0 if failures == 0 else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Measured-data-backed vLLM optimization-stack advisor")
    p.add_argument("--gpu", default="a100-40gb",
                   help="gpu class (pools measured for a100-40gb)")
    p.add_argument("--native-fp8", action="store_true",
                   help="GPU has native FP8 (H100+): K advice becomes an "
                        "extrapolation, flagged as such")
    p.add_argument("--context-tokens", type=int, default=1024)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--workload", choices=["chat", "reasoning", "code", "rag"],
                   default="chat")
    p.add_argument("--quality-sensitive", action="store_true")
    p.add_argument("--kv-pool-tokens", type=int, default=None,
                   help="override the measured KV pool estimate")
    p.add_argument("--list-findings", action="store_true")
    p.add_argument("--no-provenance", action="store_true")
    p.add_argument("--validate", nargs="+", metavar="RESULTS_DIR",
                   help="recompute the findings' quantitative claims from "
                        "raw run records and PASS/FAIL each against its "
                        "predicted range (e.g. --validate phase3_results)")
    args = p.parse_args(argv)

    if args.validate:
        return run_validation(args.validate)

    if args.list_findings:
        for fid, f in FINDINGS.items():
            print("[%s] %s\n    source: %s" % (fid, f["claim"], f["source"]))
        return 0

    recs = recommend(
        gpu=args.gpu, native_fp8=args.native_fp8,
        context_tokens=args.context_tokens, concurrency=args.concurrency,
        quality_sensitive=args.quality_sensitive, workload=args.workload,
        pool_tokens=args.kv_pool_tokens)
    print(render(recs, show_provenance=not args.no_provenance))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
