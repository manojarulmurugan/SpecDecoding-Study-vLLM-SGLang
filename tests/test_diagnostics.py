"""Phase-3c diagnostics config set (configs/diagnostics/): the bundled
sanity/bisection session that settles the tau=1.14 question, measures the
eager no-spec long-context baseline, bisects the vLLM 0.24.0 crash to
CUDA-graph capture or not, and bounds the attention-backend component of
the K contrasts. See the generator docstring for the interpretation grid.
"""
from __future__ import annotations

import json

from analysis.factorial import collect as factorial_collect
from analysis.k_stress import collect as ks_collect, collect_ks_probe
from harness.config import load_configs
from harness.engines.vllm_adapter import VllmAdapter
from harness.sweep import group_by_server

GLOB = ["configs/diagnostics/diag_*.yaml"]


def test_diagnostics_config_set():
    cfgs = load_configs(GLOB)
    assert len(cfgs) == 9
    assert len({c.run_id for c in cfgs}) == 9
    assert len(group_by_server(cfgs)) == 4, "one launch per corner group"
    for c in cfgs:
        assert c.block == "diagnostics"
        assert c.gpu_target == "a100_40gb", "compared against Phase-3b 40GB cells"
        assert c.decoding == "greedy" and c.seed == 1234


def test_tau_check_mirrors_factorial_gsm8k_on_probe_server():
    cfgs = [c for c in load_configs(GLOB) if "tau-eager-short" in c.run_id]
    assert len(cfgs) == 2 and {c.repeat_idx for c in cfgs} == {0, 1}
    for c in cfgs:
        # workload mirrors the factorial gsm8k cells (tau ~2.85 reference)
        assert c.workload == "gsm8k" and c.concurrency == 1
        assert c.workload_params["num_requests"] == 64
        assert c.workload_params["n_shot"] == 8
        # server byte-compatible with the validated Phase-3b probe launch
        cmd = VllmAdapter(c).build_launch_command()
        assert "--speculative-config" in cmd
        assert "--enforce-eager" in cmd
        assert cmd[cmd.index("--max-num-batched-tokens") + 1] == "8192"


def test_slong_baseline_is_probe_server_minus_spec():
    cfgs = [c for c in load_configs(GLOB) if "slong-eager-base" in c.run_id]
    assert len(cfgs) == 4
    assert {(c.concurrency, c.repeat_idx) for c in cfgs} == {(1, 0), (1, 1), (8, 0), (8, 1)}
    for c in cfgs:
        # same long-doc workload as the k_stress probe cells
        assert c.workload == "rag_shared_prefix"
        assert c.workload_params["doc_target_tokens"] == 7400
        assert c.workload_params["prompt_token_budget"] == 7900
        assert c.workload_params["num_requests"] == {1: 24, 8: 64}[c.concurrency]
        cmd = VllmAdapter(c).build_launch_command()
        assert "--speculative-config" not in cmd, "the whole point: S off"
        assert "--enforce-eager" in cmd, "must stay in the probe's regime"
        assert cmd[cmd.index("--max-num-batched-tokens") + 1] == "8192"


def test_cudagraph_probe_compiles_without_graphs():
    cfgs = [c for c in load_configs(GLOB) if "cudagraph-probe" in c.run_id]
    assert len(cfgs) == 1
    cmd = VllmAdapter(cfgs[0]).build_launch_command()
    assert "--speculative-config" in cmd
    assert "--enforce-eager" not in cmd, "compile must stay ON for the bisection"
    cc = cmd[cmd.index("--compilation-config") + 1]
    assert json.loads(cc) == {"cudagraph_mode": "NONE"}


def test_backend_pin_is_compiled_flashinfer():
    cfgs = [c for c in load_configs(GLOB) if "backendpin-flashinfer" in c.run_id]
    assert len(cfgs) == 2
    for c in cfgs:
        assert c.engine_args.env == {"VLLM_ATTENTION_BACKEND": "FLASHINFER"}
        cmd = VllmAdapter(c).build_launch_command()
        # compiled and default-budget, like the k_stress fp16kv cells it
        # is compared against (FLASH_ATTN, ~221 tok/s at c8)
        assert "--enforce-eager" not in cmd
        assert "--max-num-batched-tokens" not in cmd
        assert "--speculative-config" not in cmd


def test_diagnostics_records_excluded_from_other_analyses():
    rec = {
        "run_id": "diag_slong-eager-base_c8_r0",
        "config": {"block": "diagnostics", "workload": "rag_shared_prefix",
                   "concurrency": 8, "repeat_idx": 0,
                   "factors": {"weight_quant": "fp16", "kv_quant": "fp16",
                               "spec_decode": "none"}},
        "env": {}, "status": "ok",
        "measured": {"goodput_tok_s": 100.0},
    }
    assert factorial_collect([rec]) == {}
    assert ks_collect([rec]) == {}
    assert collect_ks_probe([rec]) == {}
