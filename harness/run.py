"""Execute one run cell: config -> timed load -> scored, atomic result record.

Usable three ways:
  - ``python -m harness.run config.yaml`` launches its own server, runs,
    tears down.
  - ``python -m harness.run config.yaml --server-url http://...`` drives an
    already-running server (manual Colab control, or the test fake server).
  - ``execute_run(...)`` called by sweep.py with a shared server handle.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .config import RunConfig, load_configs
from .correctness import score_run
from .engines import get_adapter
from .engines.base import EngineAdapter, ServerHandle
from .env_info import collect_env
from .load import run_closed_loop
from .metrics import aggregate_run, spec_decode_stats
from .results import ResultsStore
from .workloads import get_workload


def execute_run(
    config: RunConfig,
    store: ResultsStore,
    adapter: EngineAdapter,
    handle: ServerHandle,
    log=print,
) -> Dict[str, Any]:
    workload = get_workload(config.workload)(config.workload_params, seed=config.seed)
    items = workload.build()
    log(
        "[run] %s: %d prompts, concurrency=%d, max_new_tokens=%d"
        % (config.run_id, len(items), config.concurrency, workload.max_new_tokens())
    )

    # Warmup happens inside run_closed_loop; scrape counters after it would
    # be ideal, but warmup uses the same code path, so scrape happens in two
    # steps: warmup first, then metrics, then the timed window.
    if config.warmup_requests:
        run_closed_loop(
            handle.base_url, config.model,
            [items[i % len(items)].prompt for i in range(config.warmup_requests)],
            concurrency=1,
            max_tokens=workload.max_new_tokens(),
            temperature=config.temperature(),
            stop=workload.stop(),
            seed=config.seed,
            progress_every=0,
            log=log,
        )

    metrics_before = adapter.scrape_metrics(handle)
    load_result = run_closed_loop(
        handle.base_url, config.model,
        [item.prompt for item in items],
        concurrency=config.concurrency,
        max_tokens=workload.max_new_tokens(),
        temperature=config.temperature(),
        stop=workload.stop(),
        seed=config.seed,
        log=log,
    )
    metrics_after = adapter.scrape_metrics(handle)

    spec_stats = None
    if config.factors.spec_decode != "none":
        spec_stats = spec_decode_stats(metrics_before, metrics_after)
        if spec_stats is None:
            log("[run] WARNING: spec decoding on but no spec_decode counters "
                "found at /metrics -- tau will be null")

    measured = aggregate_run(load_result.results, load_result.wall_time_s, spec_stats)

    outputs = [r.text for r in sorted(load_result.results, key=lambda r: r.index)]
    score = score_run(workload, items, outputs)
    measured["accuracy"] = score.accuracy

    n_err = measured["num_errors"]
    status = "ok" if n_err == 0 else ("partial" if n_err < len(items) else "failed")
    record: Dict[str, Any] = {
        "run_id": config.run_id,
        "config": config.to_dict(),
        "env": collect_env(engine_version=adapter.server_version(handle)),
        "measured": measured,
        "score_details": score.details,
        "status": status,
    }
    path = store.write(record)
    log(
        "[run] %s -> %s (status=%s, %.1f tok/s mean/request, tau=%s, acc=%s)"
        % (
            config.run_id, path, status,
            measured["request_tok_s_mean"] or float("nan"),
            _fmt(measured["accepted_length_tau"]),
            _fmt(measured["accuracy"]),
        )
    )
    return record


def _fmt(value) -> str:
    return "%.3f" % value if isinstance(value, float) else str(value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="run-config YAML")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--server-url", default=None,
                        help="drive an already-running server instead of launching")
    parser.add_argument("--force", action="store_true",
                        help="rerun even if a completed record exists")
    args = parser.parse_args(argv)

    config = load_configs([args.config])[0]
    store = ResultsStore(args.results_dir)
    if store.is_complete(config.run_id) and not args.force:
        print("[run] %s already complete; use --force to rerun" % config.run_id)
        return 0

    adapter = get_adapter(config.engine)(config)
    if args.server_url:
        handle = ServerHandle(process=None, base_url=args.server_url, external=True)
    else:
        print("[run] launching server: %s" % " ".join(adapter.build_launch_command()))
        handle = adapter.launch(Path(args.results_dir) / "server_logs")
        adapter.wait_ready(handle)
    try:
        record = execute_run(config, store, adapter, handle)
    finally:
        adapter.teardown(handle)
    return 0 if record["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
