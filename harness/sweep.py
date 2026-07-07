"""Sweep orchestrator: group runs by server signature, launch once per group.

Server startup (model download, CUDA graph capture) dominates cell runtime,
so all cells sharing a launch command run against one server process
(HARNESS_SPEC.md §5 "amortize server startup"). Completed cells are skipped
on resume, so a Colab disconnect costs one cell, never a sweep.

Usage:
    python -m harness.sweep "configs/repro/*.yaml" --results-dir results
    python -m harness.sweep cfg.yaml --dry-run          # print commands only
    python -m harness.sweep cfg.yaml --server-url URL   # external server
"""
from __future__ import annotations

import argparse
import sys
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

from .config import RunConfig, load_configs
from .engines import get_adapter
from .engines.base import ServerHandle
from .results import ResultsStore
from .run import execute_run


def group_by_server(configs: List[RunConfig]) -> "OrderedDict[str, List[RunConfig]]":
    groups: "OrderedDict[str, List[RunConfig]]" = OrderedDict()
    for cfg in configs:
        groups.setdefault(cfg.server_signature(), []).append(cfg)
    return groups


def run_sweep(
    configs: List[RunConfig],
    store: ResultsStore,
    dry_run: bool = False,
    server_url: Optional[str] = None,
    log=print,
) -> Dict[str, List[str]]:
    """Returns {"ok": [...], "skipped": [...], "failed": [...]} run_ids."""
    outcome: Dict[str, List[str]] = {"ok": [], "skipped": [], "failed": []}
    groups = group_by_server(configs)
    if server_url and len(groups) > 1:
        raise ValueError(
            "--server-url given but configs span %d distinct server "
            "configurations; run them separately" % len(groups)
        )
    log("[sweep] %d run(s) in %d server group(s)" % (len(configs), len(groups)))

    for gi, (signature, group) in enumerate(groups.items(), 1):
        pending = [c for c in group if not store.is_complete(c.run_id)]
        for c in group:
            if store.is_complete(c.run_id):
                log("[sweep] skip (complete): %s" % c.run_id)
                outcome["skipped"].append(c.run_id)
        if not pending:
            continue

        adapter = get_adapter(pending[0].engine)(pending[0])
        cmd = adapter.build_launch_command()
        log("[sweep] group %d/%d: %d pending run(s)" % (gi, len(groups), len(pending)))
        log("[sweep] server command: %s" % " ".join(cmd))
        if dry_run:
            for c in pending:
                log("[sweep]   would run: %s" % c.run_id)
            continue

        if server_url:
            handle = ServerHandle(process=None, base_url=server_url, external=True)
        else:
            handle = adapter.launch(Path(store.root) / "server_logs")
            log("[sweep] waiting for server (log: %s)" % handle.log_path)
            adapter.wait_ready(handle, log=log)
        try:
            for cfg in pending:
                try:
                    record = execute_run(cfg, store, adapter, handle, log=log)
                    key = "ok" if record["status"] == "ok" else "failed"
                    outcome[key].append(cfg.run_id)
                except Exception:
                    log("[sweep] run %s FAILED:\n%s" % (cfg.run_id, traceback.format_exc()))
                    outcome["failed"].append(cfg.run_id)
        finally:
            adapter.teardown(handle)

    log(
        "[sweep] done: %d ok, %d skipped, %d failed"
        % (len(outcome["ok"]), len(outcome["skipped"]), len(outcome["failed"]))
    )
    return outcome


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", help="config YAMLs or globs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--server-url", default=None)
    args = parser.parse_args(argv)

    configs = load_configs(args.configs)
    store = ResultsStore(args.results_dir)
    outcome = run_sweep(
        configs, store, dry_run=args.dry_run, server_url=args.server_url
    )
    return 1 if outcome["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
