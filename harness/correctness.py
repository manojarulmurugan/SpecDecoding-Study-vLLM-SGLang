"""Per-workload correctness scoring (HARNESS_SPEC.md §2).

Thin dispatcher: each workload owns its scoring logic; this module is the
stable entry point run.py calls.

Free sanity check the spec docs don't mention: speculative decoding under
greedy is output-preserving, so at concurrency 1 the accuracy of a spec-on
cell must match its spec-off partner (up to numeric nondeterminism). A
mismatch is a harness or engine-config bug, not a finding.
"""
from __future__ import annotations

from typing import Sequence

from .workloads.base import PromptItem, ScoreResult, Workload


def score_run(
    workload: Workload, items: Sequence[PromptItem], outputs: Sequence[str]
) -> ScoreResult:
    if len(items) != len(outputs):
        raise ValueError(
            "items/outputs length mismatch: %d vs %d" % (len(items), len(outputs))
        )
    return workload.score(items, outputs)
