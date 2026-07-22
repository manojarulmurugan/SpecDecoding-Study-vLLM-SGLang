"""The decision guide's claims must stay consistent with the committed records.

Runs the extracted validator against the real result dirs (all GPU-free JSON) and
asserts every claim recomputes inside its predicted range. If a future edit to the
records or the ranges breaks a claim, this fails instead of the guide rotting.
"""
from __future__ import annotations

import pathlib

import pytest

from analysis import validate_claims

REPO = pathlib.Path(__file__).resolve().parent.parent
DIRS = ["phase3_results", "phase3b_results", "phase3c_diagnostics_results"]


def _have_records() -> bool:
    return all((REPO / d / "runs").is_dir() for d in DIRS)


pytestmark = pytest.mark.skipif(
    not _have_records(), reason="committed result dirs not present")


def test_all_claims_pass_against_committed_records():
    records = validate_claims._load_runs([str(REPO / d) for d in DIRS])
    assert records, "no records loaded"
    failures, checked = [], 0
    for fid, name, (lo, hi), _fmt, compute in validate_claims.VALIDATIONS:
        measured = compute(records)
        assert measured is not None, "%s (%s) produced no measurement" % (fid, name)
        checked += 1
        if not (lo <= measured <= hi):
            failures.append("%s (%s): %.3f not in [%.3f, %.3f]"
                            % (fid, name, measured, lo, hi))
    assert checked == len(validate_claims.VALIDATIONS)
    assert not failures, "claims outside predicted range:\n" + "\n".join(failures)


def test_run_validation_returns_zero():
    assert validate_claims.run_validation([str(REPO / d) for d in DIRS]) == 0
