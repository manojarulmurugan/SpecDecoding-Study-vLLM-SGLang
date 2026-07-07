"""Atomic result store: one JSON file per run, resumable sweeps.

A record is written with temp-file + os.replace so a kill mid-write never
corrupts the store (HARNESS_SPEC.md §4). ``sweep.py`` resumes by skipping any
run_id already present with status "ok".
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

REQUIRED_KEYS = ("run_id", "config", "env", "measured", "status")


class ResultsStore:
    def __init__(self, root: "Path | str"):
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        return self.runs_dir / ("%s.json" % run_id)

    def write(self, record: Dict[str, Any]) -> Path:
        missing = [k for k in REQUIRED_KEYS if k not in record]
        if missing:
            raise ValueError("result record missing keys: %s" % missing)
        path = self._path(record["run_id"])
        fd, tmp = tempfile.mkstemp(dir=str(self.runs_dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(record, fh, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return path

    def load(self, run_id: str) -> Optional[Dict[str, Any]]:
        path = self._path(run_id)
        if not path.exists():
            return None
        with open(path) as fh:
            return json.load(fh)

    def is_complete(self, run_id: str) -> bool:
        record = self.load(run_id)
        return bool(record) and record.get("status") == "ok"

    def load_all(self) -> List[Dict[str, Any]]:
        records = []
        for path in sorted(self.runs_dir.glob("*.json")):
            with open(path) as fh:
                records.append(json.load(fh))
        return records

    def export_jsonl(self, path: "Path | str") -> int:
        records = self.load_all()
        with open(path, "w") as fh:
            for rec in records:
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
        return len(records)
