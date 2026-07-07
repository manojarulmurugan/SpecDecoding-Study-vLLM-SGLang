"""Workload contract: build raw completion prompts, score raw outputs.

Prompts are raw strings sent to /v1/completions (never /v1/chat/completions)
so the harness controls the template byte-for-byte -- required both for
reproduction fidelity vs SpecMQuant and, later, for the byte-identical
shared-prefix guarantee of the RAG workload.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class PromptItem:
    prompt: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreResult:
    accuracy: Optional[float]
    details: List[Dict[str, Any]] = field(default_factory=list)


class Workload(ABC):
    name: str = "base"
    default_max_new_tokens: int = 256

    def __init__(self, params: Dict[str, Any], seed: int = 1234):
        self.params = dict(params or {})
        self.seed = seed

    @abstractmethod
    def build(self) -> List[PromptItem]:
        """Deterministic prompt list for this run (given params + seed)."""

    @abstractmethod
    def score(self, items: Sequence[PromptItem], outputs: Sequence[str]) -> ScoreResult:
        """Correctness metric over raw generated texts."""

    def max_new_tokens(self) -> int:
        return int(self.params.get("max_new_tokens", self.default_max_new_tokens))

    def stop(self) -> Optional[List[str]]:
        return self.params.get("stop")

    def subsample(self, records: List[Any]) -> List[Any]:
        """Deterministic selection of exactly num_requests records.

        - subset: seeded sample, original order kept, so runs with different
          num_requests share their ordering decision.
        - oversample (num_requests > dataset size, e.g. HumanEval's 164 at a
          concurrency-scaled 320): tile the dataset deterministically. This
          is standard for serving load generation; prefix caching is off in
          controlled cells, so repeats aren't served from cache.
        """
        n = self.params.get("num_requests")
        if not n:
            return records
        n = int(n)
        if n == len(records):
            return records
        if n < len(records):
            rng = random.Random(self.seed)
            picked = sorted(rng.sample(range(len(records)), n))
            return [records[i] for i in picked]
        tiled = records * ((n + len(records) - 1) // len(records))
        return tiled[:n]
