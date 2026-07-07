from .base import PromptItem, Workload
from .gsm8k import Gsm8kWorkload
from .humaneval import HumanEvalWorkload

_WORKLOADS = {
    "gsm8k": Gsm8kWorkload,
    "humaneval": HumanEvalWorkload,
}


def get_workload(name: str) -> "type[Workload]":
    try:
        return _WORKLOADS[name]
    except KeyError:
        raise ValueError(
            "no workload %r (have: %s; rag_shared_prefix/mt_bench land in "
            "Phase 2+)" % (name, sorted(_WORKLOADS))
        )
