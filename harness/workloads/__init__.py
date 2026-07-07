from .base import PromptItem, Workload
from .gsm8k import Gsm8kWorkload
from .humaneval import HumanEvalWorkload
from .rag_shared_prefix import RagSharedPrefixWorkload

_WORKLOADS = {
    "gsm8k": Gsm8kWorkload,
    "humaneval": HumanEvalWorkload,
    "rag_shared_prefix": RagSharedPrefixWorkload,
}


def get_workload(name: str) -> "type[Workload]":
    try:
        return _WORKLOADS[name]
    except KeyError:
        raise ValueError(
            "no workload %r (have: %s; mt_bench is an optional Phase-3+ "
            "extension)" % (name, sorted(_WORKLOADS))
        )
