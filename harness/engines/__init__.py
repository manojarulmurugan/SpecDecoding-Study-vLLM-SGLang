from .base import EngineAdapter, ServerHandle
from .vllm_adapter import VllmAdapter

_ADAPTERS = {
    "vllm": VllmAdapter,
}


def get_adapter(engine: str) -> "type[EngineAdapter]":
    try:
        return _ADAPTERS[engine]
    except KeyError:
        raise ValueError(
            "no adapter for engine %r (have: %s)" % (engine, sorted(_ADAPTERS))
        )
