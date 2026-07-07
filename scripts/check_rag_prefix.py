"""Pre-sweep guard: verify byte-identical RAG prefixes survive the REAL
model tokenizer (HARNESS_SPEC §7/§10) — beyond the synthetic-tokenizer unit
test. Run from the repo root, inside the vLLM venv (needs transformers):

    python scripts/check_rag_prefix.py [tokenizer_id] [spec_bench_file]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.workloads.rag_shared_prefix import (  # noqa: E402
    RagSharedPrefixWorkload,
    check_shared_prefix_token_ids,
)


def main() -> int:
    tokenizer_id = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.1-8B-Instruct"
    sb_file = sys.argv[2] if len(sys.argv) > 2 else "external/Spec-Bench/data/spec_bench/question.jsonl"

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    workload = RagSharedPrefixWorkload(
        {"spec_bench_file": sb_file, "num_requests": 64, "prefix_overlap": "high"},
        seed=1234,
    )
    groups = check_shared_prefix_token_ids(workload.build(), tokenizer)
    print("OK: prefix token-ID equality verified across %d document groups "
          "with tokenizer %s" % (groups, tokenizer_id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
