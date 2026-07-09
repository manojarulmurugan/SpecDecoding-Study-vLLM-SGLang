"""RAG shared-prefix workload (HARNESS_SPEC.md §7) -- handle with care.

The premise: many questions about the same long document, where everything
before the question -- system preamble + document text -- is **byte-identical
and consistently ordered** across the batch. A single differing character
breaks tokenizer-level prefix matching and silently invalidates every
prefix-cache comparison built on this workload (vLLM APC and, in Block 5,
SGLang RadixAttention).

Guarantees enforced here:
1. One fixed template, ordered system-preamble -> document -> question,
   never reordered. The shared prefix ends at a clean "\\n\\n" boundary so
   BPE tokenization of the prefix is stable regardless of the question.
2. ``check_shared_prefix_token_ids`` verifies token-ID equality of the
   common prefix across a same-document batch for a given tokenizer --
   unit-tested with synthetic tokenizers, and runnable with the real model
   tokenizer in Colab before a sweep.
3. The ``prefix_overlap`` knob (low | mid | high) maps to questions-per-
   document; requests are deterministically seeded-shuffled so same-doc
   requests interleave the way concurrent clients would.

Document sources (params, priority order):
- ``questions_file``: JSONL rows {"document": ..., "question": optional}.
- ``spec_bench_file``: Spec-Bench's question.jsonl; the 80 "rag" rows are
  NQ-style ``<passage>\\n<question-last-line>`` turns -- split accordingly
  (comparability anchor; PREREQ_RESULTS Check 5).
- ``synthetic: true``: deterministic clean-room documents (control set).

Correctness scoring: templated questions have no ground truth; accuracy is
None by design for this workload (throughput/latency instrument, not a
quality benchmark -- EXPERIMENT_MATRIX §4 "the instrument, not the subject").
"""
from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Sequence

from .base import PromptItem, ScoreResult, Workload

SYSTEM_PREAMBLE = (
    "You are a careful assistant. Answer the question using only the "
    "information in the document below.\n\n"
)

# questions-per-document per overlap level (HARNESS_SPEC §7: the knob)
OVERLAP_QPD = {"low": 1, "mid": 8, "high": 32}

QUESTION_TEMPLATES = [
    "What is the main topic of the document?",
    "Summarize the document in two sentences.",
    "List three key facts stated in the document.",
    "What problem or question does the document address?",
    "What conclusions does the document reach?",
    "Who or what are the main entities discussed in the document?",
    "What evidence does the document provide for its main claim?",
    "What time period or dates does the document refer to?",
    "What locations are mentioned in the document?",
    "What numbers or quantities appear in the document, and what do they measure?",
    "What causes or explanations does the document offer?",
    "What consequences or effects does the document describe?",
    "How does the document define its central concept?",
    "What comparisons does the document make?",
    "What limitations or caveats does the document mention?",
    "What is the most surprising claim in the document?",
    "What terminology does the document introduce or rely on?",
    "What sequence of events does the document describe?",
    "What disagreements or controversies does the document mention?",
    "What recommendations or implications follow from the document?",
    "Which sections of the document contain factual claims versus opinions?",
    "What background knowledge does the document assume?",
    "What questions does the document leave unanswered?",
    "How would you title this document, and why?",
]


def build_prefix(doc_text: str) -> str:
    return SYSTEM_PREAMBLE + "Document:\n" + doc_text.strip() + "\n\n"


def build_prompt(prefix: str, question: str) -> str:
    return prefix + "Question: " + question.strip() + "\nAnswer:"


def questions_for_doc(qpd: int, original_question: Optional[str] = None) -> List[str]:
    """Deterministic question list: the source question first (when the
    dataset has one), then generic templates, extended with an aspect
    suffix when qpd exceeds the template pool."""
    questions: List[str] = []
    if original_question:
        questions.append(original_question.strip())
    i = 0
    while len(questions) < qpd:
        base = QUESTION_TEMPLATES[i % len(QUESTION_TEMPLATES)]
        round_num = i // len(QUESTION_TEMPLATES)
        questions.append(base if round_num == 0 else "%s (aspect %d)" % (base, round_num + 1))
        i += 1
    return questions[:qpd]


def split_spec_bench_rag_turn(turn: str) -> "tuple[str, str]":
    """Spec-Bench RAG turns are '<passage>\\n<question as last line>'."""
    doc, _, question = turn.rstrip().rpartition("\n")
    if not doc:
        raise ValueError("spec-bench rag turn has no document/question split")
    return doc, question


def synthetic_documents(n_docs: int, sentences_per_doc: int = 120) -> List[str]:
    """Deterministic clean-room documents (seeded by doc index only, so the
    corpus is identical across runs and seeds)."""
    subjects = ["The committee", "The survey", "The river system", "The archive",
                "The observatory", "The cooperative", "The expedition", "The registry"]
    verbs = ["documented", "measured", "reorganized", "compared", "preserved",
             "catalogued", "reassessed", "expanded"]
    objects = ["the northern district", "seasonal rainfall patterns",
               "the original manuscripts", "trade records from the period",
               "the irrigation network", "population estimates",
               "the classification scheme", "long-term observations"]
    tails = ["over several decades.", "despite limited funding.",
             "before the reforms took effect.", "across all regions studied.",
             "with newly standardized methods.", "under changing conditions.",
             "in collaboration with local groups.", "for the annual report."]
    docs = []
    for d in range(n_docs):
        rng = random.Random(10_000 + d)  # doc identity, not run seed
        sentences = [
            " ".join([rng.choice(subjects), rng.choice(verbs),
                      rng.choice(objects), rng.choice(tails)])
            for _ in range(sentences_per_doc)
        ]
        docs.append(" ".join(sentences))
    return docs


def check_shared_prefix_token_ids(items: Sequence[PromptItem], tokenizer) -> int:
    """Assert every same-document prompt starts with the identical prefix
    token IDs under ``tokenizer`` (anything with .encode -> list[int]).

    Returns the number of groups checked. Raises AssertionError naming the
    offending document on the first violation -- this is the unit-test /
    pre-sweep guard HARNESS_SPEC §10 requires.
    """
    by_doc: Dict[Any, List[PromptItem]] = {}
    for item in items:
        by_doc.setdefault(item.meta["doc_id"], []).append(item)
    for doc_id, group in by_doc.items():
        prefixes = {g.meta["prefix"] for g in group}
        assert len(prefixes) == 1, (
            "doc %r: prefixes are not byte-identical across the batch" % doc_id
        )
        prefix_ids = list(tokenizer.encode(group[0].meta["prefix"]))
        for g in group:
            ids = list(tokenizer.encode(g.prompt))
            assert ids[: len(prefix_ids)] == prefix_ids, (
                "doc %r: prompt token IDs diverge inside the shared prefix "
                "(tokenizer merged across the prefix boundary?)" % doc_id
            )
    return len(by_doc)


class RagSharedPrefixWorkload(Workload):
    name = "rag_shared_prefix"
    default_max_new_tokens = 256

    def _load_docs(self) -> List[Dict[str, Any]]:
        """-> [{"document": str, "question": Optional[str]}]"""
        path = self.params.get("questions_file")
        if path:
            records = []
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        row = json.loads(line)
                        records.append({
                            "document": row["document"],
                            "question": row.get("question"),
                        })
            return records
        sb_path = self.params.get("spec_bench_file")
        if sb_path:
            records = []
            with open(sb_path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if row.get("category") != "rag":
                        continue
                    doc, question = split_spec_bench_rag_turn(row["turns"][0])
                    records.append({"document": doc, "question": question})
            if not records:
                raise ValueError("no 'rag' rows found in %s" % sb_path)
            return records
        n_docs = int(self.params.get("synthetic_num_docs", 16))
        return [
            {"document": doc, "question": None}
            for doc in synthetic_documents(n_docs)
        ]

    def _size_docs(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Grow each document to ~params["doc_target_tokens"] by cyclically
        concatenating source documents (K-stress addendum: Spec-Bench
        passages are ~800 tokens; capacity pressure needs ~7k+).

        Token count is approximated as words * 1.3 -- adequate because the
        capacity arithmetic only needs the right ballpark, and every run
        records the measured prompt_tokens_mean. Docs are trimmed to a
        uniform word count so per-request KV demand is homogeneous.
        """
        target_tokens = self.params.get("doc_target_tokens")
        if not target_tokens:
            return records
        target_words = max(1, int(int(target_tokens) / 1.3))
        sized = []
        n = len(records)
        for j in range(n):
            words: List[str] = []
            i = j
            while len(words) < target_words:
                words.extend(records[i % n]["document"].split())
                i += 1
            sized.append({
                "document": " ".join(words[:target_words]),
                "question": records[j].get("question"),
            })
        return sized

    def questions_per_doc(self) -> int:
        if "questions_per_doc" in self.params:
            return int(self.params["questions_per_doc"])
        overlap = self.params.get("prefix_overlap", "high")
        if overlap not in OVERLAP_QPD:
            raise ValueError(
                "prefix_overlap=%r not in %s" % (overlap, sorted(OVERLAP_QPD))
            )
        return OVERLAP_QPD[overlap]

    def build(self) -> List[PromptItem]:
        num_requests = int(self.params.get("num_requests", 64))
        qpd = self.questions_per_doc()
        docs = self._size_docs(self._load_docs())

        n_docs_needed = (num_requests + qpd - 1) // qpd
        if n_docs_needed > len(docs):
            docs = (docs * ((n_docs_needed + len(docs) - 1) // len(docs)))
        docs = docs[:n_docs_needed]

        items: List[PromptItem] = []
        for doc_id, rec in enumerate(docs):
            prefix = build_prefix(rec["document"])
            for q in questions_for_doc(qpd, rec.get("question")):
                items.append(PromptItem(
                    prompt=build_prompt(prefix, q),
                    meta={"doc_id": doc_id, "prefix": prefix, "question": q},
                ))
        items = items[:num_requests]
        # Interleave documents the way concurrent clients would: seeded
        # shuffle, deterministic per (seed), recorded via the config.
        rng = random.Random(self.seed)
        rng.shuffle(items)
        return items

    def score(self, items: Sequence[PromptItem], outputs: Sequence[str]) -> ScoreResult:
        # No ground truth for templated questions: this workload is the
        # performance instrument, not a quality benchmark. Record emptiness
        # rate as a smoke signal, not accuracy.
        empty = sum(1 for o in outputs if not (o or "").strip())
        return ScoreResult(
            accuracy=None,
            details=[{"num_empty_outputs": empty, "num_outputs": len(outputs)}],
        )
