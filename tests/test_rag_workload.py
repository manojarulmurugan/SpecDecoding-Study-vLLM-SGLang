from __future__ import annotations

import json

import pytest

from harness.workloads.rag_shared_prefix import (
    OVERLAP_QPD,
    RagSharedPrefixWorkload,
    build_prefix,
    build_prompt,
    check_shared_prefix_token_ids,
    questions_for_doc,
    split_spec_bench_rag_turn,
    synthetic_documents,
)


class ByteTokenizer:
    """Deterministic prefix-preserving tokenizer: one token per byte."""

    def encode(self, text):
        return list(text.encode("utf-8"))


class BoundaryMergingTokenizer:
    """Hostile tokenizer modeling the real BPE failure mode: it merges the
    prefix's trailing "\\n\\n" with the start of the question into one token,
    so the prompt's leading token IDs no longer contain the prefix's IDs."""

    MERGED_TOKEN = 999_999

    def encode(self, text):
        marker = "\n\nQuestion:"
        ids = []
        i = 0
        while i < len(text):
            if text.startswith(marker, i):
                ids.append(self.MERGED_TOKEN)
                i += len(marker)
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids


def _workload(**params):
    defaults = {"synthetic_num_docs": 4, "num_requests": 16, "prefix_overlap": "mid"}
    defaults.update(params)
    return RagSharedPrefixWorkload(defaults, seed=1234)


# -- template & structure ------------------------------------------------------

def test_prefix_is_byte_identical_and_ordered():
    prefix = build_prefix("  Doc body.  ")
    assert prefix.startswith("You are a careful assistant")
    assert prefix.endswith("Document:\nDoc body.\n\n")
    p1 = build_prompt(prefix, "Q one?")
    p2 = build_prompt(prefix, "Q two?")
    assert p1.startswith(prefix) and p2.startswith(prefix)
    assert p1.endswith("Question: Q one?\nAnswer:")


def test_questions_for_doc_deterministic_and_original_first():
    qs = questions_for_doc(5, original_question=" Why? ")
    assert qs[0] == "Why?"
    assert len(qs) == 5
    assert qs == questions_for_doc(5, original_question=" Why? ")
    # beyond the template pool, aspect suffixes keep questions distinct
    many = questions_for_doc(60)
    assert len(many) == len(set(many)) == 60


def test_overlap_knob_controls_doc_count():
    for overlap, qpd in OVERLAP_QPD.items():
        items = _workload(num_requests=32, prefix_overlap=overlap,
                          synthetic_num_docs=64).build()
        n_docs = len({i.meta["doc_id"] for i in items})
        assert len(items) == 32
        assert n_docs == (32 + qpd - 1) // qpd


def test_build_is_deterministic_given_seed():
    a = _workload().build()
    b = _workload().build()
    assert [i.prompt for i in a] == [i.prompt for i in b]
    c = RagSharedPrefixWorkload(
        {"synthetic_num_docs": 4, "num_requests": 16, "prefix_overlap": "mid"},
        seed=99,
    ).build()
    assert [i.prompt for i in a] != [i.prompt for i in c]  # order reshuffled
    assert {i.prompt for i in a} == {i.prompt for i in c}  # same content


def test_synthetic_docs_stable_across_seeds():
    assert synthetic_documents(3) == synthetic_documents(3)


# -- the byte-identical / token-ID guarantee (HARNESS_SPEC §7 + §10) ----------

def test_shared_prefix_token_ids_across_same_document_batch():
    items = _workload(num_requests=24, prefix_overlap="high",
                      synthetic_num_docs=2).build()
    groups_checked = check_shared_prefix_token_ids(items, ByteTokenizer())
    assert groups_checked == len({i.meta["doc_id"] for i in items})


def test_token_check_catches_boundary_merging_tokenizer():
    items = _workload(num_requests=8, prefix_overlap="high").build()
    with pytest.raises(AssertionError, match="diverge inside the shared prefix"):
        check_shared_prefix_token_ids(items, BoundaryMergingTokenizer())


def test_token_check_catches_non_identical_prefixes():
    items = _workload(num_requests=8, prefix_overlap="high").build()
    items[0].meta["prefix"] = items[0].meta["prefix"] + " "
    with pytest.raises(AssertionError, match="not byte-identical"):
        check_shared_prefix_token_ids(items, ByteTokenizer())


# -- data sources --------------------------------------------------------------

def test_spec_bench_turn_split():
    doc, q = split_spec_bench_rag_turn("Passage line 1.\nPassage line 2.\nWhat is X?")
    assert doc == "Passage line 1.\nPassage line 2."
    assert q == "What is X?"


def test_spec_bench_file_source(tmp_path):
    rows = [
        {"question_id": 1, "category": "rag",
         "turns": ["Long passage text here.\nWhat does it say?"]},
        {"question_id": 2, "category": "math", "turns": ["2+2?"]},
    ]
    path = tmp_path / "question.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    wl = _workload(spec_bench_file=str(path), synthetic_num_docs=None,
                   num_requests=4, prefix_overlap="mid")
    wl.params.pop("synthetic_num_docs")
    items = wl.build()
    assert len(items) == 4
    assert all("Long passage text here." in i.meta["prefix"] for i in items)
    # original NQ question is included among the per-doc questions
    assert any(i.meta["question"] == "What does it say?" for i in items)


def test_questions_file_source(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text(json.dumps({"document": "Doc A.", "question": "QA?"}))
    wl = _workload(questions_file=str(path), num_requests=6, prefix_overlap="mid")
    items = wl.build()
    assert len(items) == 6
    assert all(i.meta["doc_id"] == 0 for i in items)


def test_score_reports_no_accuracy():
    wl = _workload(num_requests=4)
    items = wl.build()
    score = wl.score(items, ["answer"] * 3 + ["  "])
    assert score.accuracy is None
    assert score.details[0]["num_empty_outputs"] == 1
