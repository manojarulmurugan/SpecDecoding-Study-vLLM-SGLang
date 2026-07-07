from __future__ import annotations

import pytest

from harness.workloads.gsm8k import (
    Gsm8kWorkload,
    build_prompt,
    clean_answer,
    create_demo_text,
    extract_answer_from_output,
    is_correct,
)
from harness.workloads.humaneval import (
    HumanEvalWorkload,
    build_prompt as he_build_prompt,
    check_candidate,
    extract_code,
)


# -- GSM8K -------------------------------------------------------------------

def test_demo_text_has_eight_shots():
    demo = create_demo_text(8)
    assert demo.count("Q: ") == 8
    assert demo.count("The answer is") == 8
    # exact tail of the SpecMQuant/GSM8K-eval demo block
    assert "23 - 15 is 8. The answer is 8.\n\n" in demo


def test_build_prompt_shape():
    prompt = build_prompt("What is 2 + 2?")
    assert prompt.endswith("Q: What is 2 + 2?\nA:")
    assert prompt.startswith("Q: There are 15 trees")


def test_extract_answer_from_ground_truth():
    assert extract_answer_from_output("blah blah\n#### 1,234") == "1234"
    assert extract_answer_from_output("no marker") == "[invalid]"


@pytest.mark.parametrize(
    "pred,expected",
    [
        ("Reasoning... The answer is 42.", "42"),
        ("The answer is 1,234.", "1234"),
        ("I think 7 + 5 = 12", "12"),          # no trigger: last number
        ("The answer is -3.5, obviously", "-3.5"),
        ("no numbers here", "[invalid]"),
        # trigger present: first number after the FIRST trigger wins even if
        # the model rambles into more Q/A pairs afterwards
        ("The answer is 6.\n\nQ: next?\nA: ... The answer is 9.", "6"),
    ],
)
def test_clean_answer(pred, expected):
    assert clean_answer(pred) == expected


def test_is_correct():
    assert is_correct("6", "2 + 4 = 6\n#### 6")
    assert not is_correct("7", "2 + 4 = 6\n#### 6")


def test_gsm8k_workload_from_file(gsm8k_questions_file):
    wl = Gsm8kWorkload({"questions_file": gsm8k_questions_file}, seed=1)
    items = wl.build()
    assert len(items) == 3
    score = wl.score(items, [" The answer is 6."] * 3)
    assert score.accuracy == 1.0
    score = wl.score(items, [" The answer is 6.", " The answer is 5.", ""])
    assert score.accuracy == pytest.approx(1 / 3)


def test_gsm8k_subsample_deterministic(gsm8k_questions_file):
    wl1 = Gsm8kWorkload({"questions_file": gsm8k_questions_file, "num_requests": 2}, seed=7)
    wl2 = Gsm8kWorkload({"questions_file": gsm8k_questions_file, "num_requests": 2}, seed=7)
    assert [i.prompt for i in wl1.build()] == [i.prompt for i in wl2.build()]
    assert len(wl1.build()) == 2


# -- HumanEval ---------------------------------------------------------------

class FakeTokenizer:
    bos_token = "<BOS>"

    def apply_chat_template(self, messages, tokenize=False):
        out = self.bos_token
        for m in messages:
            out += "[%s]%s[/%s]" % (m["role"], m["content"], m["role"])
        return out


def test_humaneval_prompt_with_tokenizer_strips_bos_and_splits():
    prompt = he_build_prompt("def add(a, b):\n    ...", tokenizer=FakeTokenizer())
    assert not prompt.startswith("<BOS>")
    assert prompt.rstrip().endswith("```python")
    assert "self-contained Python script" in prompt


def test_humaneval_prompt_fallback_template():
    prompt = he_build_prompt("def add(a, b):\n    ...")
    assert "<|start_header_id|>assistant<|end_header_id|>" in prompt
    assert not prompt.startswith("<|begin_of_text|>")  # vLLM adds BOS itself
    assert prompt.rstrip().endswith("```python")


def test_extract_code_stops_at_fence():
    completion = "def f():\n    return 1\n```\nSome prose after."
    assert extract_code(completion) == "def f():\n    return 1"
    assert extract_code("def g(): pass") == "def g(): pass"


def test_check_candidate_pass_and_fail():
    test_code = (
        "def check(candidate):\n"
        "    assert candidate(2, 3) == 5\n"
    )
    good = "def add(a, b):\n    return a + b"
    bad = "def add(a, b):\n    return a - b"
    assert check_candidate(good, test_code, "add")
    assert not check_candidate(bad, test_code, "add")


def test_humaneval_workload_end_to_end(tmp_path):
    import json

    rows = [{
        "task_id": "T/0",
        "prompt": "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n",
        "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
        "entry_point": "add",
    }]
    path = tmp_path / "he.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    wl = HumanEvalWorkload({"questions_file": str(path)}, seed=1)
    items = wl.build()
    assert len(items) == 1
    good_output = "def add(a, b):\n    return a + b\n```\ndone"
    assert wl.score(items, [good_output]).accuracy == 1.0
    assert wl.score(items, ["def add(a, b):\n    return 0\n```"]).accuracy == 0.0


def test_humaneval_scoring_can_be_disabled(tmp_path):
    import json

    path = tmp_path / "he.jsonl"
    path.write_text(json.dumps({
        "task_id": "T/0", "prompt": "def f():\n", "test": "x", "entry_point": "f",
    }))
    wl = HumanEvalWorkload(
        {"questions_file": str(path), "run_correctness": False}, seed=1
    )
    assert wl.score(wl.build(), ["anything"]).accuracy is None
