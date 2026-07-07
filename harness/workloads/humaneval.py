"""HumanEval workload: instruct-templated code-generation prompts.

Prompt format ported from SpecMQuant's evaluation/humaneval/eval.py
(evalplus-style): the user turn wraps the problem in the instruction
prefix, and the assistant turn is pre-filled up to an open ```python fence
so the model continues directly with code. Decoding protocol matched to
their run scripts: greedy, max_new_tokens=512.

The chat template is applied with the model tokenizer when available
(``params["tokenizer"]`` or transformers in Colab); otherwise a hardcoded
Llama-3 template is used. The leading <|begin_of_text|> is stripped either
way because vLLM's completions endpoint adds BOS itself -- keeping it would
double the BOS token.

Scoring executes generated code against the HumanEval tests in a
subprocess with a timeout. This runs untrusted model output: fine in a
throwaway Colab VM, think twice elsewhere. Disable with
params["run_correctness"] = false.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence

from .base import PromptItem, ScoreResult, Workload

INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)
RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the "
    "problem and passes corresponding tests:"
)
_MAGIC_SPLITTER = "-[[]]-this-is-really-our-highest-priority-[[]]-"

EXEC_TIMEOUT_S = 20.0


def _task_and_response(task_prompt: str) -> "tuple[str, str]":
    user = "%s\n```\n%s\n```\n" % (INSTRUCTION_PREFIX, task_prompt.strip())
    assistant = "%s\n```python\n%s\n```\n" % (RESPONSE_PREFIX, _MAGIC_SPLITTER)
    return user, assistant


def build_prompt(task_prompt: str, tokenizer: Optional[Any] = None) -> str:
    user, assistant = _task_and_response(task_prompt)
    if tokenizer is not None:
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            tokenize=False,
        ).split(_MAGIC_SPLITTER)[0]
        bos = getattr(tokenizer, "bos_token", None)
        if bos and text.startswith(bos):
            text = text[len(bos):]
        return text
    return _llama3_template(user, assistant).split(_MAGIC_SPLITTER)[0]


def _llama3_template(user: str, assistant: str) -> str:
    # Official Llama-3-Instruct format, minus <|begin_of_text|> (see module
    # docstring). Used when transformers isn't installed (local tests).
    return (
        "<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n%s" % (user, assistant)
    )


def extract_code(completion: str) -> str:
    """The prompt ends inside an open ```python fence, so the completion is
    code up to the closing fence (or everything, if the model never closes)."""
    return completion.split("```")[0].rstrip()


def check_candidate(code: str, test: str, entry_point: str) -> bool:
    program = "%s\n\n%s\n\ncheck(%s)\n" % (code, test, entry_point)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(program)
        path = fh.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            timeout=EXEC_TIMEOUT_S,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False


class HumanEvalWorkload(Workload):
    name = "humaneval"
    default_max_new_tokens = 512

    def _load_records(self) -> List[Dict[str, Any]]:
        path = self.params.get("questions_file")
        if path:
            records = []
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records
        from datasets import load_dataset  # deferred: Colab-only dependency

        ds = load_dataset("openai/openai_humaneval", split="test")
        return [
            {
                "task_id": r["task_id"],
                "prompt": r["prompt"],
                "test": r["test"],
                "entry_point": r["entry_point"],
            }
            for r in ds
        ]

    def _tokenizer(self) -> Optional[Any]:
        if "tokenizer" in self.params:
            return self.params["tokenizer"]
        model = self.params.get("tokenizer_model")
        if not model:
            return None
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model)

    def build(self) -> List[PromptItem]:
        tokenizer = self._tokenizer()
        records = self.subsample(self._load_records())
        return [
            PromptItem(
                prompt=build_prompt(r["prompt"], tokenizer=tokenizer),
                meta={
                    "task_id": r.get("task_id"),
                    "test": r["test"],
                    "entry_point": r["entry_point"],
                },
            )
            for r in records
        ]

    def score(self, items: Sequence[PromptItem], outputs: Sequence[str]) -> ScoreResult:
        if not self.params.get("run_correctness", True):
            return ScoreResult(accuracy=None)
        details = []
        n_correct = 0
        for item, output in zip(items, outputs):
            code = extract_code(output or "")
            passed = check_candidate(code, item.meta["test"], item.meta["entry_point"])
            n_correct += int(passed)
            details.append({"task_id": item.meta.get("task_id"), "correct": passed})
        accuracy = n_correct / len(details) if details else None
        return ScoreResult(accuracy=accuracy, details=details)
