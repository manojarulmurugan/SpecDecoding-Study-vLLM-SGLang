"""GSM8K workload: 8-shot chain-of-thought completion prompts.

Prompt construction and answer extraction are ported verbatim from
SpecMQuant's evaluation/gsm8k/eval.py (which itself borrows from
Guangxuan-Xiao/GSM8K-eval) so the reproduction gate measures the same
prompts SpecMQuant timed. Decoding protocol matched to their run scripts:
greedy, max_new_tokens=256, no stop strings.

Question sources, in priority order:
  1. params["questions_file"]: JSONL with {"question": ..., "answer": ...}
     (answer in GSM8K format, ending "#### <number>").
  2. HuggingFace ``openai/gsm8k`` (config "main", split "test") -- requires
     the ``datasets`` package; used in Colab.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence

from .base import PromptItem, ScoreResult, Workload

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
ANSWER_TRIGGER = "The answer is"

_DEMO = [
    (
        "There are 15 trees in the grove. Grove workers will plant trees in "
        "the grove today. After they are done, there will be 21 trees. "
        "How many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some "
        "more were planted. So there must have been 21 - 15 = 6.",
        "6",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.",
        "5",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, "
        "how many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total "
        "they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.",
        "39",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
        "has 12 lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some "
        "to Denny. So he gave Denny 20 - 12 = 8.",
        "8",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9.",
        "9",
    ),
    (
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. "
        "How many computers are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. "
        "9 + 20 is 29.",
        "29",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, "
        "he had 58 - 23 = 35. After losing 2 more, "
        "he had 35 - 2 = 33 golf balls.",
        "33",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. "
        "How much money does she have left?",
        "Olivia had 23 dollars. "
        "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
        "So she has 23 - 15 dollars left. 23 - 15 is 8.",
        "8",
    ),
]


def create_demo_text(n_shot: int = 8, cot_flag: bool = True) -> str:
    demo_text = ""
    for question, chain, answer in _DEMO[:n_shot]:
        if cot_flag:
            demo_text += (
                "Q: " + question + "\nA: " + chain + " "
                + ANSWER_TRIGGER + " " + answer + ".\n\n"
            )
        else:
            demo_text += (
                "Question: " + question + "\nAnswer: "
                + ANSWER_TRIGGER + " " + answer + ".\n\n"
            )
    return demo_text


def build_prompt(input_text: str, n_shot: int = 8, cot_flag: bool = True) -> str:
    return create_demo_text(n_shot, cot_flag) + "Q: " + input_text + "\n" + "A:"


def extract_answer_from_output(completion: str) -> str:
    match = ANS_RE.search(completion)
    if match:
        return match.group(1).strip().replace(",", "")
    return INVALID_ANS


def clean_answer(model_pred: str) -> str:
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = len(preds) > 1
    pred = preds[1] if answer_flag else preds[-1]
    pred = pred.replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", pred)
    if not numbers:
        return INVALID_ANS
    pred = numbers[0] if answer_flag else numbers[-1]
    if pred.endswith("."):
        pred = pred[:-1]
    return pred


def is_correct(model_answer: str, gt_field: str) -> bool:
    gt_answer = extract_answer_from_output(gt_field)
    assert gt_answer != INVALID_ANS, "ground-truth answer field malformed"
    return model_answer == gt_answer


class Gsm8kWorkload(Workload):
    name = "gsm8k"
    default_max_new_tokens = 256

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

        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [{"question": r["question"], "answer": r["answer"]} for r in ds]

    def build(self) -> List[PromptItem]:
        n_shot = int(self.params.get("n_shot", 8))
        records = self.subsample(self._load_records())
        return [
            PromptItem(
                prompt=build_prompt(r["question"], n_shot=n_shot),
                meta={"answer": r["answer"]},
            )
            for r in records
        ]

    def score(self, items: Sequence[PromptItem], outputs: Sequence[str]) -> ScoreResult:
        details = []
        n_correct = 0
        for item, output in zip(items, outputs):
            pred = clean_answer(output or "")
            correct = is_correct(pred, item.meta["answer"])
            n_correct += int(correct)
            details.append({"pred": pred, "correct": correct})
        accuracy = n_correct / len(details) if details else None
        return ScoreResult(accuracy=accuracy, details=details)
