"""BoolQ benchmark: boolean reading comprehension."""

from __future__ import annotations

import re
from typing import Any

from harness_evals.benchmarks.base import BaseBenchmark
from harness_evals.benchmarks.dataset_cache import fetch_hf_dataset


class BoolQ(BaseBenchmark):
    """BoolQ: Boolean reading comprehension.

    Given a passage and a yes/no question, the model must answer true or false.
    """

    def __init__(self) -> None:
        super().__init__(name="boolq")

    async def load_dataset(self, *, offline: bool = False) -> list[dict]:
        return await fetch_hf_dataset("google/boolq", split="validation", offline=offline)

    def format_prompt(self, item: dict, *, shots: int = 0) -> str:
        passage = item["passage"]
        question = item["question"]

        few_shot_prefix = ""
        if shots > 0:
            examples = _FEW_SHOT_EXAMPLES[:shots]
            for ex in examples:
                few_shot_prefix += f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer: {ex['answer']}\n\n"

        return f"{few_shot_prefix}Passage: {passage}\nQuestion: {question}\nAnswer (true or false):"

    def score_response(self, item: dict, response: str) -> tuple[float, str | None]:
        expected = item["answer"]
        predicted = _extract_bool(response)

        if predicted is None:
            return 0.0, "Could not parse boolean answer"

        if predicted == expected:
            return 1.0, None
        return 0.0, f"Expected {expected}, got {predicted}"

    def _get_expected(self, item: dict) -> str:
        return str(item["answer"]).lower()

    def _get_item_metadata(self, item: dict) -> dict[str, Any]:
        return {"question": item["question"]}


def _extract_bool(response: str) -> bool | None:
    """Extract a boolean answer from model response."""
    text = response.strip().lower()

    if re.search(r"\b(true|yes)\b", text):
        if not re.search(r"\b(false|no)\b", text):
            return True
        last_true = max(text.rfind("true"), text.rfind("yes"))
        last_false = max(text.rfind("false"), text.rfind("no"))
        return last_true > last_false

    if re.search(r"\b(false|no)\b", text):
        return False

    return None


_FEW_SHOT_EXAMPLES = [
    {
        "passage": "Persian (/ˈpɜːrʒən, -ʃən/), also known as Farsi, is a Western Iranian language.",
        "question": "do iran and afghanistan speak the same language",
        "answer": True,
    },
    {
        "passage": "The longest living hummingbird is the buff-bellied hummingbird which has been documented to have a lifespan of 11 years 2 months.",
        "question": "can hummingbirds live for more than 10 years",
        "answer": True,
    },
    {
        "passage": "Windows text editors typically use CRLF line endings.",
        "question": "is the line ending the same on all operating systems",
        "answer": False,
    },
]
