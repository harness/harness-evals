"""WinoGrande benchmark: commonsense pronoun resolution."""

from __future__ import annotations

import re
from typing import Any

from harness_evals.benchmarks.base import BaseBenchmark
from harness_evals.benchmarks.dataset_cache import fetch_hf_dataset


class WinoGrande(BaseBenchmark):
    """WinoGrande: Commonsense pronoun resolution.

    Given a sentence with a blank, choose which of two options correctly fills it.
    """

    def __init__(self) -> None:
        super().__init__(name="winogrande")

    async def load_dataset(self, *, offline: bool = False) -> list[dict]:
        return await fetch_hf_dataset("allenai/winogrande", split="validation", config="winogrande_xl", offline=offline)

    def format_prompt(self, item: dict, *, shots: int = 0) -> str:
        sentence = item["sentence"]
        option1 = item["option1"]
        option2 = item["option2"]

        few_shot_prefix = ""
        if shots > 0:
            examples = _FEW_SHOT_EXAMPLES[:shots]
            for ex in examples:
                few_shot_prefix += (
                    f"Sentence: {ex['sentence']}\n1. {ex['option1']}\n2. {ex['option2']}\nAnswer: {ex['answer']}\n\n"
                )

        return f"{few_shot_prefix}Sentence: {sentence}\n1. {option1}\n2. {option2}\nAnswer (1 or 2):"

    def score_response(self, item: dict, response: str) -> tuple[float, str | None]:
        expected = item["answer"]
        predicted = _extract_option(response)

        if predicted is None:
            return 0.0, "Could not parse option (1 or 2)"

        if predicted == expected:
            return 1.0, None
        return 0.0, f"Expected {expected}, got {predicted}"

    def _get_expected(self, item: dict) -> str:
        return item["answer"]

    def _get_item_metadata(self, item: dict) -> dict[str, Any] | None:
        return {"sentence": item["sentence"]}


def _extract_option(response: str) -> str | None:
    """Extract 1 or 2 from model response."""
    text = response.strip()

    match = re.search(r"(?:answer is|answer:)\s*([12])", text, re.IGNORECASE)
    if match:
        return match.group(1)

    match = re.search(r"\b([12])\b", text)
    if match:
        return match.group(1)

    return None


_FEW_SHOT_EXAMPLES = [
    {
        "sentence": "The trophy doesn't fit in the brown suitcase because _ is too big.",
        "option1": "the trophy",
        "option2": "the suitcase",
        "answer": "1",
    },
    {
        "sentence": "Joan made sure to thank Susan for all the help she had given. _ was grateful.",
        "option1": "Joan",
        "option2": "Susan",
        "answer": "1",
    },
    {
        "sentence": "The city councilmen refused the demonstrators a permit because _ feared violence.",
        "option1": "the city councilmen",
        "option2": "the demonstrators",
        "answer": "1",
    },
]
