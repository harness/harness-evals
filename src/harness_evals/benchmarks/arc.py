"""ARC benchmark: AI2 Reasoning Challenge (Easy + Challenge)."""

from __future__ import annotations

from typing import Any

from harness_evals.benchmarks._answer_utils import extract_choice
from harness_evals.benchmarks.base import BaseBenchmark
from harness_evals.benchmarks.dataset_cache import fetch_hf_dataset


class ARC(BaseBenchmark):
    """ARC: AI2 Reasoning Challenge.

    Multiple-choice science questions at grade-school level.
    Supports both Easy and Challenge subsets.
    """

    def __init__(self, *, subset: str = "ARC-Challenge") -> None:
        """Initialize ARC benchmark.

        Args:
            subset: "ARC-Challenge" or "ARC-Easy".
        """
        if subset not in ("ARC-Challenge", "ARC-Easy"):
            raise ValueError(f"subset must be 'ARC-Challenge' or 'ARC-Easy', got '{subset}'")
        self.subset = subset
        super().__init__(name=f"arc_{subset.lower().replace('-', '_')}")

    async def load_dataset(self, *, offline: bool = False) -> list[dict]:
        return await fetch_hf_dataset("allenai/ai2_arc", split="test", config=self.subset, offline=offline)

    def format_prompt(self, item: dict, *, shots: int = 0) -> str:
        question = item["question"]
        choices = item["choices"]

        labels = choices["label"]
        texts = choices["text"]
        choices_str = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts, strict=True))

        few_shot_prefix = ""
        if shots > 0:
            examples = _FEW_SHOT_EXAMPLES[:shots]
            for ex in examples:
                few_shot_prefix += f"Question: {ex['question']}\n{ex['choices']}\nAnswer: {ex['answer']}\n\n"

        return f"{few_shot_prefix}Question: {question}\n{choices_str}\nAnswer:"

    def score_response(self, item: dict, response: str) -> tuple[float, str | None]:
        expected = item["answerKey"]
        labels = item["choices"]["label"]
        predicted = extract_choice(response, choices=labels)

        if predicted is None:
            return 0.0, "Could not parse choice answer"

        if predicted == expected:
            return 1.0, None
        return 0.0, f"Expected {expected}, got {predicted}"

    def _get_expected(self, item: dict) -> str:
        return item["answerKey"]

    def _get_item_metadata(self, item: dict) -> dict[str, Any]:
        return {"subset": self.subset, "id": item.get("id", "")}

    def _get_result_metadata(self) -> dict[str, Any]:
        return {"subset": self.subset}


_FEW_SHOT_EXAMPLES = [
    {
        "question": "Which property of a mineral can be determined just by looking at it?",
        "choices": "A. luster\nB. mass\nC. weight\nD. hardness",
        "answer": "A",
    },
    {
        "question": "A student is given 3 cups of water with different temperatures. Which tool should the student use to find the temperature of each cup?",
        "choices": "A. ruler\nB. thermometer\nC. graduated cylinder\nD. balance",
        "answer": "B",
    },
    {
        "question": "Which of these is a nonrenewable resource?",
        "choices": "A. trees\nB. coal\nC. sunlight\nD. wind",
        "answer": "B",
    },
]
