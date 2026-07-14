"""GSM8K benchmark: Grade School Math word problems."""

from __future__ import annotations

from typing import Any

from harness_evals.benchmarks._answer_utils import extract_number
from harness_evals.benchmarks.base import BaseBenchmark
from harness_evals.benchmarks.dataset_cache import fetch_hf_dataset


class GSM8K(BaseBenchmark):
    """GSM8K: Grade School Math 8K.

    Math word problems requiring multi-step reasoning.
    Default is 8-shot chain-of-thought evaluation.
    """

    def __init__(self, *, shots: int = 8) -> None:
        super().__init__(name="gsm8k", default_shots=shots)
        self._few_shot_examples: list[dict] = []

    async def load_dataset(self, *, offline: bool = False) -> list[dict]:
        data = await fetch_hf_dataset("openai/gsm8k", split="test", config="main", offline=offline)

        train = await fetch_hf_dataset("openai/gsm8k", split="train", config="main", offline=offline)
        self._few_shot_examples = train[:8]

        return data

    def format_prompt(self, item: dict, *, shots: int = 8) -> str:
        question = item["question"]

        header = "Solve the following math problem step by step. End with the final numeric answer after ####.\n\n"

        few_shot_prefix = ""
        if shots > 0:
            examples = self._few_shot_examples[:shots]
            for ex in examples:
                few_shot_prefix += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"

        return f"{header}{few_shot_prefix}Question: {question}\nAnswer:"

    def score_response(self, item: dict, response: str) -> tuple[float, str | None]:
        expected_answer = _extract_gsm8k_answer(item["answer"])
        predicted = extract_number(response)

        if predicted is None:
            return 0.0, "Could not extract numeric answer"

        try:
            pred_val = float(predicted)
            exp_val = float(expected_answer)
        except ValueError:
            if predicted.strip() == expected_answer.strip():
                return 1.0, None
            return 0.0, f"Expected {expected_answer}, got {predicted}"

        if abs(pred_val - exp_val) < 1e-6:
            return 1.0, None
        return 0.0, f"Expected {expected_answer}, got {predicted}"

    def _get_expected(self, item: dict) -> str:
        return _extract_gsm8k_answer(item["answer"])

    def _get_item_metadata(self, item: dict) -> dict[str, Any]:
        return {"question": item["question"]}


def _extract_gsm8k_answer(answer_text: str) -> str:
    """Extract the final numeric answer from GSM8K's format (after ####)."""
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip().replace(",", "")
    nums = extract_number(answer_text)
    return nums if nums else answer_text.strip()
