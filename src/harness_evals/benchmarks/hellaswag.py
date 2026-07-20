"""HellaSwag benchmark: commonsense reasoning about physical situations."""

from __future__ import annotations

from typing import Any

from harness_evals.benchmarks._answer_utils import extract_choice
from harness_evals.benchmarks.base import BaseBenchmark
from harness_evals.benchmarks.dataset_cache import fetch_hf_dataset


class HellaSwag(BaseBenchmark):
    """HellaSwag: Commonsense NLI about physical situations.

    Given a context and activity label, choose the most plausible continuation
    from four options.
    """

    def __init__(self) -> None:
        super().__init__(name="hellaswag")

    async def load_dataset(self, *, offline: bool = False) -> list[dict]:
        return await fetch_hf_dataset("Rowan/hellaswag", split="validation", offline=offline)

    def format_prompt(self, item: dict, *, shots: int = 0) -> str:
        ctx = item.get("ctx", item.get("context", ""))
        activity = item.get("activity_label", "")
        endings = item["endings"]

        choices_str = "\n".join(f"{chr(65 + i)}. {ending}" for i, ending in enumerate(endings))

        few_shot_prefix = ""
        if shots > 0:
            examples = _FEW_SHOT_EXAMPLES[:shots]
            for ex in examples:
                few_shot_prefix += (
                    f"Activity: {ex['activity']}\nContext: {ex['context']}\n{ex['choices']}\nAnswer: {ex['answer']}\n\n"
                )

        prefix = f"Activity: {activity}\n" if activity else ""
        return f"{few_shot_prefix}{prefix}Context: {ctx}\nWhich ending is most plausible?\n{choices_str}\nAnswer:"

    def score_response(self, item: dict, response: str) -> tuple[float, str | None]:
        label = item["label"]
        expected = str(label) if isinstance(label, int) else label

        expected_letter = chr(65 + int(expected)) if expected.isdigit() else expected.upper()

        num_endings = len(item["endings"])
        valid_choices = [chr(65 + i) for i in range(num_endings)]
        predicted = extract_choice(response, choices=valid_choices)

        if predicted is None:
            return 0.0, "Could not parse choice answer"

        if predicted == expected_letter:
            return 1.0, None
        return 0.0, f"Expected {expected_letter}, got {predicted}"

    def _get_expected(self, item: dict) -> str:
        label = item["label"]
        if isinstance(label, int) or (isinstance(label, str) and label.isdigit()):
            return chr(65 + int(label))
        return str(label).upper()

    def _get_item_metadata(self, item: dict) -> dict[str, Any]:
        return {"activity_label": item.get("activity_label", "")}


_FEW_SHOT_EXAMPLES = [
    {
        "activity": "Roof shingle removal",
        "context": "A man is sitting on a roof. He starts pulling up roofing on a roof.",
        "choices": "A. He is using a brush to scrub the shingles.\nB. He starts pulling up shingles and dumping them off the roof.\nC. He starts to climb back down the roof.\nD. He is ripping off the roof.",
        "answer": "B",
    },
    {
        "activity": "Doing crunches",
        "context": "A lady walks to a mat and does a set of crunches.",
        "choices": "A. She does a set of pushups.\nB. She sits down on the mat.\nC. She does a cartwheel.\nD. She does a second set of crunches.",
        "answer": "D",
    },
]
