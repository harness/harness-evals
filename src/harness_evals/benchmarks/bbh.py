"""BBH benchmark: BIG-Bench Hard (23 challenging tasks)."""

from __future__ import annotations

import re
from typing import Any

from harness_evals.benchmarks._answer_utils import normalize_text
from harness_evals.benchmarks.base import BaseBenchmark
from harness_evals.benchmarks.dataset_cache import fetch_hf_dataset

BBH_TASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_three_objects",
]


class BBH(BaseBenchmark):
    """BBH: BIG-Bench Hard.

    23 challenging tasks from BIG-Bench that prior language models struggled with.
    Default is 3-shot chain-of-thought evaluation.
    """

    def __init__(self, *, tasks: list[str] | None = None, shots: int = 3) -> None:
        """Initialize BBH benchmark.

        Args:
            tasks: List of BBH task names to evaluate. None = all 23 tasks.
            shots: Default number of few-shot examples (default 3, CoT).
        """
        super().__init__(name="bbh", default_shots=shots)
        if tasks:
            invalid = set(tasks) - set(BBH_TASKS)
            if invalid:
                raise ValueError(f"Unknown BBH tasks: {invalid}")
        self.tasks = tasks or BBH_TASKS
        self._few_shot_examples: dict[str, list[dict]] = {}

    async def load_dataset(self, *, offline: bool = False) -> list[dict]:
        items: list[dict] = []
        for task in self.tasks:
            data = await fetch_hf_dataset("lukaemon/bbh", split="test", config=task, offline=offline)
            for item in data:
                item["task"] = task
            self._few_shot_examples[task] = data[:3]
            items.extend(data[3:])

        return items

    def format_prompt(self, item: dict, *, shots: int = 3) -> str:
        task = item.get("task", "")
        input_text = item.get("input", item.get("question", ""))

        task_display = task.replace("_", " ").title()
        header = f"Task: {task_display}\nThink step by step, then give your final answer after 'The answer is'.\n\n"

        few_shot_prefix = ""
        if shots > 0:
            examples = self._few_shot_examples.get(task, [])[:shots]
            for ex in examples:
                ex_input = ex.get("input", ex.get("question", ""))
                ex_target = ex.get("target", ex.get("answer", ""))
                few_shot_prefix += f"Q: {ex_input}\nA: The answer is {ex_target}\n\n"

        return f"{header}{few_shot_prefix}Q: {input_text}\nA:"

    def score_response(self, item: dict, response: str) -> tuple[float, str | None]:
        expected = item.get("target", item.get("answer", ""))
        predicted = _extract_bbh_answer(response)

        if not expected:
            return 0.0, "No expected answer"

        expected_norm = normalize_text(str(expected))
        predicted_norm = normalize_text(predicted)

        if predicted_norm == expected_norm:
            return 1.0, None

        if expected_norm in predicted_norm or predicted_norm in expected_norm:
            return 1.0, None

        return 0.0, f"Expected '{expected}', got '{predicted}'"

    def _get_expected(self, item: dict) -> str:
        return str(item.get("target", item.get("answer", "")))

    def _get_item_metadata(self, item: dict) -> dict[str, Any]:
        return {"task": item.get("task", "")}

    def _get_result_metadata(self) -> dict[str, Any]:
        return {"tasks": self.tasks, "num_tasks": len(self.tasks)}


def _extract_bbh_answer(response: str) -> str:
    """Extract the final answer from a BBH response."""
    match = re.search(r"(?:the answer is|answer:)\s*(.+?)(?:\.|$)", response, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    lines = response.strip().split("\n")
    return lines[-1].strip()
