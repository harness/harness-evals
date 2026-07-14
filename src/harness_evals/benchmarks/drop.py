"""DROP benchmark: Discrete Reasoning Over Paragraphs."""

from __future__ import annotations

import re
from typing import Any

from harness_evals.benchmarks._answer_utils import compute_f1, normalize_text
from harness_evals.benchmarks.base import BaseBenchmark
from harness_evals.benchmarks.dataset_cache import fetch_hf_dataset
from harness_evals.llm.base import BaseLLM


class DROP(BaseBenchmark):
    """DROP: Discrete Reasoning Over Paragraphs.

    Reading comprehension requiring numerical reasoning (counting, sorting,
    arithmetic). Scored with both F1 and exact match.
    """

    def __init__(self, *, shots: int = 3) -> None:
        super().__init__(name="drop", default_shots=shots)
        self._few_shot_examples: list[dict] = []
        self._em_scores: list[float] = []
        self._f1_scores: list[float] = []

    async def load_dataset(self, *, offline: bool = False) -> list[dict]:
        data = await fetch_hf_dataset("ucinlp/drop", split="validation", offline=offline)

        self._few_shot_examples = data[:3]
        return data[3:]

    def format_prompt(self, item: dict, *, shots: int = 3) -> str:
        passage = item["passage"]
        question = item["question"]

        header = "Answer the following question based on the passage. Give a short, precise answer.\n\n"

        few_shot_prefix = ""
        if shots > 0:
            examples = self._few_shot_examples[:shots]
            for ex in examples:
                answer = _get_drop_answer_str(ex)
                few_shot_prefix += (
                    f"Passage: {ex['passage'][:200]}...\nQuestion: {ex['question']}\nAnswer: {answer}\n\n"
                )

        return f"{header}{few_shot_prefix}Passage: {passage}\nQuestion: {question}\nAnswer:"

    def score_response(self, item: dict, response: str) -> tuple[float, str | None]:
        expected_answers = _get_all_answers(item)
        predicted = _clean_response(response)

        if not expected_answers:
            return 0.0, "No expected answers available"

        best_em = 0.0
        best_f1 = 0.0

        for expected in expected_answers:
            if normalize_text(predicted) == normalize_text(expected):
                best_em = 1.0
                best_f1 = 1.0
                break

            f1 = compute_f1(predicted, expected)
            best_f1 = max(best_f1, f1)

            pred_nums = re.findall(r"-?\d+\.?\d*", predicted)
            exp_nums = re.findall(r"-?\d+\.?\d*", expected)
            if pred_nums and exp_nums and pred_nums[-1] == exp_nums[-1]:
                best_em = 1.0
                best_f1 = 1.0
                break

        value = max(best_em, best_f1)

        reason = None
        if value < 1.0:
            reason = f"Expected one of {expected_answers[:3]}, got '{predicted}'"

        # Store both metrics in reason metadata for aggregation
        self._last_em = best_em
        self._last_f1 = best_f1
        self._em_scores.append(best_em)
        self._f1_scores.append(best_f1)

        return value, reason

    async def run(self, model: BaseLLM, **kwargs: Any) -> Any:
        self._em_scores: list[float] = []
        self._f1_scores: list[float] = []
        return await super().run(model, **kwargs)

    def _compute_metrics(self, scores: list) -> dict[str, float]:
        """Report both F1 and exact match aggregates.

        Uses total scores count (not just _em_scores length) so that items
        that errored before score_response() count as zeros in the averages.
        """
        total = len(scores)
        if total == 0:
            return {}
        return {
            "exact_match": sum(self._em_scores) / total,
            "f1": sum(self._f1_scores) / total,
        }

    def _get_expected(self, item: dict) -> str:
        return _get_drop_answer_str(item)

    def _get_item_metadata(self, item: dict) -> dict[str, Any]:
        return {"question": item["question"], "section_id": item.get("section_id", "")}


def _get_all_answers(item: dict) -> list[str]:
    """Extract all valid answer strings from a DROP item."""
    answers_info = item.get("answers_spans", item.get("answer", {}))

    if isinstance(answers_info, dict):
        spans = answers_info.get("spans", [])
        if spans:
            return spans

        number = answers_info.get("number", "")
        if number:
            return [str(number)]

        date = answers_info.get("date", {})
        if date:
            parts = [str(v) for v in date.values() if v]
            if parts:
                return [" ".join(parts)]

    if isinstance(answers_info, list):
        return [str(a) for a in answers_info if a]

    return []


def _get_drop_answer_str(item: dict) -> str:
    """Get a single answer string for display."""
    answers = _get_all_answers(item)
    return answers[0] if answers else ""


def _clean_response(response: str) -> str:
    """Clean model response for comparison."""
    text = response.strip()
    text = re.sub(r"^(the answer is|answer:)\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().rstrip(".")
    return text
