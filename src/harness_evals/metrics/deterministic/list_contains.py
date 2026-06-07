"""List contains metric — checks if expected items are present in output list."""

from __future__ import annotations

import json

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score


def _parse_list(value: object) -> list[str]:
    """Coerce value to a list of strings."""
    if isinstance(value, list):
        return [str(item) for item in value]
    s = str(value)
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except (json.JSONDecodeError, TypeError):
        pass
    return [item.strip() for item in s.split(",") if item.strip()]


class ListContainsMetric(BaseMetric):
    """Score based on overlap between expected and output item lists.

    Accepts lists, JSON array strings, or comma-separated strings.

    Modes:
        ``"subset"`` (default): score = fraction of expected items found in output.
        ``"exact"``: score = 1.0 if sets are equal, else Jaccard-like fraction.
    """

    _VALID_MODES = ("subset", "exact")

    def __init__(
        self,
        threshold: float = 1.0,
        mode: str = "subset",
        case_sensitive: bool = True,
        **kwargs: object,
    ) -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(f"mode must be one of {self._VALID_MODES}, got '{mode}'")
        super().__init__(name="list_contains", dimension=Dimension.CORRECTNESS, threshold=threshold, **kwargs)
        self.mode = mode
        self.case_sensitive = case_sensitive

    def _normalize(self, items: list[str]) -> list[str]:
        if self.case_sensitive:
            return items
        return [item.lower() for item in items]

    def measure(self, eval_case: EvalCase) -> Score:
        if eval_case.expected is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="No expected answer provided to compare against (expected is None)",
            )

        output_items = self._normalize(_parse_list(eval_case.output))
        expected_items = self._normalize(_parse_list(eval_case.expected))

        if not expected_items:
            return Score(name=self.name, value=1.0, threshold=self.threshold)

        output_set = set(output_items)
        expected_set = set(expected_items)

        if self.mode == "exact":
            if output_set == expected_set:
                value = 1.0
            else:
                union = output_set | expected_set
                intersection = output_set & expected_set
                value = len(intersection) / len(union) if union else 1.0
        else:
            matched = sum(1 for item in expected_items if item in output_set)
            value = matched / len(expected_items)

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            metadata={"output_items": len(output_items), "expected_items": len(expected_items)},
        )
