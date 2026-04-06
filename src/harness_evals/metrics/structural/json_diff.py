from __future__ import annotations

import json
from typing import Any

from deepdiff import DeepDiff

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score


class JsonDiffMetric(BaseMetric):
    """Structural similarity between actual and expected JSON/dict outputs.

    Uses DeepDiff's deep_distance for scoring:
    - 0.0 distance = perfect match -> value = 1.0
    - 1.0 distance = completely different -> value = 0.0

    Both output and expected can be dict, list, or JSON strings.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        ignore_order: bool = True,
        exclude_paths: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(name="json_diff", dimension=Dimension.CORRECTNESS, threshold=threshold, **kwargs)
        self.ignore_order = ignore_order
        self.exclude_paths = exclude_paths

    def _parse(self, value: Any) -> Any:
        if isinstance(value, str):
            return json.loads(value)
        return value

    def measure(self, eval_case: EvalCase) -> Score:
        try:
            actual = self._parse(eval_case.output)
            expected = self._parse(eval_case.expected)
        except (json.JSONDecodeError, TypeError) as e:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"JSON parse error: {e}",
            )

        diff = DeepDiff(
            expected,
            actual,
            ignore_order=self.ignore_order,
            exclude_paths=self.exclude_paths,
            get_deep_distance=True,
            threshold_to_diff_deeper=0,
        )

        distance = diff.get("deep_distance", 0.0)
        value = max(0.0, min(1.0, 1.0 - distance))

        reason = None
        if diff:
            changes = {k: v for k, v in diff.items() if k != "deep_distance"}
            if changes:
                reason = f"Differences: {list(changes.keys())}"

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=reason,
        )
