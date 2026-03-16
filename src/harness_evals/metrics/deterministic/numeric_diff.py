from __future__ import annotations

from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.core.test_case import TestCase


class NumericDiffMetric(BaseMetric):
    """Score based on relative closeness of two numeric values.

    value = 1.0 - (|actual - expected| / max(|expected|, epsilon))
    Clamped to [0.0, 1.0].
    """

    def __init__(self, threshold: float = 0.95, epsilon: float = 1e-9, **kwargs: object) -> None:
        super().__init__(name="numeric_diff", threshold=threshold, **kwargs)
        self.epsilon = epsilon

    def measure(self, test_case: TestCase) -> Score:
        try:
            actual = float(test_case.actual_output)  # type: ignore[arg-type]
            expected = float(test_case.expected_output)  # type: ignore[arg-type]
        except (TypeError, ValueError) as e:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                success=False,
                reason=f"Cannot parse as numbers: {e}",
            )

        diff = abs(actual - expected)
        denominator = max(abs(expected), self.epsilon)
        value = max(0.0, 1.0 - diff / denominator)

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            success=value >= self.threshold,
        )
