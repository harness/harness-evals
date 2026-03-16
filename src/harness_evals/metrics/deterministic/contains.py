from __future__ import annotations

from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.core.test_case import TestCase


class ContainsMetric(BaseMetric):
    """Score 1.0 if expected_output is contained within actual_output, else 0.0.

    Useful for checking that a response includes a required substring.
    """

    def __init__(self, threshold: float = 1.0, case_sensitive: bool = True, **kwargs: object) -> None:
        super().__init__(name="contains", threshold=threshold, **kwargs)
        self.case_sensitive = case_sensitive

    def measure(self, test_case: TestCase) -> Score:
        actual = str(test_case.actual_output)
        expected = str(test_case.expected_output)

        if not self.case_sensitive:
            actual = actual.lower()
            expected = expected.lower()

        value = 1.0 if expected in actual else 0.0
        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            success=value >= self.threshold,
        )
