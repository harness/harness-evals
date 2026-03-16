from __future__ import annotations

import re

from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.core.test_case import TestCase


class RegexMetric(BaseMetric):
    """Score 1.0 if actual_output matches the regex pattern in expected_output.

    expected_output should be a regex pattern string.
    """

    def __init__(self, threshold: float = 1.0, **kwargs: object) -> None:
        super().__init__(name="regex", threshold=threshold, **kwargs)

    def measure(self, test_case: TestCase) -> Score:
        actual = str(test_case.actual_output)
        pattern = str(test_case.expected_output)

        try:
            match = bool(re.search(pattern, actual))
        except re.error as e:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                success=False,
                reason=f"Invalid regex: {e}",
            )

        value = 1.0 if match else 0.0
        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            success=value >= self.threshold,
        )
