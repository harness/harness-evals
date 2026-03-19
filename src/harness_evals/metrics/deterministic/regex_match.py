from __future__ import annotations

import re

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score


class RegexMetric(BaseMetric):
    """Score 1.0 if output matches the regex pattern in expected.

    ``expected`` should be a regex pattern string.
    """

    def __init__(self, threshold: float = 1.0, **kwargs: object) -> None:
        super().__init__(name="regex", threshold=threshold, **kwargs)

    def measure(self, eval_case: EvalCase) -> Score:
        if eval_case.expected is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="expected is None — no regex pattern provided",
            )

        actual = str(eval_case.output)
        pattern = str(eval_case.expected)

        try:
            match = bool(re.search(pattern, actual))
        except re.error as e:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"Invalid regex: {e}",
            )

        value = 1.0 if match else 0.0
        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
        )
