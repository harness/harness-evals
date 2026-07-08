from __future__ import annotations

import re

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score


class RegexMetric(BaseMetric):
    """Score 1.0 if output matches the regex pattern in expected.

    ``expected`` should be a regex pattern string.
    """

    def __init__(self, threshold: float = 1.0, **kwargs: object) -> None:
        super().__init__(name="regex", dimension=Dimension.CORRECTNESS, threshold=threshold, **kwargs)

    def measure(self, eval_case: EvalCase) -> Score:
        if eval_case.expected is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="No regex pattern provided to match against (expected is None)",
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
                reason=f"Invalid regex pattern — could not be compiled ({e})",
            )

        value = 1.0 if match else 0.0
        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason="Output matched the expected regex pattern" if match else "Output did not match the expected regex pattern",
        )
