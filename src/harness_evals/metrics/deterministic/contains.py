from __future__ import annotations

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score


class ContainsMetric(BaseMetric):
    """Score 1.0 if expected is contained within output, else 0.0.

    Useful for checking that a response includes a required substring.
    """

    def __init__(self, threshold: float = 1.0, case_sensitive: bool = True, **kwargs: object) -> None:
        super().__init__(name="contains", dimension=Dimension.CORRECTNESS, threshold=threshold, **kwargs)
        self.case_sensitive = case_sensitive

    def measure(self, eval_case: EvalCase) -> Score:
        if eval_case.expected is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="No expected answer provided to compare against (expected is None)",
            )

        actual = str(eval_case.output)
        expected = str(eval_case.expected)

        if not self.case_sensitive:
            actual = actual.lower()
            expected = expected.lower()

        value = 1.0 if expected in actual else 0.0
        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=(
                "Output contained the expected substring"
                if value == 1.0
                else "Output did not contain the expected substring"
            ),
        )
