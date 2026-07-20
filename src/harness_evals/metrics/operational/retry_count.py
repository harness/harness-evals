from __future__ import annotations

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score


class RetryCountMetric(BaseMetric):
    """Score based on retry count from eval_case.retry_count.

    value = max(0, 1 - retry_count / max_retries). 0 retries = 1.0.
    """

    def __init__(self, max_retries: int = 5, threshold: float = 0.5, **kwargs: object) -> None:
        super().__init__(name="retry_count", dimension=Dimension.PERFORMANCE, threshold=threshold, **kwargs)
        if max_retries <= 0:
            raise ValueError(f"max_retries must be positive, got {max_retries}")
        self.max_retries = max_retries

    def measure(self, eval_case: EvalCase) -> Score:
        if eval_case.retry_count is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="No retry count data provided on this eval case (retry_count is None)",
            )

        retries = int(eval_case.retry_count)
        if retries < 0:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"Retry count value is invalid — must be non-negative, got {retries}",
            )

        value = max(0.0, 1.0 - retries / self.max_retries)

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=f"Retry count was {retries} against max {self.max_retries}",
            metadata={"retry_count": retries, "max_retries": self.max_retries},
        )
