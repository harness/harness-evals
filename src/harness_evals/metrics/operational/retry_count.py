from __future__ import annotations

from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.core.test_case import TestCase


class RetryCountMetric(BaseMetric):
    """Score based on retry count from metadata["retry_count"].

    value = max(0, 1 - retry_count / max_retries). 0 retries = 1.0.
    """

    def __init__(self, max_retries: int = 5, threshold: float = 0.5, **kwargs: object) -> None:
        super().__init__(name="retry_count", threshold=threshold, **kwargs)
        self.max_retries = max_retries

    def measure(self, test_case: TestCase) -> Score:
        retries = (test_case.metadata or {}).get("retry_count")
        if retries is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                success=False,
                reason="metadata['retry_count'] not provided",
            )

        retries = int(retries)
        value = max(0.0, 1.0 - retries / self.max_retries)

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            success=value >= self.threshold,
            metadata={"retry_count": retries, "max_retries": self.max_retries},
        )
