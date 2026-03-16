from __future__ import annotations

from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.core.test_case import TestCase


class TokenCostMetric(BaseMetric):
    """Score based on total token usage from metadata["token_usage"].

    value = max(0, 1 - token_usage / max_tokens). At max_tokens, score = 0.
    """

    def __init__(self, max_tokens: int = 10000, threshold: float = 0.5, **kwargs: object) -> None:
        super().__init__(name="token_cost", threshold=threshold, **kwargs)
        self.max_tokens = max_tokens

    def measure(self, test_case: TestCase) -> Score:
        tokens = (test_case.metadata or {}).get("token_usage")
        if tokens is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                success=False,
                reason="metadata['token_usage'] not provided",
            )

        tokens = int(tokens)
        value = max(0.0, 1.0 - tokens / self.max_tokens)

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            success=value >= self.threshold,
            metadata={"token_usage": tokens, "max_tokens": self.max_tokens},
        )
