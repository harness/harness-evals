from __future__ import annotations

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score


class TokenCostMetric(BaseMetric):
    """Score based on total token usage from eval_case.token_count.

    value = max(0, 1 - token_count / max_tokens). At max_tokens, score = 0.
    """

    def __init__(self, max_tokens: int = 10000, threshold: float = 0.5, **kwargs: object) -> None:
        super().__init__(name="token_cost", threshold=threshold, **kwargs)
        self.max_tokens = max_tokens

    def measure(self, eval_case: EvalCase) -> Score:
        if eval_case.token_count is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="token_count not provided",
            )

        tokens = int(eval_case.token_count)
        value = max(0.0, 1.0 - tokens / self.max_tokens)

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            metadata={"token_count": tokens, "max_tokens": self.max_tokens},
        )
