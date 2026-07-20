from __future__ import annotations

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score


class TokenCostMetric(BaseMetric):
    """Score based on total token usage from eval_case.token_count.

    value = max(0, 1 - token_count / max_tokens). At max_tokens, score = 0.
    """

    def __init__(self, max_tokens: int = 10000, threshold: float = 0.5, **kwargs: object) -> None:
        super().__init__(name="token_cost", dimension=Dimension.PERFORMANCE, threshold=threshold, **kwargs)
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        self.max_tokens = max_tokens

    def measure(self, eval_case: EvalCase) -> Score:
        if eval_case.token_count is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="No token count data provided on this eval case (token_count is None)",
            )

        tokens = int(eval_case.token_count)
        if tokens < 0:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"Token count value is invalid — must be non-negative, got {tokens}",
            )

        value = max(0.0, 1.0 - tokens / self.max_tokens)

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=f"Token count was {tokens} against max {self.max_tokens}",
            metadata={"token_count": tokens, "max_tokens": self.max_tokens},
        )
