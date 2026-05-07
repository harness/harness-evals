"""TurnTokenCostMetric — scores average per-turn token usage across assistant messages."""

from __future__ import annotations

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score


class TurnTokenCostMetric(BaseMetric):
    """Score based on per-turn token count from Message.token_count.

    Reads token_count from each assistant Message. Scores each turn as
    max(0, 1 - token_count / max_tokens_per_turn), then returns the mean.
    Turns with no token_count are skipped. Returns 0.0 if no turns have data.
    """

    def __init__(
        self,
        max_tokens_per_turn: int = 2000,
        threshold: float = 0.5,
        **kwargs: object,
    ) -> None:
        super().__init__(
            name="turn_token_cost",
            dimension=Dimension.PERFORMANCE,
            threshold=threshold,
            **kwargs,
        )
        self.max_tokens_per_turn = max_tokens_per_turn

    def measure(self, eval_case: EvalCase) -> Score:
        messages = eval_case.messages
        if not messages:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="no messages provided",
            )

        token_counts = []
        for msg in messages:
            if msg.role == "assistant" and msg.token_count is not None:
                if msg.token_count < 0:
                    continue
                token_counts.append(msg.token_count)

        if not token_counts:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="no token count data on assistant turns",
            )

        scores = [max(0.0, 1.0 - tc / self.max_tokens_per_turn) for tc in token_counts]
        mean_score = sum(scores) / len(scores)
        mean_token_count = sum(token_counts) / len(token_counts)

        return Score(
            name=self.name,
            value=mean_score,
            threshold=self.threshold,
            metadata={
                "turn_token_counts": token_counts,
                "mean_token_count": mean_token_count,
                "max_tokens_per_turn": self.max_tokens_per_turn,
                "n_turns_scored": len(token_counts),
            },
        )
