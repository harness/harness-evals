"""TurnLatencyMetric — scores average per-turn latency across assistant messages."""

from __future__ import annotations

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score


class TurnLatencyMetric(BaseMetric):
    """Score based on per-turn latency from Message.latency_ms.

    Reads latency_ms from each assistant Message. Scores each turn as
    max(0, 1 - latency_ms / max_ms_per_turn), then returns the mean.
    Turns with no latency_ms are skipped. Returns 0.0 if no turns have data.
    """

    def __init__(
        self,
        max_ms_per_turn: float = 3000,
        threshold: float = 0.5,
        **kwargs: object,
    ) -> None:
        super().__init__(
            name="turn_latency",
            dimension=Dimension.PERFORMANCE,
            threshold=threshold,
            **kwargs,
        )
        self.max_ms_per_turn = max_ms_per_turn

    def measure(self, eval_case: EvalCase) -> Score:
        messages = eval_case.messages
        if not messages:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="no messages provided",
            )

        latencies = []
        for msg in messages:
            if msg.role == "assistant" and msg.latency_ms is not None:
                if msg.latency_ms < 0:
                    continue
                latencies.append(msg.latency_ms)

        if not latencies:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="no latency data on assistant turns",
            )

        scores = [max(0.0, 1.0 - lat / self.max_ms_per_turn) for lat in latencies]
        mean_score = sum(scores) / len(scores)
        mean_latency = sum(latencies) / len(latencies)

        return Score(
            name=self.name,
            value=mean_score,
            threshold=self.threshold,
            metadata={
                "turn_latencies": latencies,
                "mean_latency_ms": mean_latency,
                "max_ms_per_turn": self.max_ms_per_turn,
                "n_turns_scored": len(latencies),
            },
        )
