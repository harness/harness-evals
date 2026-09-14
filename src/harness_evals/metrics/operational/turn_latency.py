"""TurnLatencyMetric — scores average per-turn latency across assistant messages."""

from __future__ import annotations

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score


class TurnLatencyMetric(BaseMetric):
    """Binary pass/fail based on per-turn latency from Message.latency_ms.

    Reads latency_ms from each assistant Message. Each scored turn passes if
    latency_ms <= max_ms_per_turn. value = 1.0 only if every scored turn
    passes (all-must-pass), else 0.0. Turns with no latency_ms are skipped.
    Returns 0.0 if no turns have data.
    """

    def __init__(
        self,
        max_ms_per_turn: float = 3000,
        threshold: float = 1.0,
        **kwargs: object,
    ) -> None:
        super().__init__(
            name="turn_latency",
            dimension=Dimension.PERFORMANCE,
            threshold=threshold,
            **kwargs,
        )
        if max_ms_per_turn <= 0:
            raise ValueError(f"max_ms_per_turn must be positive, got {max_ms_per_turn}")
        self.max_ms_per_turn = max_ms_per_turn

    def measure(self, eval_case: EvalCase) -> Score:
        messages = eval_case.messages
        if not messages:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="Cannot measure turn latency — no messages provided in the eval case",
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
                reason="Cannot measure turn latency — no latency data found on any assistant turn",
            )

        per_turn_pass = [lat <= self.max_ms_per_turn for lat in latencies]
        n_turns_failed = sum(1 for p in per_turn_pass if not p)
        value = 1.0 if n_turns_failed == 0 else 0.0
        mean_latency = sum(latencies) / len(latencies)

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=(
                f"{len(latencies) - n_turns_failed}/{len(latencies)} scored turns within "
                f"{self.max_ms_per_turn:g}ms per turn (mean latency {mean_latency:g}ms)"
            ),
            metadata={
                "turn_latencies": latencies,
                "mean_latency_ms": mean_latency,
                "max_ms_per_turn": self.max_ms_per_turn,
                "n_turns_scored": len(latencies),
                "n_turns_failed": n_turns_failed,
            },
        )
