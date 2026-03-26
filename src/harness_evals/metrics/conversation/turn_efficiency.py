"""TurnEfficiency metric — ratio of expected to actual conversation turns."""

from __future__ import annotations

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score


class TurnEfficiencyMetric(BaseMetric):
    """Measures conversation efficiency as ``min(expected_turns / actual_turns, 1.0)``.

    Penalizes conversations that take more turns than expected to reach
    resolution.  Conversations that resolve in fewer turns than expected
    are capped at 1.0 (no bonus for being faster).

    Derives ``actual_turns`` from ``len(eval_case.messages)`` and reads
    ``metadata["expected_turns"]`` (int) for the target.
    """

    def __init__(self, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(name="turn_efficiency", threshold=threshold, **kwargs)

    def measure(self, eval_case: EvalCase) -> Score:
        if not eval_case.messages:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="messages is empty or not provided",
            )

        actual = len(eval_case.messages)
        expected = eval_case.meta("expected_turns")

        if expected is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason='metadata must contain "expected_turns"',
            )

        if not isinstance(expected, (int, float)):
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"expected_turns must be numeric, got {type(expected).__name__}",
            )

        if expected <= 0:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"expected_turns must be > 0, got {expected}",
            )

        value = min(expected / actual, 1.0)

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=f"{expected} expected / {actual} actual = {value:.4f}",
            metadata={"actual_turns": actual, "expected_turns": expected},
        )
