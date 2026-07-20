from __future__ import annotations

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score


class NumericDiffMetric(BaseMetric):
    """Score based on relative closeness of two numeric values.

    value = 1.0 - (|actual - expected| / max(|expected|, epsilon))
    Clamped to [0.0, 1.0].
    """

    def __init__(self, threshold: float = 0.95, epsilon: float = 1e-9, **kwargs: object) -> None:
        super().__init__(name="numeric_diff", dimension=Dimension.CORRECTNESS, threshold=threshold, **kwargs)
        self.epsilon = epsilon

    def measure(self, eval_case: EvalCase) -> Score:
        try:
            actual = float(eval_case.output)  # type: ignore[arg-type]
            expected = float(eval_case.expected)  # type: ignore[arg-type]
        except (TypeError, ValueError) as e:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"Cannot parse output or expected value as numbers ({e})",
            )

        diff = abs(actual - expected)
        denominator = max(abs(expected), self.epsilon)
        value = max(0.0, 1.0 - diff / denominator)

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=f"Numeric difference was {diff:g} relative to denominator {denominator:g}",
        )
