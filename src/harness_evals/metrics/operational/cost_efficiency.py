from __future__ import annotations

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score


class CostEfficiencyMetric(BaseMetric):
    """Score based on cost per request from eval_case.cost_usd.

    value = max(0, 1 - cost_usd / max_cost_usd). At max_cost, score = 0.
    """

    def __init__(self, max_cost_usd: float = 0.10, threshold: float = 0.5, **kwargs: object) -> None:
        super().__init__(name="cost_efficiency", threshold=threshold, **kwargs)
        self.max_cost_usd = max_cost_usd

    def measure(self, eval_case: EvalCase) -> Score:
        if eval_case.cost_usd is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="cost_usd not provided",
            )

        cost = float(eval_case.cost_usd)
        value = max(0.0, 1.0 - cost / self.max_cost_usd)

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            metadata={"cost_usd": cost, "max_cost_usd": self.max_cost_usd},
        )
