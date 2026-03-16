from __future__ import annotations

from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.core.test_case import TestCase


class CostEfficiencyMetric(BaseMetric):
    """Score based on cost per request from metadata["cost_usd"].

    value = max(0, 1 - cost_usd / max_cost_usd). At max_cost, score = 0.
    """

    def __init__(self, max_cost_usd: float = 0.10, threshold: float = 0.5, **kwargs: object) -> None:
        super().__init__(name="cost_efficiency", threshold=threshold, **kwargs)
        self.max_cost_usd = max_cost_usd

    def measure(self, test_case: TestCase) -> Score:
        cost = (test_case.metadata or {}).get("cost_usd")
        if cost is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                success=False,
                reason="metadata['cost_usd'] not provided",
            )

        cost = float(cost)
        value = max(0.0, 1.0 - cost / self.max_cost_usd)

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            success=value >= self.threshold,
            metadata={"cost_usd": cost, "max_cost_usd": self.max_cost_usd},
        )
