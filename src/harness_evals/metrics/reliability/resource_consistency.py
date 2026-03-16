from __future__ import annotations

import statistics

from harness_evals.core.metric import ReliabilityMetric
from harness_evals.core.score import Score
from harness_evals.core.test_case import TestCase


class ResourceConsistencyMetric(ReliabilityMetric):
    """Measures consistency of resource usage (tokens, latency) across K runs.

    Maps to C_res from Rabanser et al. Uses coefficient of variation (CV):
    value = max(0, 1 - CV). A CV of 0 means perfectly consistent resource usage.
    Reads metadata["token_usage"] or a configurable metadata key from each run.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        k: int = 5,
        resource_key: str = "token_usage",
        **kwargs: object,
    ) -> None:
        super().__init__(name="resource_consistency", threshold=threshold, k=k, **kwargs)
        self.resource_key = resource_key

    def measure_runs(self, test_case: TestCase) -> Score:
        runs = test_case.runs or []
        if len(runs) < 2:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                success=False,
                reason=f"Need at least 2 runs, got {len(runs)}",
            )

        values: list[float] = []
        for run in runs:
            v = (run.metadata or {}).get(self.resource_key)
            if v is not None:
                values.append(float(v))

        if len(values) < 2:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                success=False,
                reason=f"metadata['{self.resource_key}'] found in {len(values)} of {len(runs)} runs",
            )

        mean = statistics.mean(values)
        if mean == 0:
            cv = 0.0
        else:
            stdev = statistics.stdev(values)
            cv = stdev / mean

        score_value = max(0.0, 1.0 - cv)

        return Score(
            name=self.name,
            value=score_value,
            threshold=self.threshold,
            success=score_value >= self.threshold,
            metadata={
                "k": len(runs),
                "resource_key": self.resource_key,
                "mean": mean,
                "stdev": statistics.stdev(values),
                "cv": cv,
            },
        )
