from __future__ import annotations

import statistics

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import Dimension, ReliabilityMetric
from harness_evals.core.score import Score


class ResourceConsistencyMetric(ReliabilityMetric):
    """Measures consistency of resource usage (tokens, latency) across K runs.

    Maps to C_res from Rabanser et al. Uses coefficient of variation (CV):
    value = max(0, 1 - CV). A CV of 0 means perfectly consistent resource usage.

    Reads a typed field (e.g. ``token_count``) first, then falls back to
    ``metadata[resource_key]`` for custom keys like ``gpu_memory``.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        k: int = 5,
        resource_key: str = "token_count",
        **kwargs: object,
    ) -> None:
        super().__init__(
            name="resource_consistency", dimension=Dimension.PERFORMANCE, threshold=threshold, k=k, **kwargs
        )
        self.resource_key = resource_key

    def _get_resource_value(self, run: EvalCase) -> float | None:
        """Try typed field first, then fall back to metadata."""
        value = getattr(run, self.resource_key, None)
        if value is None:
            value = (run.metadata or {}).get(self.resource_key)
        if value is not None:
            return float(value)
        return None

    def measure_runs(self, eval_case: EvalCase) -> Score:
        runs = eval_case.runs or []
        if len(runs) < 2:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"Cannot measure resource consistency — need at least 2 runs, but only {len(runs)} provided",
            )

        values: list[float] = []
        for run in runs:
            v = self._get_resource_value(run)
            if v is not None:
                values.append(v)

        if len(values) < 2:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"Insufficient data — '{self.resource_key}' was only found in {len(values)} of {len(runs)} runs (need at least 2)",
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
            metadata={
                "k": len(runs),
                "resource_key": self.resource_key,
                "mean": mean,
                "stdev": statistics.stdev(values),
                "cv": cv,
            },
        )
