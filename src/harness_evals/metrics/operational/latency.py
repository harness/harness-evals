from __future__ import annotations

from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.core.test_case import TestCase


class LatencyMetric(BaseMetric):
    """Score based on response latency from metadata["latency_ms"].

    value = max(0, 1 - latency_ms / max_ms). At max_ms, score = 0.
    """

    def __init__(self, max_ms: float = 5000, threshold: float = 0.5, **kwargs: object) -> None:
        super().__init__(name="latency", threshold=threshold, **kwargs)
        self.max_ms = max_ms

    def measure(self, test_case: TestCase) -> Score:
        latency = (test_case.metadata or {}).get("latency_ms")
        if latency is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                success=False,
                reason="metadata['latency_ms'] not provided",
            )

        latency = float(latency)
        value = max(0.0, 1.0 - latency / self.max_ms)

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            success=value >= self.threshold,
            metadata={"latency_ms": latency, "max_ms": self.max_ms},
        )
