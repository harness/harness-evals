from __future__ import annotations

from abc import ABC, abstractmethod

from harness_evals.core.score import Score
from harness_evals.core.test_case import TestCase


class BaseMetric(ABC):
    """Base class for all evaluation metrics.

    Subclasses implement measure() which takes a TestCase and returns a Score.
    """

    def __init__(self, name: str, threshold: float = 1.0, **kwargs: object) -> None:
        self.name = name
        self.threshold = threshold

    @abstractmethod
    def measure(self, test_case: TestCase) -> Score:
        """Evaluate the test case and return a Score."""
        ...


class ReliabilityMetric(BaseMetric):
    """Base class for metrics that evaluate across multiple runs.

    Expects test_case.runs to contain K repeated runs of the same task.
    Subclasses implement measure_runs() instead of measure().
    """

    def __init__(self, name: str, threshold: float = 1.0, k: int = 5, **kwargs: object) -> None:
        super().__init__(name=name, threshold=threshold, **kwargs)
        self.k = k

    @abstractmethod
    def measure_runs(self, test_case: TestCase) -> Score:
        """Evaluate across test_case.runs. Called by measure() when runs are present."""
        ...

    def measure(self, test_case: TestCase) -> Score:
        if test_case.runs:
            return self.measure_runs(test_case)
        return Score(
            name=self.name,
            value=0.0,
            threshold=self.threshold,
            success=False,
            reason=f"No runs provided (expected {self.k})",
        )
