from __future__ import annotations

from abc import ABC, abstractmethod

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score


class BaseMetric(ABC):
    """Base class for all evaluation metrics.

    Subclasses implement ``measure()`` (sync) which takes an EvalCase and
    returns a Score. For I/O-bound metrics (LLM-judged), override
    ``a_measure()`` instead — the default calls ``measure()`` synchronously.
    """

    def __init__(self, name: str, threshold: float = 1.0, **kwargs: object) -> None:
        self.name = name
        self.threshold = threshold

    @abstractmethod
    def measure(self, eval_case: EvalCase) -> Score:
        """Evaluate the case and return a Score. Sync — override for deterministic metrics."""
        ...

    async def a_measure(self, eval_case: EvalCase) -> Score:
        """Async variant. Override for I/O-bound metrics (LLM-judged). Default calls measure()."""
        return self.measure(eval_case)


class ReliabilityMetric(BaseMetric):
    """Base class for metrics that evaluate across multiple runs.

    Expects ``eval_case.runs`` to contain K repeated runs of the same task.
    Subclasses implement ``measure_runs()`` instead of ``measure()``.
    """

    def __init__(self, name: str, threshold: float = 1.0, k: int = 5, **kwargs: object) -> None:
        super().__init__(name=name, threshold=threshold, **kwargs)
        self.k = k

    @abstractmethod
    def measure_runs(self, eval_case: EvalCase) -> Score:
        """Evaluate across eval_case.runs. Called by measure() when runs are present."""
        ...

    async def a_measure_runs(self, eval_case: EvalCase) -> Score:
        """Async variant of measure_runs(). Default calls measure_runs()."""
        return self.measure_runs(eval_case)

    def measure(self, eval_case: EvalCase) -> Score:
        if eval_case.runs:
            return self.measure_runs(eval_case)
        return Score(
            name=self.name,
            value=0.0,
            threshold=self.threshold,
            reason=f"No runs provided (expected {self.k})",
        )

    async def a_measure(self, eval_case: EvalCase) -> Score:
        if eval_case.runs:
            return await self.a_measure_runs(eval_case)
        return Score(
            name=self.name,
            value=0.0,
            threshold=self.threshold,
            reason=f"No runs provided (expected {self.k})",
        )
