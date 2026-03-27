from __future__ import annotations

from abc import ABC, abstractmethod

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score


class BaseMetric(ABC):
    """Base class for all evaluation metrics.

    Subclasses implement ``measure()`` (sync) which takes an EvalCase and
    returns a Score — or ``None`` to signal that the metric does not apply
    to this particular case (e.g. missing metadata).  Skipped (``None``)
    results are excluded from aggregation and do not count as failures.

    For I/O-bound metrics (LLM-judged), override ``a_measure()`` instead —
    the default calls ``measure()`` synchronously.
    """

    def __init__(self, name: str, threshold: float = 1.0, **kwargs: object) -> None:
        self.name = name
        self.threshold = threshold

    @abstractmethod
    def measure(self, eval_case: EvalCase) -> Score | None:
        """Evaluate the case and return a Score, or ``None`` to skip."""
        ...

    async def a_measure(self, eval_case: EvalCase) -> Score | None:
        """Async variant. Override for I/O-bound metrics (LLM-judged). Default calls measure()."""
        return self.measure(eval_case)

    def measure_dataset(self, cases: list[EvalCase], outcomes: list[bool]) -> Score | None:
        """Evaluate across a full dataset with known outcomes.

        Override for dataset-level metrics (e.g. calibration, discrimination)
        that need the complete set of cases to compute a meaningful score.
        Returns ``None`` by default, signalling the runner should fall back
        to per-case evaluation.
        """
        return None


class SafetyMetric(BaseMetric):
    """Marker base class for safety metrics.

    Safety metrics are reported separately and never averaged into an overall
    score (hard-constraint design per Rabanser et al.). Use ``isinstance(m,
    SafetyMetric)`` to identify safety metrics programmatically.
    """


class ReliabilityMetric(BaseMetric):
    """Base class for metrics that evaluate across multiple runs.

    Expects ``eval_case.runs`` to contain K repeated runs of the same task.
    Subclasses implement ``measure_runs()`` instead of ``measure()``.
    """

    def __init__(self, name: str, threshold: float = 1.0, k: int = 5, **kwargs: object) -> None:
        super().__init__(name=name, threshold=threshold, **kwargs)
        self.k = k

    @abstractmethod
    def measure_runs(self, eval_case: EvalCase) -> Score | None:
        """Evaluate across eval_case.runs. Called by measure() when runs are present."""
        ...

    async def a_measure_runs(self, eval_case: EvalCase) -> Score | None:
        """Async variant of measure_runs(). Default calls measure_runs()."""
        return self.measure_runs(eval_case)

    def measure(self, eval_case: EvalCase) -> Score | None:
        if eval_case.runs:
            return self.measure_runs(eval_case)
        return Score(
            name=self.name,
            value=0.0,
            threshold=self.threshold,
            reason=f"No runs provided (expected {self.k})",
        )

    async def a_measure(self, eval_case: EvalCase) -> Score | None:
        if eval_case.runs:
            return await self.a_measure_runs(eval_case)
        return Score(
            name=self.name,
            value=0.0,
            threshold=self.threshold,
            reason=f"No runs provided (expected {self.k})",
        )
