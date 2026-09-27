"""StatementCoverageMetric — precomputed coverage fraction from metadata.

Reads a coverage fraction [0, 1] stashed in ``EvalCase.metadata`` by a grading
pipeline (which has already excluded test files per its own rubric). The metric
is a **pure reader** — no subprocess, no I/O.

A stashed value of ``0.0`` is a valid floor score; a *missing* path skips the
metric entirely (inert on ordinary evals). Pipelines that distinguish
"unmeasured" should stash ``0.0`` (floor credit, never absent credit).

Example:
    EvalCase(metadata={"mb_grade": {"app_coverage": 0.83}})
    StatementCoverageMetric(path="mb_grade.app_coverage") -> Score 0.83
"""

from __future__ import annotations

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.metrics.coding._metadata import resolve_metadata


class StatementCoverageMetric(BaseMetric):
    def __init__(self, path: str, name: str | None = None, threshold: float = 1.0) -> None:
        self.path = path
        super().__init__(
            name=name or f"statement_coverage({path})",
            dimension=Dimension.CORRECTNESS,
            threshold=threshold,
        )

    def measure(self, eval_case: EvalCase) -> Score | None:
        node = resolve_metadata(getattr(eval_case, "metadata", None), self.path)
        if node is None:
            return None  # skip: nothing stashed for this path
        try:
            value = float(node)
        except (TypeError, ValueError):
            return None  # skip: uninterpretable node, not a zero Score
        return Score(
            name=self.name,
            value=min(max(value, 0.0), 1.0),
            threshold=self.threshold,
            reason=f"{value:.2%} statement coverage at '{self.path}'",
            metadata={"path": self.path},
        )
