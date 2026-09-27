"""TestSuitePassMetric — pass fraction of a stashed test-suite result.

For code-producing agents whose grading pipeline runs a test suite (authored,
provided, or hidden) and records the outcome into ``EvalCase.metadata``, this
metric scores the pass fraction. It is a **pure reader** — no subprocess, no
I/O — so it is safe inside the runner's unthrottled metric fan-out.

Example:
    EvalCase(metadata={"mb_grade": {"conformance": {"ran": True, "passed": 9, "total": 10}}})
    TestSuitePassMetric(path="mb_grade.conformance") -> Score 0.9

Returns ``None`` (skip) when the path is absent so the metric is inert on
ordinary evals.
"""

from __future__ import annotations

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.metrics.coding._metadata import resolve_metadata, suite_pass_fraction


class TestSuitePassMetric(BaseMetric):
    def __init__(self, path: str, name: str | None = None, threshold: float = 1.0) -> None:
        self.path = path
        super().__init__(
            name=name or f"test_suite_pass({path})",
            dimension=Dimension.CORRECTNESS,
            threshold=threshold,
        )

    def measure(self, eval_case: EvalCase) -> Score | None:
        node = resolve_metadata(getattr(eval_case, "metadata", None), self.path)
        if node is None:
            return None  # skip: no stashed suite result for this path
        parsed = suite_pass_fraction(node)
        if parsed is None:
            return None  # skip: unrecognizable node (misconfigured path)
        fraction, reason = parsed
        return Score(
            name=self.name,
            value=fraction,
            threshold=self.threshold,
            reason=f"suite '{self.path}': {reason}",
            metadata={"path": self.path},
        )
