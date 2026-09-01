from __future__ import annotations

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.metrics.deterministic._negation import resolve_negation


class ContainsMetric(BaseMetric):
    """Check required or forbidden substring presence in output.

    Positive checks compare output with ``EvalCase.expected``. Negative checks
    require a metric-level ``forbidden`` substring so dataset expectations can
    continue to hold the row's ground-truth answer.

    ``case_sensitive`` governs the negated comparison too, and its ``True``
    default fails *open* there: ``forbidden="internal-system-prompt"`` passes on
    the output ``"Internal-System-Prompt: ..."``. Pass ``case_sensitive=False``
    for leak-style checks where a recased match should still fail.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        case_sensitive: bool = True,
        negate: bool = False,
        forbidden: str | None = None,
        **kwargs: object,
    ) -> None:
        forbidden, threshold = resolve_negation(negate=negate, forbidden=forbidden, threshold=threshold)
        super().__init__(name="contains", dimension=Dimension.CORRECTNESS, threshold=threshold, **kwargs)
        self.case_sensitive = case_sensitive
        self.negate = negate
        self.forbidden = forbidden

    def measure(self, eval_case: EvalCase) -> Score:
        if eval_case.output is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="No output provided to inspect (output is None)",
            )

        actual = str(eval_case.output)
        if self.negate:
            comparison = self.forbidden
        elif eval_case.expected is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="No expected answer provided to compare against (expected is None)",
            )
        else:
            comparison = str(eval_case.expected)

        assert comparison is not None

        if not self.case_sensitive:
            actual = actual.lower()
            comparison = comparison.lower()

        contains = comparison in actual
        value = 1.0 if contains else 0.0
        if self.negate:
            value = 1.0 - value
            reason = (
                "Output contained the forbidden substring"
                if contains
                else "Output did not contain the forbidden substring"
            )
        else:
            reason = (
                "Output contained the expected substring"
                if contains
                else "Output did not contain the expected substring"
            )
        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=reason,
        )
