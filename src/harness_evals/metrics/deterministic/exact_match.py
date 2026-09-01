from __future__ import annotations

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.metrics.deterministic._negation import resolve_negation


class ExactMatchMetric(BaseMetric):
    """Check required equality or reject a configured forbidden answer.

    Positive checks compare output with ``EvalCase.expected``. Negative checks
    compare output with a metric-level ``forbidden`` answer.

    ``case_sensitive`` governs the negated comparison too, and its ``True``
    default fails *open* there: ``forbidden="BLOCKED"`` passes on the output
    ``"blocked"``. Pass ``case_sensitive=False`` for leak-style checks where a
    recased match should still fail.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        case_sensitive: bool = True,
        negate: bool = False,
        forbidden: str | None = None,
        **kwargs: object,
    ) -> None:
        # ``forbidden: ""`` is meaningful here: "the output must not be empty".
        forbidden, threshold = resolve_negation(
            negate=negate, forbidden=forbidden, threshold=threshold, allow_empty_forbidden=True
        )
        super().__init__(name="exact_match", dimension=Dimension.CORRECTNESS, threshold=threshold, **kwargs)
        self.case_sensitive = case_sensitive
        self.negate = negate
        self.forbidden = forbidden

    def measure(self, eval_case: EvalCase) -> Score:
        if eval_case.output is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="No output provided to compare (output is None)",
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

        matches = actual == comparison
        value = 1.0 if matches else 0.0
        if self.negate:
            value = 1.0 - value
            reason = (
                "Output exactly matched the forbidden answer"
                if matches
                else "Output did not exactly match the forbidden answer"
            )
        else:
            reason = (
                "Output exactly matched expected answer" if matches else "Output did not exactly match expected answer"
            )
        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=reason,
        )
