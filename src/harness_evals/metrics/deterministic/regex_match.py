from __future__ import annotations

import re

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.metrics.deterministic._negation import resolve_negation


class RegexMetric(BaseMetric):
    """Check required or forbidden regex matches in output.

    Positive checks use the pattern in ``EvalCase.expected``. Negative checks
    require a metric-level ``forbidden`` pattern.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        negate: bool = False,
        forbidden: str | None = None,
        **kwargs: object,
    ) -> None:
        forbidden, threshold = resolve_negation(negate=negate, forbidden=forbidden, threshold=threshold)

        forbidden_pattern: re.Pattern[str] | None = None
        if forbidden is not None:
            try:
                forbidden_pattern = re.compile(forbidden)
            except re.error as e:
                raise ValueError(f"Invalid forbidden regex pattern — could not be compiled ({e})") from e

        super().__init__(name="regex", dimension=Dimension.CORRECTNESS, threshold=threshold, **kwargs)
        self.negate = negate
        self.forbidden = forbidden
        self._forbidden_pattern = forbidden_pattern

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
            assert self._forbidden_pattern is not None
            match = bool(self._forbidden_pattern.search(actual))
        else:
            if eval_case.expected is None:
                return Score(
                    name=self.name,
                    value=0.0,
                    threshold=self.threshold,
                    reason="No regex pattern provided to match against (expected is None)",
                )
            pattern = str(eval_case.expected)
            try:
                match = bool(re.search(pattern, actual))
            except re.error as e:
                return Score(
                    name=self.name,
                    value=0.0,
                    threshold=self.threshold,
                    reason=f"Invalid regex pattern — could not be compiled ({e})",
                )

        value = 1.0 if match else 0.0
        if self.negate:
            value = 1.0 - value
            reason = (
                "Output matched the forbidden regex pattern"
                if match
                else "Output did not match the forbidden regex pattern"
            )
        else:
            reason = (
                "Output matched the expected regex pattern"
                if match
                else "Output did not match the expected regex pattern"
            )
        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=reason,
        )
