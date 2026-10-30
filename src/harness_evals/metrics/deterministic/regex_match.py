from __future__ import annotations

import re

from harness_evals.applicability import applicability_for_inputs
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension, MetricInput, MetricInputs
from harness_evals.core.score import Score
from harness_evals.metrics.deterministic._negation import resolve_negation


def _resolve_pattern(*, negate: bool, forbidden: str | None, threshold: float) -> tuple[re.Pattern[str] | None, float]:
    forbidden, threshold = resolve_negation(negate=negate, forbidden=forbidden, threshold=threshold)
    if forbidden is None:
        return None, threshold
    try:
        return re.compile(forbidden), threshold
    except re.error as e:
        raise ValueError(f"Invalid forbidden regex pattern — could not be compiled ({e})") from e


class RegexMetric(BaseMetric):
    """Check required or forbidden regex matches in output.

    Positive checks use the pattern in ``EvalCase.expected``. Negative checks
    require a metric-level ``forbidden`` pattern.
    """

    eval_inputs = MetricInputs(required={MetricInput.OUTPUT, MetricInput.EXPECTED})
    catalog_applicability = applicability_for_inputs(MetricInputs(required={MetricInput.OUTPUT}))

    def __init__(
        self,
        threshold: float = 1.0,
        negate: bool = False,
        forbidden: str | None = None,
        **kwargs: object,
    ) -> None:
        _, threshold = _resolve_pattern(negate=negate, forbidden=forbidden, threshold=threshold)

        super().__init__(name="regex", dimension=Dimension.CORRECTNESS, threshold=threshold, **kwargs)
        self.negate = negate
        self.forbidden = forbidden

    @property
    def negate(self) -> bool:
        return self._negate

    @negate.setter
    def negate(self, value: bool) -> None:
        self._negate = value
        inputs = MetricInputs(required={MetricInput.OUTPUT}) if value is True else RegexMetric.eval_inputs
        self._configure_inputs(RegexMetric, inputs)

    def measure(self, eval_case: EvalCase) -> Score:
        # Public options are mutable: resolve the current configuration, not a
        # constructor snapshot. Invalid mutations fail before inspecting evidence.
        negate = self.negate
        forbidden_pattern, threshold = _resolve_pattern(
            negate=negate, forbidden=self.forbidden, threshold=self.threshold
        )
        if eval_case.output is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=threshold,
                reason="No output provided to inspect (output is None)",
            )

        actual = str(eval_case.output)

        if forbidden_pattern is not None:
            match = bool(forbidden_pattern.search(actual))
        else:
            if eval_case.expected is None:
                return Score(
                    name=self.name,
                    value=0.0,
                    threshold=threshold,
                    reason="No regex pattern provided to match against (expected is None)",
                )
            pattern = str(eval_case.expected)
            try:
                match = bool(re.search(pattern, actual))
            except re.error as e:
                return Score(
                    name=self.name,
                    value=0.0,
                    threshold=threshold,
                    reason=f"Invalid regex pattern — could not be compiled ({e})",
                )

        value = 1.0 if match else 0.0
        if negate:
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
            threshold=threshold,
            reason=reason,
        )
