"""NoulMetric — calibrated yes/no decision primitive."""

from __future__ import annotations

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension, MetricInput, MetricInputs
from harness_evals.core.score import Score
from harness_evals.decision.base import BaseDecisionProvider
from harness_evals.decision.types import NoulQuestion
from harness_evals.metrics.decision._common import (
    _MISSING,
    decision_inputs,
    resolve_mode,
    resolve_state,
    score_noul,
    validate_expected,
)


class NoulMetric(BaseMetric):
    """Calibrated yes/no decision, scored against ``expected`` or self-reported.

    With ``expected`` set (or ``mode="correctness"``): ``1.0`` if the noul
    probability rounds to the expected bool, else ``0.0``.

    Without ``expected`` (or ``mode="confidence"``): ``value`` is the raw noul
    probability (or ``1 - noul`` when ``invert=True``) — a calibration signal,
    not a pass/fail judgment.
    """

    eval_inputs: MetricInputs = MetricInputs(required={MetricInput.OUTPUT}, optional={MetricInput.EXPECTED})

    def __init__(
        self,
        provider: BaseDecisionProvider,
        instructions: str | dict | list,
        criteria: dict[str, str] | None = None,
        state_field: str = "output",
        mode: str | None = None,
        invert: bool = False,
        threshold: float = 1.0,
        dimension: Dimension = Dimension.CORRECTNESS,
        name: str = "noul",
        **kwargs: object,
    ) -> None:
        super().__init__(name=name, dimension=dimension, threshold=threshold, **kwargs)
        self.provider = provider
        self.instructions = instructions
        self.criteria = criteria
        self.state_field = state_field
        self.mode = mode
        self._configure_inputs(NoulMetric, decision_inputs(state_field, mode))
        self.invert = invert

    def measure(self, eval_case: EvalCase) -> Score:
        return _run_async(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        state = resolve_state(eval_case, self.state_field)
        if state is _MISSING:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"Missing state field '{self.state_field}'",
            )

        effective_mode = resolve_mode(self.mode, eval_case.expected)
        question = NoulQuestion(instructions=self.instructions, criteria=self.criteria)
        validate_expected(question, effective_mode, eval_case.expected)
        response = await self.provider.a_ask(state, {self.name: question})
        answer = response.answers[self.name]

        value, metadata = score_noul(answer, effective_mode, eval_case.expected, self.invert)
        metadata.update(
            {
                "model": response.model,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            }
        )

        return Score(name=self.name, value=value, threshold=self.threshold, metadata=metadata)
