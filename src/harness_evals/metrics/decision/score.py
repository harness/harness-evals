"""ScoreMetric — calibrated ordered-rubric rating decision primitive."""

from __future__ import annotations

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.decision.base import BaseDecisionProvider
from harness_evals.decision.types import ScoreQuestion
from harness_evals.metrics.decision._common import _MISSING, level_norm, resolve_mode, resolve_state


class ScoreMetric(BaseMetric):
    """Rate against an ordered rubric, scored against ``expected`` or self-reported.

    With ``expected`` set (or ``mode="correctness"``): ``expected`` must be a
    numeric rubric level (0-indexed), not a legend string. Value is
    ``1.0 - abs(level_norm(score) - level_norm(expected))``, i.e. partial
    credit for near misses.

    Without ``expected`` (or ``mode="confidence"``): ``value`` is the model's
    self-reported confidence in the score; the raw probability-weighted score
    is preserved in ``Score.metadata``.
    """

    def __init__(
        self,
        provider: BaseDecisionProvider,
        instructions: str | dict | list,
        criteria: list[str | dict | list],
        state_field: str = "output",
        mode: str | None = None,
        threshold: float = 1.0,
        dimension: Dimension = Dimension.CORRECTNESS,
        name: str = "score",
        **kwargs: object,
    ) -> None:
        super().__init__(name=name, dimension=dimension, threshold=threshold, **kwargs)
        self.provider = provider
        self.instructions = instructions
        self.criteria = criteria
        self.state_field = state_field
        self.mode = mode

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

        effective_mode = resolve_mode(self.mode, eval_case)
        question = ScoreQuestion(instructions=self.instructions, criteria=self.criteria)
        response = await self.provider.a_ask(state, {self.name: question})
        answer = response.answers[self.name]
        num_levels = len(self.criteria)

        metadata: dict[str, object] = {
            "score": answer.score,
            "confidence": answer.confidence,
            "legend": answer.legend,
            "probabilities": answer.probabilities,
            "mode": effective_mode,
            "model": response.model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        }

        if effective_mode == "correctness":
            expected = eval_case.expected
            if not isinstance(expected, int | float) or isinstance(expected, bool):
                raise ValueError(f"ScoreMetric expected must be a numeric rubric level (0-indexed), got {expected!r}")
            if not (0 <= expected <= num_levels - 1):
                raise ValueError(
                    f"ScoreMetric expected level {expected} is out of range for {num_levels} criteria "
                    f"(valid range: 0-{num_levels - 1})"
                )
            value = 1.0 - abs(level_norm(answer.score, num_levels) - level_norm(expected, num_levels))
        else:
            value = max(0.0, min(1.0, answer.confidence))

        return Score(name=self.name, value=max(0.0, min(1.0, value)), threshold=self.threshold, metadata=metadata)
