"""ScoreMetric — calibrated ordered-rubric rating decision primitive."""

from __future__ import annotations

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.decision.base import BaseDecisionProvider
from harness_evals.decision.types import ScoreQuestion
from harness_evals.metrics.decision._common import _MISSING, resolve_mode, resolve_state, score_rubric


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

        effective_mode = resolve_mode(self.mode, eval_case.expected)
        question = ScoreQuestion(instructions=self.instructions, criteria=self.criteria)
        response = await self.provider.a_ask(state, {self.name: question})
        answer = response.answers[self.name]
        num_levels = len(self.criteria)

        try:
            value, metadata = score_rubric(answer, effective_mode, eval_case.expected, num_levels)
        except ValueError as e:
            raise ValueError(f"ScoreMetric {e}") from e
        metadata.update(
            {
                "model": response.model,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            }
        )

        return Score(name=self.name, value=value, threshold=self.threshold, metadata=metadata)
