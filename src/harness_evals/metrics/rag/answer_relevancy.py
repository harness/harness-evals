"""Answer Relevancy metric — checks if the output actually answers the input."""

from __future__ import annotations

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM

_PROMPT_TEMPLATE = """You are evaluating whether an answer is relevant to the given question.

**Question**: {input}

**Answer**: {output}

Rate the relevancy of the answer to the question. Consider:
1. Does the answer address the question directly?
2. Is the answer on-topic?
3. Does the answer provide useful information for the question?

Respond with JSON:
{{"reasoning": "your reasoning", "score": <float between 0.0 and 1.0>}}

Where 1.0 = perfectly relevant, 0.0 = completely irrelevant.
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["reasoning", "score"],
    "properties": {
        "reasoning": {"type": "string"},
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


class AnswerRelevancyMetric(BaseMetric):
    """LLM judges whether the output is a relevant answer to the input.

    Score 0.0-1.0 based on how directly and completely the output
    addresses the question/task in the input.
    """

    def __init__(self, llm: BaseLLM, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(name="answer_relevancy", threshold=threshold, **kwargs)
        self.llm = llm

    def measure(self, eval_case: EvalCase) -> Score:
        return _run_async(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        prompt = _PROMPT_TEMPLATE.format(
            input=eval_case.input,
            output=eval_case.output,
        )

        result = await self.llm.generate_json(prompt, _RESPONSE_SCHEMA)
        value = float(result.get("score", 0.0))
        value = max(0.0, min(1.0, value))
        reasoning = result.get("reasoning", "")

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=reasoning,
        )
