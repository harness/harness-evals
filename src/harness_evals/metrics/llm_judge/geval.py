"""GEval metric — LLM scores output against configurable criteria with chain-of-thought."""

from __future__ import annotations

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM

_PROMPT_TEMPLATE = """You are an expert evaluator. Score the following output against the given criteria.

**Criteria**: {criteria}

**Input**: {input}

**Output**: {output}

{expected_section}

First, reason step-by-step about how well the output meets the criteria.
Then provide your score.

Respond with JSON:
{{"reasoning": "your chain-of-thought reasoning", "score": <float between 0.0 and 1.0>}}
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["reasoning", "score"],
    "properties": {
        "reasoning": {"type": "string"},
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


class GEvalMetric(BaseMetric):
    """LLM-judged evaluation using configurable criteria and chain-of-thought.

    The LLM scores the output on a 0.0-1.0 scale with reasoning.
    """

    def __init__(
        self,
        llm: BaseLLM,
        criteria: str = "Is the response accurate, relevant, and complete?",
        threshold: float = 0.7,
        **kwargs: object,
    ) -> None:
        super().__init__(name="geval", threshold=threshold, **kwargs)
        self.llm = llm
        self.criteria = criteria

    def measure(self, eval_case: EvalCase) -> Score:
        return _run_async(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        expected_section = f"**Expected output**: {eval_case.expected}" if eval_case.expected else ""
        prompt = _PROMPT_TEMPLATE.format(
            criteria=self.criteria,
            input=eval_case.input,
            output=eval_case.output,
            expected_section=expected_section,
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
