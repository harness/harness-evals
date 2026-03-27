"""Pairwise metric — LLM compares output against expected (A/B comparison)."""

from __future__ import annotations

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM

_PROMPT_TEMPLATE = """You are an expert evaluator. Compare the two responses below and judge which is better.

**Evaluation criteria**: {criteria}

**Input / Task**: {input}

**Response A (candidate)**: {output}

**Response B (reference)**: {expected}

First, reason step-by-step about the quality of each response according to the criteria.
Then decide which response is better, or if they are tied.

Respond with JSON:
{{"reasoning": "your chain-of-thought reasoning", "winner": "A" or "B" or "tie", "score": <float between 0.0 and 1.0>}}

Where score reflects how good Response A is:
- 1.0 = A is clearly better
- 0.5 = tie / equivalent
- 0.0 = B is clearly better
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["reasoning", "winner", "score"],
    "properties": {
        "reasoning": {"type": "string"},
        "winner": {"type": "string"},
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


class PairwiseMetric(BaseMetric):
    """LLM-judged A/B comparison between output and expected.

    The LLM evaluates both responses against configurable criteria and
    returns a score reflecting how good the output (A) is relative to
    the expected (B).
    """

    def __init__(
        self,
        llm: BaseLLM,
        criteria: str = "Overall quality, accuracy, and helpfulness",
        threshold: float = 0.5,
        **kwargs: object,
    ) -> None:
        super().__init__(name="pairwise", threshold=threshold, **kwargs)
        self.llm = llm
        self.criteria = criteria

    def measure(self, eval_case: EvalCase) -> Score:
        return _run_async(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        if eval_case.expected is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="expected is required for pairwise comparison",
            )

        prompt = _PROMPT_TEMPLATE.format(
            criteria=self.criteria,
            input=eval_case.input,
            output=eval_case.output,
            expected=eval_case.expected,
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
            metadata={"winner": result.get("winner", "unknown")},
        )
