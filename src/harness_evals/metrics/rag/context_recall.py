"""Context Recall metric — fraction of expected claims supported by context."""

from __future__ import annotations

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM

_PROMPT_TEMPLATE = """Determine which statements from the expected answer can be attributed to the provided context.

**Expected answer**: {expected}

**Context**:
{context}

For each statement in the expected answer, determine if it can be attributed to information in the context.

Respond with JSON:
{{"statements": [{{"statement": "the statement", "attributed": true/false}}, ...]}}
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["statements"],
    "properties": {
        "statements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "attributed": {"type": "boolean"},
                },
            },
        }
    },
}


class ContextRecallMetric(BaseMetric):
    """Fraction of claims in expected output that are supported by context.

    Measures whether the retrieved context contains enough information
    to produce the expected answer. Score = attributed / total.
    Requires ``eval_case.expected`` and ``eval_case.context``.
    """

    def __init__(self, llm: BaseLLM, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(name="context_recall", threshold=threshold, **kwargs)
        self.llm = llm

    def measure(self, eval_case: EvalCase) -> Score:
        return _run_async(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        if not eval_case.context:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="No context provided",
            )
        if eval_case.expected is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="No expected output provided for recall check",
            )

        context_text = "\n---\n".join(eval_case.context)
        prompt = _PROMPT_TEMPLATE.format(expected=eval_case.expected, context=context_text)
        result = await self.llm.generate_json(prompt, _RESPONSE_SCHEMA)
        statements = result.get("statements", [])

        if not statements:
            return Score(
                name=self.name,
                value=1.0,
                threshold=self.threshold,
                reason="No statements found in expected output",
            )

        attributed = sum(1 for s in statements if s.get("attributed", False))
        total = len(statements)
        value = attributed / total

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=f"{attributed}/{total} expected statements attributed to context",
            metadata={"total_statements": total, "attributed_statements": attributed},
        )
