"""Context Entity Recall metric — entity overlap between expected and context."""

from __future__ import annotations

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM

_EXTRACT_PROMPT = """Extract all named entities (people, places, organizations, dates, numbers,
technical terms, and other proper nouns) from the following text.

**Text**: {text}

Respond with JSON:
{{"entities": ["entity1", "entity2", ...]}}
"""

_EXTRACT_SCHEMA = {
    "type": "object",
    "required": ["entities"],
    "properties": {"entities": {"type": "array", "items": {"type": "string"}}},
}


class ContextEntityRecallMetric(BaseMetric):
    """Fraction of entities in expected answer that also appear in context.

    Measures whether the retrieved context contains the key entities
    needed to produce the expected answer.
    Requires ``eval_case.expected`` and ``eval_case.context``.
    """

    def __init__(self, llm: BaseLLM, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(name="context_entity_recall", threshold=threshold, **kwargs)
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
                reason="No expected output provided for entity recall",
            )

        expected_result = await self.llm.generate_json(_EXTRACT_PROMPT.format(text=eval_case.expected), _EXTRACT_SCHEMA)
        expected_entities = expected_result.get("entities", [])

        if not expected_entities:
            return Score(
                name=self.name,
                value=1.0,
                threshold=self.threshold,
                reason="No entities found in expected output",
            )

        context_text = "\n---\n".join(eval_case.context)
        context_result = await self.llm.generate_json(_EXTRACT_PROMPT.format(text=context_text), _EXTRACT_SCHEMA)
        context_entities = context_result.get("entities", [])

        context_lower = {e.lower() for e in context_entities}
        matched = sum(1 for e in expected_entities if e.lower() in context_lower)
        total = len(expected_entities)
        value = matched / total

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=f"{matched}/{total} expected entities found in context",
            metadata={
                "expected_entities": total,
                "matched_entities": matched,
                "context_entities": len(context_entities),
            },
        )
