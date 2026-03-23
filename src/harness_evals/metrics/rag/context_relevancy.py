"""Context Relevancy metric — how relevant retrieved context is to the question."""

from __future__ import annotations

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM

_PROMPT_TEMPLATE = """For each context chunk below, determine if it is relevant to the given question.
A chunk is "relevant" if it contains information that could help answer the question.

**Question**: {input}

**Context chunks**:
{chunks}

Respond with JSON:
{{"verdicts": [{{"chunk_index": 0, "relevant": true/false}}, ...]}}
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["verdicts"],
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_index": {"type": "integer"},
                    "relevant": {"type": "boolean"},
                },
            },
        }
    },
}


class ContextRelevancyMetric(BaseMetric):
    """Fraction of retrieved context chunks that are relevant to the input question.

    Measures retrieval quality. Score = relevant_chunks / total_chunks.
    Unlike ``ContextPrecisionMetric``, this focuses purely on question-to-context
    relevance without considering the expected answer.
    Requires ``eval_case.context``.
    """

    def __init__(self, llm: BaseLLM, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(name="context_relevancy", threshold=threshold, **kwargs)
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

        chunks_text = "\n".join(f"[Chunk {i}]: {chunk}" for i, chunk in enumerate(eval_case.context))
        prompt = _PROMPT_TEMPLATE.format(input=eval_case.input, chunks=chunks_text)

        result = await self.llm.generate_json(prompt, _RESPONSE_SCHEMA)
        verdicts = result.get("verdicts", [])

        total = len(eval_case.context)
        relevant = sum(1 for v in verdicts if v.get("relevant", False))
        value = relevant / total if total > 0 else 0.0

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=f"{relevant}/{total} context chunks relevant to question",
            metadata={"total_chunks": total, "relevant_chunks": relevant},
        )
