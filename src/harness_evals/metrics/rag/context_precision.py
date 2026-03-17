"""Context Precision metric — fraction of retrieved context chunks relevant to the input."""

from __future__ import annotations

import asyncio

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM

_PROMPT_TEMPLATE = """For each context chunk below, determine if it is relevant to answering the question.
A chunk is "relevant" if it contains information useful for answering the question.

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


class ContextPrecisionMetric(BaseMetric):
    """Fraction of retrieved context chunks that are relevant to the input.

    Measures retrieval quality — how much of the context is actually useful.
    Score = relevant_chunks / total_chunks. Requires ``eval_case.context``.
    """

    def __init__(self, llm: BaseLLM, threshold: float = 0.5, **kwargs: object) -> None:
        super().__init__(name="context_precision", threshold=threshold, **kwargs)
        self.llm = llm

    def measure(self, eval_case: EvalCase) -> Score:
        return asyncio.run(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        if not eval_case.context:
            return Score(
                name=self.name, value=0.0, threshold=self.threshold,
                reason="No context provided",
            )

        chunks_text = "\n".join(
            f"[Chunk {i}]: {chunk}" for i, chunk in enumerate(eval_case.context)
        )

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
            reason=f"{relevant}/{total} context chunks relevant",
            metadata={"total_chunks": total, "relevant_chunks": relevant},
        )
