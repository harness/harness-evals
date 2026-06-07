"""Faithfulness metric — checks if claims in output are supported by context."""

from __future__ import annotations

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM

_CLAIMS_PROMPT = """Extract all factual claims from the following text. Return each claim as a separate item.

**Text**: {text}

Respond with JSON:
{{"claims": ["claim 1", "claim 2", ...]}}
"""

_VERIFY_PROMPT = """For each claim below, determine if it is supported by the provided context.
A claim is "supported" if the context contains information that confirms or implies it.

**Context**:
{context}

**Claims**:
{claims}

For each claim, respond with "supported" or "unsupported".

Respond with JSON:
{{"verdicts": [{{"claim": "the claim", "verdict": "supported" or "unsupported"}}]}}
"""

_CLAIMS_SCHEMA = {
    "type": "object",
    "required": ["claims"],
    "properties": {"claims": {"type": "array", "items": {"type": "string"}}},
}

_VERIFY_SCHEMA = {
    "type": "object",
    "required": ["verdicts"],
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "verdict": {"type": "string"},
                },
            },
        }
    },
}


class FaithfulnessMetric(BaseMetric):
    """Fraction of claims in output that are supported by context.

    Decomposes the output into atomic claims, then verifies each against
    the retrieved context. Score = supported_claims / total_claims.
    Requires ``eval_case.context`` to be set.
    """

    def __init__(self, llm: BaseLLM, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(name="faithfulness", dimension=Dimension.GROUNDEDNESS, threshold=threshold, **kwargs)
        self.llm = llm

    def measure(self, eval_case: EvalCase) -> Score:
        return _run_async(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        if eval_case.context is None or len(eval_case.context) == 0:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="No context provided — cannot verify faithfulness without reference material",
            )

        # Step 1: Extract claims from output
        claims_result = await self.llm.generate_json(_CLAIMS_PROMPT.format(text=eval_case.output), _CLAIMS_SCHEMA)
        claims = claims_result.get("claims", [])

        if not claims:
            return Score(
                name=self.name,
                value=1.0,
                threshold=self.threshold,
                reason="No factual claims were identified in the output to verify",
            )

        # Step 2: Verify claims against context
        context_text = "\n---\n".join(eval_case.context)
        claims_text = "\n".join(f"- {c}" for c in claims)

        verify_result = await self.llm.generate_json(
            _VERIFY_PROMPT.format(context=context_text, claims=claims_text),
            _VERIFY_SCHEMA,
        )
        verdicts = verify_result.get("verdicts", [])

        supported = sum(1 for v in verdicts if v.get("verdict", "").lower() == "supported")
        total = len(claims)
        value = supported / total

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=f"{supported} of {total} claims in the output are supported by the retrieved context ({supported}/{total})",
            metadata={"total_claims": total, "supported_claims": supported},
        )
