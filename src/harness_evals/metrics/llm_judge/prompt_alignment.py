"""PromptAlignment metric — evaluates whether output follows prompt instructions."""

from __future__ import annotations

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM

_PROMPT_TEMPLATE = """You are an expert evaluator checking whether an LLM output follows specific instructions.

**Input given to the LLM**: {input}
**LLM output**: {output}

**Instructions to verify** (each must be individually checked):
{instructions_text}

For each instruction, determine if the output follows it.

Respond with JSON:
{{"results": [{{"instruction": "the instruction", "followed": true/false, "reasoning": "brief explanation"}}]}}
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["results"],
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["instruction", "followed", "reasoning"],
                "properties": {
                    "instruction": {"type": "string"},
                    "followed": {"type": "boolean"},
                    "reasoning": {"type": "string"},
                },
            },
        }
    },
}


class PromptAlignmentMetric(BaseMetric):
    """Evaluates whether output follows specific prompt instructions.

    Score = instructions_followed / total_instructions.
    Each instruction is independently verified by the LLM judge.
    """

    def __init__(
        self,
        llm: BaseLLM,
        prompt_instructions: list[str],
        threshold: float = 0.5,
        **kwargs: object,
    ) -> None:
        super().__init__(name="prompt_alignment", dimension=Dimension.CORRECTNESS, threshold=threshold, **kwargs)
        self.llm = llm
        self.prompt_instructions = prompt_instructions

        if not prompt_instructions:
            raise ValueError("prompt_instructions must contain at least one instruction")

    def measure(self, eval_case: EvalCase) -> Score:
        return _run_async(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        instructions_text = "\n".join(f"- {inst}" for inst in self.prompt_instructions)

        prompt = _PROMPT_TEMPLATE.format(
            input=eval_case.input,
            output=eval_case.output,
            instructions_text=instructions_text,
        )

        result = await self.llm.generate_json(prompt, _RESPONSE_SCHEMA)
        results = result.get("results", [])

        followed = sum(1 for r in results if r.get("followed", False))
        total = len(self.prompt_instructions)

        value = followed / total if total > 0 else 0.0

        failed = [r for r in results if not r.get("followed", False)]
        reason_parts = [f"FAILED: {r.get('instruction', '?')} — {r.get('reasoning', '')}" for r in failed]

        return Score(
            name=self.name,
            value=max(0.0, min(1.0, value)),
            threshold=self.threshold,
            reason="; ".join(reason_parts) if reason_parts else f"All {total} instructions followed",
            metadata={
                "followed": followed,
                "total": total,
                "results": results,
            },
        )
