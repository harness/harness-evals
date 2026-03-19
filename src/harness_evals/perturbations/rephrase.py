"""PromptRephrase perturbation — LLM-based semantic rephrasing."""

from __future__ import annotations

from harness_evals.llm.base import BaseLLM
from harness_evals.perturbations.base import BasePerturbation

_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["rephrasings"],
    "properties": {
        "rephrasings": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}


def _build_prompt(input_text: str, n: int) -> str:
    return (
        f"Rephrase the following text {n} different ways. "
        "Each rephrasing must preserve the exact same meaning and intent, "
        "but use different wording, sentence structure, or phrasing.\n\n"
        f"**Original text**:\n{input_text}\n\n"
        'Respond with JSON:\n{"rephrasings": ["rephrasing 1", "rephrasing 2", ...]}\n'
    )


class PromptRephrase(BasePerturbation):
    """LLM-based semantic rephrasing — same meaning, different wording.

    Generates semantically equivalent rephrasings of an input prompt using
    an LLM. Useful for testing whether an agent's output is robust to
    variations in how the same question or instruction is phrased.

    Integrates with ``PromptRobustnessMetric`` via its ``perturbation_fn``
    parameter.
    """

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    async def perturb(self, input_text: str, n: int = 5) -> list[str]:
        if not input_text.strip():
            return [input_text] * n

        prompt = _build_prompt(input_text, n)
        result = await self.llm.generate_json(prompt, _RESPONSE_SCHEMA)
        rephrasings = result.get("rephrasings", [])

        if len(rephrasings) < n:
            rephrasings.extend([input_text] * (n - len(rephrasings)))

        return rephrasings[:n]
