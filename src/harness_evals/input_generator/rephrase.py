"""RephraseStrategy — generate variations of existing seed inputs."""

from __future__ import annotations

from harness_evals.core.golden import Golden
from harness_evals.input_generator.base import BaseInputStrategy

__all__ = ["RephraseStrategy"]

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


class RephraseStrategy(BaseInputStrategy):
    """Generate semantic rephrasings of seed inputs.

    Distributes the requested count across seed inputs, then asks the LLM
    to rephrase each seed multiple times.
    """

    strategy_name = "rephrase"

    def _build_prompt(self, n: int, **kwargs) -> str:
        seed_inputs = kwargs.get("seed_inputs") or []
        seeds_text = "\n".join(f"- {s}" for s in seed_inputs)
        return (
            f"Rephrase the following inputs {n} different ways total. "
            "Distribute rephrasings evenly across all seed inputs. "
            "Each rephrasing must preserve the exact same meaning and intent, "
            "but use different wording, sentence structure, or phrasing.\n\n"
            f"**Seed inputs**:\n{seeds_text}\n\n"
            'Respond with JSON:\n{"rephrasings": ["rephrasing 1", "rephrasing 2", ...]}\n'
        )

    def _response_schema(self) -> dict:
        return _RESPONSE_SCHEMA

    def _parse_response(self, response: dict, **kwargs) -> list[Golden]:
        rephrasings = response.get("rephrasings", [])
        return [
            Golden(
                input=text,
                metadata={
                    "strategy": self.strategy_name,
                    "seed_count": len(kwargs.get("seed_inputs") or []),
                },
            )
            for text in rephrasings
            if isinstance(text, str) and text.strip()
        ]
