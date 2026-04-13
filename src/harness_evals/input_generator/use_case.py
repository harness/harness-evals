"""UseCaseStrategy — generate diverse inputs from a use case description."""

from __future__ import annotations

from harness_evals.core.golden import Golden
from harness_evals.input_generator.base import BaseInputStrategy

__all__ = ["UseCaseStrategy"]

_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["inputs"],
    "properties": {
        "inputs": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["text", "category"],
                "properties": {
                    "text": {"type": "string"},
                    "category": {"type": "string"},
                },
            },
        },
    },
}


class UseCaseStrategy(BaseInputStrategy):
    """Generate diverse, realistic user inputs from a use case description."""

    strategy_name = "use_case"

    def _build_prompt(self, n: int, **kwargs) -> str:
        description = kwargs.get("description", "")
        return (
            "You are an expert at creating evaluation datasets for AI systems.\n\n"
            f"**Use case**: {description}\n\n"
            f"Generate exactly {n} diverse, realistic user inputs that someone would "
            "send to an AI system for this use case. Cover different:\n"
            "- Aspects and sub-tasks of the use case\n"
            "- Phrasings and communication styles\n"
            "- Levels of specificity (vague vs detailed)\n"
            "- Common and uncommon scenarios\n\n"
            "Each input should be a complete, self-contained user message.\n\n"
            "Respond with JSON:\n"
            '{"inputs": [{"text": "the user input", "category": "aspect being tested"}, ...]}\n'
        )

    def _response_schema(self) -> dict:
        return _RESPONSE_SCHEMA

    def _parse_response(self, response: dict, **kwargs) -> list[Golden]:
        items = response.get("inputs", [])
        return [
            Golden(
                input=item["text"],
                metadata={
                    "strategy": self.strategy_name,
                    "category": item.get("category", ""),
                },
            )
            for item in items
            if isinstance(item, dict) and item.get("text")
        ]
