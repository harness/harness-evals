"""ComplexityLadderStrategy — generate inputs at varying complexity levels."""

from __future__ import annotations

from harness_evals.core.golden import Golden
from harness_evals.input_generator.base import BaseInputStrategy

__all__ = ["ComplexityLadderStrategy"]

_DEFAULT_LEVELS = ["simple", "moderate", "complex", "expert"]

_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["inputs"],
    "properties": {
        "inputs": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["text"],
                "properties": {
                    "text": {"type": "string"},
                    "expected_output": {"type": "string"},
                },
            },
        },
    },
}


class ComplexityLadderStrategy(BaseInputStrategy):
    """Generate inputs across a spectrum of complexity levels.

    Distributes the requested count evenly across complexity levels
    (e.g., simple, moderate, complex, expert) and generates inputs
    at each level.
    """

    strategy_name = "complexity_ladder"

    def _build_prompt(self, n: int, **kwargs) -> str:
        description = kwargs.get("description", "")
        level = kwargs.get("_current_level", "moderate")
        return (
            "You are an expert at creating evaluation datasets for AI systems.\n\n"
            f"**Use case**: {description}\n\n"
            f"Generate exactly {n} user inputs at **{level}** complexity level.\n\n"
            "Complexity guidelines:\n"
            "- **simple**: Short, direct, single-step requests with clear intent\n"
            "- **moderate**: Multi-part requests, some ambiguity, requires reasoning\n"
            "- **complex**: Multi-step workflows, constraints, context-dependent\n"
            "- **expert**: Domain-specific jargon, edge cases, complex constraints, "
            "requires deep understanding\n\n"
            "Each input should be a complete, self-contained user message.\n\n"
            "For each input, also provide the ideal expected response that a perfect AI "
            "assistant should give. The response complexity and depth should match the "
            "input complexity level.\n\n"
            "Respond with JSON:\n"
            '{"inputs": [{"text": "the user input", "expected_output": "the ideal response"}, ...]}\n'
        )

    def _response_schema(self) -> dict:
        return _RESPONSE_SCHEMA

    def _parse_response(self, response: dict, **kwargs) -> list[Golden]:
        level = kwargs.get("_current_level", "moderate")
        items = response.get("inputs", [])
        return [
            Golden(
                input=item["text"],
                expected=item.get("expected_output"),
                metadata={
                    "strategy": self.strategy_name,
                    "complexity": level,
                },
            )
            for item in items
            if isinstance(item, dict) and item.get("text")
        ]

    async def generate(
        self,
        count: int,
        description: str | None = None,
        seed_inputs: list[str] | None = None,
        **kwargs,
    ) -> list[Golden]:
        """Generate inputs distributed across complexity levels."""
        levels = kwargs.pop("levels", None) or _DEFAULT_LEVELS
        per_level = max(1, count // len(levels))
        remainder = count - per_level * len(levels)

        all_goldens: list[Golden] = []
        for i, level in enumerate(levels):
            level_count = per_level + (1 if i < remainder else 0)
            goldens = await super().generate(
                count=level_count,
                description=description,
                seed_inputs=seed_inputs,
                _current_level=level,
                **kwargs,
            )
            all_goldens.extend(goldens)

        return self._deduplicate(all_goldens)[:count]
