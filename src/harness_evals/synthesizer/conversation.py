"""Conversation synthesizers — generate ConversationGolden datasets from documents."""

from __future__ import annotations

from harness_evals.conversation.golden import ConversationGolden
from harness_evals.core.golden import Golden
from harness_evals.core.types import Message
from harness_evals.synthesizer.base import BaseSynthesizer

__all__ = ["ConversationSynthesizer", "ScriptedConversationSynthesizer"]

_SCENARIO_PROMPT_TEMPLATE = """\
You are an expert at creating evaluation datasets for conversational AI systems.
Given the following document, generate exactly {n} conversation scenarios.

**Difficulty level**: {difficulty}
- easy: Simple, single-topic conversations.
- medium: Conversations requiring understanding of multiple concepts.
- hard: Complex conversations requiring reasoning across the full document.
- mixed: A mix of easy, medium, and hard scenarios.

**Document**:
{chunk}

Generate exactly {n} conversation scenarios. Each scenario should describe a realistic
user-agent interaction that can be evaluated against the document.

Respond with JSON:
{{"scenarios": [{{"scenario": "description of the scenario", "expected_outcome": "what a successful outcome looks like", "user_persona": "description of the user (optional)"}}]}}
"""

_SCRIPTED_PROMPT_TEMPLATE = """\
You are an expert at creating evaluation datasets for conversational AI systems.
Given the following document, generate exactly {n} scripted multi-turn conversations.

**Difficulty level**: {difficulty}
- easy: Short conversations with simple, direct exchanges.
- medium: Conversations with a few turns covering multiple topics.
- hard: Long conversations requiring complex reasoning across the full document.
- mixed: A mix of easy, medium, and hard conversations.

**Document**:
{chunk}

Generate exactly {n} scripted conversations with realistic turn-by-turn exchanges.

Respond with JSON:
{{"conversations": [{{"scenario": "description", "expected_outcome": "what success looks like", "turns": [{{"role": "user|assistant", "content": "message content"}}]}}]}}
"""

_SCENARIO_SCHEMA: dict = {
    "type": "object",
    "required": ["scenarios"],
    "properties": {
        "scenarios": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["scenario", "expected_outcome"],
                "properties": {
                    "scenario": {"type": "string"},
                    "expected_outcome": {"type": "string"},
                    "user_persona": {"type": "string"},
                },
            },
        }
    },
}

_SCRIPTED_SCHEMA: dict = {
    "type": "object",
    "required": ["conversations"],
    "properties": {
        "conversations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["scenario", "expected_outcome", "turns"],
                "properties": {
                    "scenario": {"type": "string"},
                    "expected_outcome": {"type": "string"},
                    "turns": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["role", "content"],
                            "properties": {
                                "role": {"type": "string"},
                                "content": {"type": "string"},
                            },
                        },
                    },
                },
            },
        }
    },
}


class ConversationSynthesizer(BaseSynthesizer):
    """Generate scenario-based ConversationGolden datasets from source documents."""

    task_type = "conversation"

    def _build_prompt(self, chunk: str, n: int, difficulty: str) -> str:
        return _SCENARIO_PROMPT_TEMPLATE.format(n=n, difficulty=difficulty, chunk=chunk)

    def _response_schema(self) -> dict:
        return _SCENARIO_SCHEMA

    def _parse_response(
        self,
        response: dict,
        doc_index: int,
        difficulty: str,
    ) -> list[Golden]:  # type: ignore[override]
        scenarios = response.get("scenarios", [])
        goldens: list[ConversationGolden] = []
        for item in scenarios:
            scenario = item.get("scenario", "")
            expected_outcome = item.get("expected_outcome", "")
            if not scenario or not expected_outcome:
                continue
            user_persona = item.get("user_persona") or None
            goldens.append(
                ConversationGolden(
                    scenario=scenario,
                    expected_outcome=expected_outcome,
                    user_persona=user_persona,
                    metadata={
                        "task_type": self.task_type,
                        "difficulty": difficulty,
                        "source_document_index": doc_index,
                    },
                )
            )
        return goldens  # type: ignore[return-value]

    @staticmethod
    def _deduplicate(goldens: list[Golden]) -> list[Golden]:  # type: ignore[override]
        """Remove ConversationGoldens with duplicate ``scenario`` values."""
        seen: set[str] = set()
        result: list[Golden] = []
        for g in goldens:
            key = g.scenario if isinstance(g, ConversationGolden) else str(g)  # type: ignore[union-attr]
            if key not in seen:
                seen.add(key)
                result.append(g)
        return result


class ScriptedConversationSynthesizer(BaseSynthesizer):
    """Generate scripted multi-turn ConversationGolden datasets from source documents."""

    task_type = "conversation_scripted"

    def _build_prompt(self, chunk: str, n: int, difficulty: str) -> str:
        return _SCRIPTED_PROMPT_TEMPLATE.format(n=n, difficulty=difficulty, chunk=chunk)

    def _response_schema(self) -> dict:
        return _SCRIPTED_SCHEMA

    def _parse_response(
        self,
        response: dict,
        doc_index: int,
        difficulty: str,
    ) -> list[Golden]:  # type: ignore[override]
        conversations = response.get("conversations", [])
        goldens: list[ConversationGolden] = []
        for item in conversations:
            scenario = item.get("scenario", "")
            expected_outcome = item.get("expected_outcome", "")
            if not scenario or not expected_outcome:
                continue
            raw_turns = item.get("turns", [])
            turns: list[Message] = []
            for turn in raw_turns:
                role = turn.get("role")
                content = turn.get("content", "")
                if not role:
                    continue
                turns.append(Message(role=role, content=content))
            goldens.append(
                ConversationGolden(
                    scenario=scenario,
                    expected_outcome=expected_outcome,
                    turns=turns if turns else None,
                    metadata={
                        "task_type": self.task_type,
                        "difficulty": difficulty,
                        "source_document_index": doc_index,
                    },
                )
            )
        return goldens  # type: ignore[return-value]

    @staticmethod
    def _deduplicate(goldens: list[Golden]) -> list[Golden]:  # type: ignore[override]
        """Remove ConversationGoldens with duplicate ``scenario`` values."""
        seen: set[str] = set()
        result: list[Golden] = []
        for g in goldens:
            key = g.scenario if isinstance(g, ConversationGolden) else str(g)  # type: ignore[union-attr]
            if key not in seen:
                seen.add(key)
                result.append(g)
        return result
