"""Protocol-agnostic human-input simulation for conversational evals."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from harness_evals.conversation.golden import ConversationGolden
from harness_evals.core.types import Message
from harness_evals.llm.base import BaseLLM
from harness_evals.logging_config import compact_json

_logger = logging.getLogger(__name__)

_HUMAN_INPUT_SCHEMA = {
    "type": "object",
    "required": ["response"],
    "properties": {
        "response": {"type": "object"},
    },
}


@dataclass
class PendingHumanInput:
    """Neutral pending human-input shape from an agent turn."""

    type: str
    payload: dict
    correlation_id: str | None = None

    @classmethod
    def from_metadata(cls, raw: dict) -> PendingHumanInput:
        payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else raw
        correlation_id = raw.get("correlation_id") or raw.get("review_id")
        if correlation_id is None and isinstance(payload, dict):
            correlation_id = payload.get("review_id")
        return cls(
            type=str(raw.get("type") or ""),
            payload=payload if isinstance(payload, dict) else {},
            correlation_id=str(correlation_id) if correlation_id is not None else None,
        )

    def to_metadata(self) -> dict:
        data: dict[str, Any] = {
            "type": self.type,
            "payload": self.payload,
        }
        if self.correlation_id is not None:
            data["correlation_id"] = self.correlation_id
        return data


class ElicitationAdapter(ABC):
    """Build a protocol-specific human-input response for the next agent call."""

    @abstractmethod
    async def respond(
        self,
        pending: PendingHumanInput,
        golden: ConversationGolden,
        history: list[Message],
    ) -> dict:
        """Return an opaque response dict the target sends on the next call."""


class HumanInputSimulator:
    """Generate human-input responses for pending agent requests."""

    def __init__(
        self,
        llm: BaseLLM | None = None,
        *,
        adapter: ElicitationAdapter | None = None,
    ) -> None:
        self.llm = llm
        self.adapter = adapter

    async def respond(
        self,
        pending: PendingHumanInput | dict,
        golden: ConversationGolden,
        history: list[Message],
    ) -> dict:
        """Return a human-input response for a pending agent request."""
        if isinstance(pending, dict):
            pending = PendingHumanInput.from_metadata(pending)

        scripted = self._scripted_response(pending.type, pending.payload, golden)
        if scripted is not None:
            return self._post_process(scripted, pending)

        if self.adapter is not None:
            return await self.adapter.respond(pending, golden, history)

        if self.llm is None:
            raise ValueError(
                f"No deterministic response for human-input type {pending.type!r} and no adapter or llm provided"
            )

        result = await self.llm.generate_json(
            self._fallback_prompt(pending, golden, history),
            _HUMAN_INPUT_SCHEMA,
        )
        response = result.get("response") if isinstance(result.get("response"), dict) else result
        return self._post_process(response, pending)

    def _scripted_response(self, input_type: str, payload: dict, golden: ConversationGolden) -> dict | None:
        for step in (golden.metadata or {}).get("elicitation_script", []):
            if not isinstance(step, dict) or step.get("trigger") != input_type:
                continue
            question = str((payload.get("content") or {}).get("question", ""))
            matcher = step.get("match_question_contains")
            if matcher is not None and not _matches(question, matcher):
                continue
            for key in ("human_input", "system_event", "response"):
                response = step.get(key)
                if isinstance(response, dict):
                    return deepcopy(response)
        return None

    def _post_process(self, response: dict, pending: PendingHumanInput) -> dict:
        if self.adapter is not None and hasattr(self.adapter, "post_process"):
            return self.adapter.post_process(response, pending)  # type: ignore[attr-defined]
        return response

    def _fallback_prompt(
        self,
        pending: PendingHumanInput,
        golden: ConversationGolden,
        history: list[Message],
    ) -> str:
        history_text = "\n".join(f"[{msg.role}]: {msg.content or ''}" for msg in history) or "(empty)"
        return f"""You are simulating a human responding to an agent request for input.

Scenario: {golden.scenario}
Expected outcome: {golden.expected_outcome}
Persona: {golden.user_persona or "(none)"}
Context: {"; ".join(golden.context or [])}
Input type: {pending.type}
Input payload:
{json.dumps(pending.payload, ensure_ascii=False, indent=2)}
Input hints:
{json.dumps(golden.elicitation_hints or {}, ensure_ascii=False, indent=2)}
Conversation so far:
{history_text}

Return only JSON with a single "response" object suitable for continuing the conversation."""


@dataclass
class IntentMatchMiss:
    """Recorded when elicitation intent matching fails or falls back to LLM."""

    reason: str
    elicitation_type: str
    question: str
    intent: str | None = None
    golden_id: str | None = None
    fallback: str | None = None


def record_intent_miss(misses: list[IntentMatchMiss], miss: IntentMatchMiss) -> None:
    """Append an intent miss and emit a WARNING log."""
    misses.append(miss)
    _logger.warning("Elicitation intent miss: %s", compact_json(asdict(miss)))


def resolve_intent(question: str, golden: ConversationGolden) -> str | None:
    """Map a live question string to an intent key via golden matchers."""
    lowered = question.lower()
    for matcher in (golden.elicitation_hints or {}).get("matchers", []):
        if not isinstance(matcher, dict):
            continue
        intent = matcher.get("intent")
        contains = matcher.get("question_contains") or []
        if intent and _matches(lowered, contains):
            return str(intent)
    return None


def intents(golden: ConversationGolden) -> dict[str, str]:
    hints = golden.elicitation_hints or {}
    raw = hints.get("intents") or {}
    return {str(k): str(v) for k, v in raw.items()}


def _matches(question: str, matcher: Any) -> bool:
    if isinstance(matcher, str):
        return matcher.lower() in question.lower()
    if isinstance(matcher, list):
        return any(str(item).lower() in question.lower() for item in matcher)
    return False
