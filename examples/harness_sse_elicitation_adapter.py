"""Example elicitation adapter for Harness SSE agents.

Loads when listed under ``plugins`` in an eval YAML. Registers
``HarnessSseElicitationAdapter`` for ``conversation.elicitation_adapter:
harness_sse``.

When ``elicitation_hints`` are present on the golden, responses are resolved
deterministically via intent matchers. When hints are omitted, the adapter uses
``conversation.simulator_llm`` to choose answers while still emitting the
Harness ``system_event`` wire format.

Config::

    plugins:
      - examples.harness_sse_elicitation_adapter

    conversation:
      elicitation_adapter: harness_sse
      simulator_llm: {provider: openai, name: gpt-4o-mini}
"""

from __future__ import annotations

import json
from copy import deepcopy

from harness_evals.conversation.golden import ConversationGolden
from harness_evals.conversation.human_input import (
    ElicitationAdapter,
    HumanInputSimulator,
    IntentMatchMiss,
    PendingHumanInput,
    intents,
    record_intent_miss,
    resolve_intent,
)
from harness_evals.core.types import Message
from harness_evals.llm.base import BaseLLM
from harness_evals.plugins import register_elicitation_adapter

_LLM_RESULT_BASE = {
    "success": {"type": "boolean"},
    "action_id": {"type": "string"},
}

_LLM_FORM_VALUE_ITEMS = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "value": {"type": "string"},
    },
    "required": ["label", "value"],
}

_LLM_FORM_SCHEMA = {
    "type": "object",
    "required": ["result"],
    "properties": {
        "result": {
            "type": "object",
            "required": ["success", "action_id", "form_values"],
            "properties": {
                **_LLM_RESULT_BASE,
                "form_values": {
                    "type": "array",
                    "items": _LLM_FORM_VALUE_ITEMS,
                },
            },
        }
    },
}

_LLM_FREE_TEXT_SCHEMA = {
    "type": "object",
    "required": ["result"],
    "properties": {
        "result": {
            "type": "object",
            "required": ["success", "action_id", "free_text"],
            "properties": {
                **_LLM_RESULT_BASE,
                "free_text": {"type": "string"},
            },
        }
    },
}

_LLM_SELECT_SCHEMA = {
    "type": "object",
    "required": ["result"],
    "properties": {
        "result": {
            "type": "object",
            "required": ["success", "action_id", "selected_value"],
            "properties": {
                **_LLM_RESULT_BASE,
                "selected_value": {"type": "string"},
            },
        }
    },
}


def _llm_schema_for_pending(pending: PendingHumanInput) -> dict:
    if pending.type == "elicitation_form":
        return _LLM_FORM_SCHEMA
    if pending.type == "elicitation_free_text":
        return _LLM_FREE_TEXT_SCHEMA
    if pending.type == "elicitation_select":
        content = pending.payload.get("content") or {}
        if content.get("fields"):
            return _LLM_FORM_SCHEMA
        return _LLM_SELECT_SCHEMA
    raise ValueError(f"Unsupported LLM elicitation type {pending.type!r}")


@register_elicitation_adapter("harness_sse")
class HarnessSseElicitationAdapter(ElicitationAdapter):
    """Build Harness ``system_event`` responses for SSE elicitation payloads."""

    def __init__(self) -> None:
        self.llm: BaseLLM | None = None
        self.intent_misses: list[IntentMatchMiss] = []

    def reset_intent_misses(self) -> None:
        self.intent_misses.clear()

    async def respond(
        self,
        pending: PendingHumanInput,
        golden: ConversationGolden,
        history: list[Message],
    ) -> dict:
        if not golden.elicitation_hints:
            if pending.type == "elicitation_yaml":
                return self._with_capability_id(self._yaml_response(pending.payload, golden), pending)
            record_intent_miss(
                self.intent_misses,
                IntentMatchMiss(
                    reason="no_hints_llm_fallback",
                    elicitation_type=pending.type,
                    question=_elicitation_question(pending),
                    golden_id=golden.id,
                    fallback="llm",
                ),
            )
            return self._with_capability_id(await self._llm_system_event(pending, golden, history), pending)

        payload = pending.payload
        if pending.type == "elicitation_form":
            return self._with_capability_id(self._form_response(payload, golden), pending)
        if pending.type == "elicitation_free_text":
            return self._with_capability_id(self._free_text_response(payload, golden), pending)
        if pending.type == "elicitation_select":
            return self._with_capability_id(self._select_response(payload, golden), pending)
        if pending.type == "elicitation_yaml":
            return self._with_capability_id(self._yaml_response(payload, golden), pending)
        raise ValueError(f"Unsupported Harness elicitation type {pending.type!r}")

    def post_process(self, response: dict, pending: PendingHumanInput) -> dict:
        return self._with_capability_id(response, pending)

    async def _llm_system_event(
        self,
        pending: PendingHumanInput,
        golden: ConversationGolden,
        history: list[Message],
    ) -> dict:
        if self.llm is None:
            raise ValueError(
                "Golden has no elicitation_hints; configure conversation.simulator_llm "
                "so the harness_sse adapter can answer elicitations via LLM."
            )

        history_text = "\n".join(f"[{msg.role}]: {msg.content or ''}" for msg in history) or "(empty)"
        prompt = f"""You are simulating a human answering a Harness agent elicitation during an eval run.

Scenario: {golden.scenario}
Expected outcome: {golden.expected_outcome}
Persona: {golden.user_persona or "(none)"}
Context: {"; ".join(golden.context or [])}

Elicitation type: {pending.type}
Elicitation payload:
{json.dumps(pending.payload, ensure_ascii=False, indent=2)}

Conversation so far:
{history_text}

Return JSON with a single "result" object for a Harness action_completed system_event.
Use action_id "respond" for form/free_text/select and "accept" for yaml review unless the
payload actions suggest otherwise.

Result field guide by type:
- elicitation_form: include form_values as an array of {{label, value}} objects for each field
- elicitation_free_text: include free_text with a concise answer
- elicitation_select: include selected_value matching one of the offered options when present

Always set success=true."""

        raw = await self.llm.generate_json(prompt, _llm_schema_for_pending(pending))
        result = raw.get("result") if isinstance(raw.get("result"), dict) else raw
        if not isinstance(result, dict):
            raise ValueError(f"LLM elicitation response must include a result object, got {raw!r}")
        normalized = _normalize_llm_result(result, pending)
        return {
            "event_type": "action_completed",
            "result": {k: v for k, v in normalized.items() if v is not None},
        }

    def _form_response(self, payload: dict, golden: ConversationGolden) -> dict:
        form_values: dict[str, str] = {}
        intent_values = intents(golden)
        for field in (payload.get("content") or {}).get("fields") or []:
            if not isinstance(field, dict):
                continue
            label = str(field.get("label") or field.get("key") or "")
            intent = resolve_intent(label, golden)
            desired = intent_values.get(intent or "", "")
            if desired:
                form_values[label] = _select_option(field.get("options") or [], desired)
            elif field.get("options"):
                self._record_field_miss(
                    golden=golden,
                    pending_type="elicitation_form",
                    question=label,
                    intent=intent,
                    desired=desired,
                    fallback="first_option",
                )
                first = field["options"][0]
                form_values[label] = str(first.get("value") or first.get("label") or "")
        return {
            "event_type": "action_completed",
            "result": {
                "success": True,
                "action_id": "respond",
                "form_values": form_values,
            },
        }

    def _free_text_response(self, payload: dict, golden: ConversationGolden) -> dict:
        question = str((payload.get("content") or {}).get("question") or payload.get("title") or "")
        intent = resolve_intent(question, golden)
        free_text = intents(golden).get(intent or "", "")
        if not free_text:
            self._record_field_miss(
                golden=golden,
                pending_type="elicitation_free_text",
                question=question,
                intent=intent,
                desired=free_text,
                fallback="empty",
            )
        return {
            "event_type": "action_completed",
            "result": {
                "success": True,
                "action_id": "respond",
                "free_text": free_text,
            },
        }

    def _select_response(self, payload: dict, golden: ConversationGolden) -> dict:
        content = payload.get("content") or {}
        if content.get("fields"):
            return self._form_response(payload, golden)

        question = str(
            content.get("question") or content.get("label") or payload.get("title") or payload.get("subtitle") or ""
        )
        intent = resolve_intent(question, golden) or resolve_intent(str(payload.get("title") or ""), golden)
        desired = intents(golden).get(intent or "", "")
        options = content.get("options") or content.get("choices") or []
        selected = _select_option(options, desired) if desired else ""
        if not selected and options:
            self._record_field_miss(
                golden=golden,
                pending_type="elicitation_select",
                question=question,
                intent=intent,
                desired=desired,
                fallback="first_option",
            )
            first = options[0]
            selected = str(first.get("value") or first.get("label") or "") if isinstance(first, dict) else str(first)

        result: dict[str, object] = {
            "success": True,
            "action_id": "respond",
            "selected_value": selected,
        }
        if question:
            result["form_values"] = {question: selected}
        return {
            "event_type": "action_completed",
            "result": result,
        }

    def _yaml_response(self, payload: dict, golden: ConversationGolden) -> dict:
        hints = golden.elicitation_hints or {}
        action_id = str((hints.get("yaml") or {}).get("default_action") or "accept")
        content = payload.get("content") or {}
        entity_info = payload.get("entity_info") or {}
        result = {
            "success": True,
            "action_id": action_id,
            "yaml": content.get("yaml", ""),
            "entity_type": payload.get("entity_type") or entity_info.get("entity_type"),
            "entity_info": entity_info,
            "request_action": payload.get("request_action") or entity_info.get("request_action"),
            "tool_input": payload.get("tool_input"),
        }
        return {
            "event_type": "action_completed",
            "result": {k: v for k, v in result.items() if v is not None},
        }

    def _record_field_miss(
        self,
        *,
        golden: ConversationGolden,
        pending_type: str,
        question: str,
        intent: str | None,
        desired: str,
        fallback: str,
    ) -> None:
        reason = "no_intent_match" if intent is None else "missing_intent_value"
        record_intent_miss(
            self.intent_misses,
            IntentMatchMiss(
                reason=reason,
                elicitation_type=pending_type,
                question=question,
                intent=intent,
                golden_id=golden.id,
                fallback=fallback,
            ),
        )

    @staticmethod
    def _with_capability_id(system_event: dict, pending: PendingHumanInput) -> dict:
        result = deepcopy(system_event)
        result["event_type"] = result.get("event_type", "action_completed")
        correlation_id = pending.correlation_id or pending.payload.get("review_id")
        if correlation_id is not None:
            result["capability_id"] = correlation_id
        return result


class ElicitationSimulator(HumanInputSimulator):
    """Backward-compatible alias for Harness SSE human-input simulation."""

    def __init__(self, llm: BaseLLM | None = None) -> None:
        adapter = HarnessSseElicitationAdapter()
        adapter.llm = llm
        super().__init__(llm, adapter=adapter)

    async def generate_system_event(
        self,
        elicitation: dict,
        golden: ConversationGolden,
        history: list[Message],
    ) -> dict:
        return await self.respond(elicitation, golden, history)


def _elicitation_question(pending: PendingHumanInput) -> str:
    payload = pending.payload
    content = payload.get("content") or {}
    if pending.type == "elicitation_form":
        fields = content.get("fields") or []
        if fields and isinstance(fields[0], dict):
            return str(fields[0].get("label") or fields[0].get("key") or "")
    return str(content.get("question") or content.get("label") or payload.get("title") or payload.get("subtitle") or "")


def _normalize_llm_result(result: dict, pending: PendingHumanInput) -> dict:
    normalized = dict(result)
    form_values = normalized.get("form_values")
    if isinstance(form_values, list):
        normalized["form_values"] = {
            str(item.get("label")): str(item.get("value"))
            for item in form_values
            if isinstance(item, dict) and item.get("label") is not None and item.get("value") is not None
        }
    if pending.type == "elicitation_select" and "form_values" not in normalized:
        content = pending.payload.get("content") or {}
        question = str(
            content.get("question")
            or content.get("label")
            or pending.payload.get("title")
            or pending.payload.get("subtitle")
            or ""
        )
        selected = normalized.get("selected_value")
        if question and selected:
            normalized["form_values"] = {question: str(selected)}
    return normalized


def _select_option(options: list, desired: str) -> str:
    desired_lower = desired.lower()
    for option in options:
        if not isinstance(option, dict):
            continue
        value = str(option.get("value") or "")
        label = str(option.get("label") or "")
        if desired_lower in value.lower() or desired_lower in label.lower():
            return value or label
    return desired
