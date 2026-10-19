"""Example elicitation adapter for Harness SSE agents.

Loads when listed under ``plugins`` in an eval YAML. Registers
``HarnessSseElicitationAdapter`` for ``conversation.elicitation_adapter:
harness_sse``.

When ``elicitation_hints`` are present on the golden, responses are resolved
deterministically via intent matchers. Set ``elicitation_hints.llm_on_miss: true``
to fall back to ``conversation.simulator_llm`` when no matcher resolves (QA wizards
often ask different questions than prod skill flows). When hints are omitted entirely,
the adapter always uses the simulator LLM.

For a choice whose question resolves to no usable intent, the adapter picks the
single option exactly equal to any acceptable intent answer. With
``elicitation_hints.on_unmatched: accept`` it then answers like a satisfied user:
the single recommended option, else the single proceed/accept option.

Config::

    plugins:
      - examples.harness_sse_elicitation_adapter

    conversation:
      elicitation_adapter: harness_sse
      simulator_llm: {provider: openai, name: gpt-4o-mini}
"""

from __future__ import annotations

import json
import re
from copy import deepcopy

from harness_evals.conversation.golden import ConversationGolden
from harness_evals.conversation.human_input import (
    ElicitationAdapter,
    HumanInputSimulator,
    IntentMatchMiss,
    PendingHumanInput,
    UnresolvedHumanInputError,
    intent_choices,
    intents,
    is_specific,
    record_intent_miss,
    resolve_intent,
)
from harness_evals.core.types import Message
from harness_evals.llm.base import BaseLLM
from harness_evals.plugins import register_elicitation_adapter

_ANSWER_ONLY_ASKED = (
    "\nAnswer only the fields and question in this payload. Do not volunteer details from the\n"
    "context that the payload does not request, and leave unrequested fields out."
)

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
            "required": ["success", "action_id", "selection"],
            "properties": {
                **_LLM_RESULT_BASE,
                # OpenAI structured outputs require every properties key to be
                # listed in required — keep only the canonical field here.
                # selected_value is added post-normalize as a compatibility alias.
                "selection": {"type": "string"},
            },
        }
    },
}

_LLM_MULTI_SELECT_SCHEMA = {
    "type": "object",
    "required": ["result"],
    "properties": {
        "result": {
            "type": "object",
            "required": ["success", "action_id", "selections"],
            "properties": {
                **_LLM_RESULT_BASE,
                "selections": {"type": "array", "items": {"type": "string"}},
            },
        }
    },
}


def _llm_schema_for_pending(pending: PendingHumanInput) -> dict:
    if pending.type == "elicitation_form":
        return _LLM_FORM_SCHEMA
    if pending.type == "elicitation_free_text":
        return _LLM_FREE_TEXT_SCHEMA
    if pending.type == "elicitation_multi_select":
        return _LLM_MULTI_SELECT_SCHEMA
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
        if pending.type == "elicitation_yaml":
            return self._with_capability_id(self._yaml_response(pending.payload, golden), pending)

        if pending.type == "elicitation_confirm":
            return self._with_capability_id(self._confirm_response(pending.payload, golden), pending)

        if not golden.elicitation_hints:
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

        if _llm_on_miss(golden) and self.llm is not None and self._deterministic_would_miss(pending, golden):
            record_intent_miss(
                self.intent_misses,
                IntentMatchMiss(
                    reason="hint_miss_llm_fallback",
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
        if pending.type == "elicitation_multi_select":
            return self._with_capability_id(self._multi_select_response(payload, golden), pending)
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
        hints_block = _hints_guidance_block(golden)
        prompt = f"""You are simulating a human answering a Harness agent elicitation during an eval run.

Scenario: {golden.scenario}
Expected outcome: {golden.expected_outcome}
Persona: {golden.user_persona or "(none)"}
Context: {"; ".join(golden.context or [])}
{hints_block}
Elicitation type: {pending.type}
Elicitation payload:
{json.dumps(pending.payload, ensure_ascii=False, indent=2)}

Conversation so far:
{history_text}

Return JSON with a single "result" object for a Harness action_completed system_event.
Use action_id "respond" for form/free_text/select/multi_select and "accept" for yaml
review unless the payload actions suggest otherwise.

Result field guide by type:
- elicitation_form: include form_values as an array of {{label, value}} objects for each field
- elicitation_free_text: include free_text with a concise answer
- elicitation_select: include selection set to an offered option LABEL (never the numeric id)
- elicitation_multi_select: include selections as an array of offered option LABELS
{_ANSWER_ONLY_ASKED if is_specific(golden) else ""}
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
            intent = _field_intent(field, golden, label)
            desired = intent_values.get(intent or "", "")
            field_type = _field_type(field)
            options = field.get("options") or []

            if desired:
                if field_type == "text" or not options:
                    # Honor emitted text fields: pass curated value through as-is.
                    form_values[label] = desired
                elif field_type == "multi_select":
                    selected = _select_choice_multi(options, golden, intent, desired)
                    if not selected:
                        self._raise_unresolved(
                            golden=golden,
                            pending_type="elicitation_form",
                            question=label,
                            intent=intent,
                            desired=desired,
                            options=options,
                        )
                    form_values[label] = ", ".join(selected)
                else:
                    selected = _select_choice(options, golden, intent, desired) or self._fallback_option(
                        options, golden, pending_type="elicitation_form", question=label, intent=intent
                    )
                    if selected is None:
                        self._raise_unresolved(
                            golden=golden,
                            pending_type="elicitation_form",
                            question=label,
                            intent=intent,
                            desired=desired,
                            options=options,
                        )
                    form_values[label] = selected
            elif (
                options
                and field_type == "select"
                and (
                    selected := self._fallback_option(
                        options, golden, pending_type="elicitation_form", question=label, intent=intent
                    )
                )
            ):
                form_values[label] = selected
            elif options and field_type in {"select", "multi_select"}:
                self._raise_unresolved(
                    golden=golden,
                    pending_type="elicitation_form",
                    question=label,
                    intent=intent,
                    desired=desired,
                    options=options,
                )
            else:
                self._record_field_miss(
                    golden=golden,
                    pending_type="elicitation_form",
                    question=label,
                    intent=intent,
                    desired=desired,
                    fallback="empty",
                )
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
        options = _select_options(content)
        selected = _select_choice(options, golden, intent, desired) if desired else None
        if not selected:
            selected = self._fallback_option(
                options, golden, pending_type="elicitation_select", question=question, intent=intent
            )
        if not selected:
            self._raise_unresolved(
                golden=golden,
                pending_type="elicitation_select",
                question=question,
                intent=intent,
                desired=desired,
                options=options,
            )

        result: dict[str, object] = {
            "success": True,
            "action_id": "respond",
            "selection": selected,
            # Backward-compatible alias used by older previews/tests.
            "selected_value": selected,
        }
        if question:
            result["form_values"] = {question: selected}
        return {
            "event_type": "action_completed",
            "result": result,
        }

    def _multi_select_response(self, payload: dict, golden: ConversationGolden) -> dict:
        content = payload.get("content") or {}
        question = str(
            content.get("question") or content.get("label") or payload.get("title") or payload.get("subtitle") or ""
        )
        intent = resolve_intent(question, golden) or resolve_intent(str(payload.get("title") or ""), golden)
        desired = intents(golden).get(intent or "", "")
        options = _select_options(content)
        selected = _select_choice_multi(options, golden, intent, desired) if desired else []
        if not selected:
            self._raise_unresolved(
                golden=golden,
                pending_type="elicitation_multi_select",
                question=question,
                intent=intent,
                desired=desired,
                options=options,
            )

        result: dict[str, object] = {
            "success": True,
            "action_id": "respond",
            "selections": selected,
            # HITL resume uses selection when len(selections) == 1.
            "selection": selected[0],
        }
        return {
            "event_type": "action_completed",
            "result": result,
        }

    def _yaml_response(self, payload: dict, golden: ConversationGolden) -> dict:
        hints = golden.elicitation_hints or {}
        action_id = str((hints.get("yaml") or {}).get("default_action") or "accept")
        event_type = _event_type_for_action(payload, action_id)
        if event_type == "action_cancelled":
            return {
                "event_type": event_type,
                "result": {},
            }
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
            "event_type": event_type,
            "result": {k: v for k, v in result.items() if v is not None},
        }

    def _confirm_response(self, payload: dict, golden: ConversationGolden) -> dict:
        """Approve non-YAML entity mutations (e.g. cost_category) via confirm card."""
        hints = golden.elicitation_hints or {}
        confirm_hints = hints.get("confirm") or hints.get("yaml") or {}
        action_id = str(confirm_hints.get("default_action") or "approve")
        event_type = _event_type_for_action(payload, action_id)
        if event_type == "action_cancelled":
            return {
                "event_type": event_type,
                "result": {},
            }
        entity_info = payload.get("entity_info") or {}
        result = {
            "success": True,
            "action_id": action_id,
            "entity_info": entity_info,
            "tool_input": payload.get("tool_input"),
        }
        return {
            "event_type": event_type,
            "result": {k: v for k, v in result.items() if v is not None},
        }

    def _deterministic_would_miss(self, pending: PendingHumanInput, golden: ConversationGolden) -> bool:
        payload = pending.payload
        if pending.type == "elicitation_form":
            return self._form_would_miss(payload, golden)
        if pending.type == "elicitation_free_text":
            return self._free_text_would_miss(payload, golden)
        if pending.type == "elicitation_select":
            return self._select_would_miss(payload, golden)
        if pending.type == "elicitation_multi_select":
            return self._multi_select_would_miss(payload, golden)
        return True

    def _form_would_miss(self, payload: dict, golden: ConversationGolden) -> bool:
        intent_values = intents(golden)
        for field in (payload.get("content") or {}).get("fields") or []:
            if not isinstance(field, dict):
                continue
            label = str(field.get("label") or field.get("key") or "")
            intent = _field_intent(field, golden, label)
            desired = intent_values.get(intent or "", "")
            if not desired:
                return True
            field_type = _field_type(field)
            options = field.get("options") or []
            if field_type in {"select", "multi_select"} and options:
                if field_type == "multi_select":
                    if not _select_choice_multi(options, golden, intent, desired):
                        return True
                elif _select_choice(options, golden, intent, desired) is None:
                    return True
        return False

    def _free_text_would_miss(self, payload: dict, golden: ConversationGolden) -> bool:
        question = str((payload.get("content") or {}).get("question") or payload.get("title") or "")
        intent = resolve_intent(question, golden)
        return not intents(golden).get(intent or "", "")

    def _select_would_miss(self, payload: dict, golden: ConversationGolden) -> bool:
        content = payload.get("content") or {}
        if content.get("fields"):
            return self._form_would_miss(payload, golden)

        question = str(
            content.get("question") or content.get("label") or payload.get("title") or payload.get("subtitle") or ""
        )
        intent = resolve_intent(question, golden) or resolve_intent(str(payload.get("title") or ""), golden)
        desired = intents(golden).get(intent or "", "")
        options = _select_options(content)
        # Empty options cannot be matched deterministically — treat as miss so
        # llm_on_miss can fall back instead of raising ValueError downstream.
        if not options:
            return True
        if desired and _select_choice(options, golden, intent, desired) is not None:
            return False
        if not is_specific(golden):
            return True
        return (
            _unique_intent_option(options, golden) is None and _keyword_intent_option(options, golden, intent) is None
        )

    def _multi_select_would_miss(self, payload: dict, golden: ConversationGolden) -> bool:
        content = payload.get("content") or {}
        question = str(
            content.get("question") or content.get("label") or payload.get("title") or payload.get("subtitle") or ""
        )
        intent = resolve_intent(question, golden) or resolve_intent(str(payload.get("title") or ""), golden)
        desired = intents(golden).get(intent or "", "")
        if not desired:
            return True
        options = _select_options(content)
        # Empty options cannot be matched deterministically — treat as miss so
        # llm_on_miss can fall back instead of raising ValueError downstream.
        if not options:
            return True
        return not _select_choice_multi(options, golden, intent, desired)

    def _raise_unresolved(
        self,
        *,
        golden: ConversationGolden,
        pending_type: str,
        question: str,
        intent: str | None,
        desired: str,
        options: list | None = None,
    ) -> None:
        self._record_field_miss(
            golden=golden,
            pending_type=pending_type,
            question=question,
            intent=intent,
            desired=desired,
            fallback="unresolved",
        )
        raise UnresolvedHumanInputError(
            f"Unresolved {pending_type} answer for question={question!r} "
            f"intent={intent!r} desired={desired!r} options={options or []!r}. "
            "Add a matcher/intent or enable elicitation_hints.llm_on_miss with simulator_llm."
        )

    def _fallback_option(
        self,
        options: list,
        golden: ConversationGolden,
        *,
        pending_type: str,
        question: str,
        intent: str | None,
    ) -> str | None:
        """Answer an unmatched choice by exact intent value, then by keyword overlap, else by the accept default."""
        specific = is_specific(golden)
        if specific and (selected := _unique_intent_option(options, golden)) is not None:
            return selected
        if specific and (selected := _keyword_intent_option(options, golden, intent)) is not None:
            self._record_field_miss(
                golden=golden,
                pending_type=pending_type,
                question=question,
                intent=intent,
                desired=selected,
                fallback="keyword_overlap",
            )
            return selected
        selected = _accept_default_option(options, golden)
        if selected is not None:
            self._record_field_miss(
                golden=golden,
                pending_type=pending_type,
                question=question,
                intent=intent,
                desired=selected,
                fallback="accept_default",
            )
        return selected

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
        if desired and fallback == "unresolved":
            reason = "option_match_miss"
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


def _event_type_for_action(payload: dict, action_id: str) -> str:
    for action in payload.get("actions") or []:
        if isinstance(action, dict) and str(action.get("id")) == action_id:
            sends = action.get("sends")
            if sends in {"action_completed", "action_cancelled"}:
                return str(sends)
    if action_id.lower() in {"cancel", "deny", "reject"}:
        return "action_cancelled"
    return "action_completed"


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

    # Prefer canonical selection; keep selected_value as alias.
    if pending.type == "elicitation_select":
        selected = normalized.get("selection") or normalized.get("selected_value")
        if selected is not None:
            selected_str = str(selected)
            normalized["selection"] = selected_str
            normalized["selected_value"] = selected_str
            content = pending.payload.get("content") or {}
            question = str(
                content.get("question")
                or content.get("label")
                or pending.payload.get("title")
                or pending.payload.get("subtitle")
                or ""
            )
            if question and "form_values" not in normalized:
                normalized["form_values"] = {question: selected_str}

    if pending.type == "elicitation_multi_select":
        selections = normalized.get("selections")
        if isinstance(selections, list):
            cleaned = [str(item) for item in selections if item is not None and str(item)]
            normalized["selections"] = cleaned
            if cleaned and not normalized.get("selection"):
                normalized["selection"] = cleaned[0]
        elif normalized.get("selection"):
            normalized["selections"] = [str(normalized["selection"])]
    return normalized


def _select_options(content: dict) -> list:
    """Normalize Harness select payloads (options, choices, or items)."""
    for key in ("options", "choices", "items"):
        raw = content.get(key)
        if isinstance(raw, list) and raw:
            return raw
    return []


def _intent_for_field_key(
    field: dict, golden: ConversationGolden, *, suffix: bool = False, label: str | None = None
) -> str | None:
    """Resolve a form field by its stable key before falling back to label text.

    A key resolves when it names an intent directly, is mapped via
    ``elicitation_hints.field_keys``, or its last ``_`` segment is exactly one
    alias or intent name (``s3_region`` resolves like ``region``). Suffix
    resolution runs only with ``suffix=True``, after label matching, because a
    descriptive label outranks a key fragment (``docker_repo`` labelled "Docker
    image name" is the image, not the source repository). A suffix is
    ambiguous, and left to label matching, when another intent is a qualified
    form of it (``docker_repository`` with both ``repository`` and
    ``image_repository`` defined).
    """
    key = str(field.get("key") or "").strip()
    if not key:
        return None
    hints = golden.elicitation_hints or {}
    aliases = hints.get("field_keys") or {}
    if not isinstance(aliases, dict):
        aliases = {}
    known_intents = hints.get("intents") or {}
    if not suffix:
        if key in aliases:
            return str(aliases[key])
        return key if key in known_intents else None
    tail = key.rsplit("_", 1)[-1]
    if label is not None and not re.search(rf"\b{re.escape(tail)}\b", label, re.IGNORECASE):
        return None
    qualified = [name for name in known_intents if name != tail and str(name).endswith(f"_{tail}")]
    if tail != key and not qualified:
        if tail in aliases:
            return str(aliases[tail])
        if tail in known_intents:
            return tail
    return None


def _field_intent(field: dict, golden: ConversationGolden, label: str) -> str | None:
    """Resolve a form field to an intent by its visible label first, then its key.

    The label is what a real user reads; keys are model-chosen shorthand
    (``repo`` on an "Image repository" field), so they only break ties or
    fill in when the label matches nothing. Key resolution is specific-mode only.
    """
    if not is_specific(golden):
        return resolve_intent(label, golden)
    return (
        resolve_intent(label, golden)
        or _intent_for_field_key(field, golden)
        or _intent_for_field_key(field, golden, suffix=True, label=label)
        or _intent_for_field_key(field, golden, suffix=True)
    )


def _unique_intent_option(options: list, golden: ConversationGolden) -> str | None:
    """Pick the single offered option whose label or value exactly equals an acceptable intent answer.

    Partial text is not enough here: the question did not resolve to an intent,
    so a loose match (``Pipeline`` inside ``harness_pipelines``) would answer
    with an unrelated fact.
    """
    hints = golden.elicitation_hints or {}
    matches = {
        selected
        for intent in (hints.get("intents") or {})
        for choice in intent_choices(golden, str(intent))
        if (selected := _exact_option(options, choice)) is not None
    }
    return matches.pop() if len(matches) == 1 else None


_KEYWORD_STOPWORDS = frozenset(
    {"a", "an", "the", "to", "of", "for", "in", "on", "at", "by", "with", "and", "or", "is", "are", "be", "will"}
    | {"it", "its", "this", "that", "your", "my", "our", "you", "we", "i", "use", "using", "enter", "type", "set"}
)


_NEGATED_WORD_RE = re.compile(r"\b(?:don't|do not|doesn't|not|never|no|without)\s+[a-z0-9_]+")
_STEM_SUFFIXES = ("ing", "ion", "ed", "es", "e", "s")


def _stem(word: str) -> str:
    """Fold inflections ("create", "creating", "creation") to one stem; identifiers stay atomic."""
    if "_" in word or any(char.isdigit() for char in word):
        return word
    for suffix in _STEM_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _keywords(text: str) -> set[str]:
    """Meaningful word stems in ``text``, skipping stopwords and words a negation rules out ("don't create")."""
    text = _NEGATED_WORD_RE.sub(" ", text.lower())
    return {_stem(word) for word in re.findall(r"[a-z0-9_]+", text) if word not in _KEYWORD_STOPWORDS}


def _keyword_option(options: list, desired_values: list[str]) -> str | None:
    """Pick the option sharing the most meaningful words with any desired value.

    Matches whole words over the option label, value, and description, so a
    reworded option ("Enter a fixed command" for "Enter an exact fixed
    command") still resolves. The winner needs at least two shared words and
    twice the runner-up's score; otherwise the choice is ambiguous.
    """
    wanted = [words for value in desired_values if (words := _keywords(value))]
    if not wanted:
        return None
    scored = []
    for option in options:
        label = _option_label(option) or _option_value(option)
        if not label:
            continue
        description = option.get("description", "") if isinstance(option, dict) else ""
        offered = _keywords(f"{label} {_option_value(option) or ''} {description}")
        scored.append((max(len(words & offered) for words in wanted), label))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    best, label = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0
    return label if best >= 2 and best >= 2 * runner_up else None


def _keyword_intent_option(options: list, golden: ConversationGolden, intent: str | None) -> str | None:
    """Keyword-match the question's own intent first, then every string answer the case defines."""
    if intent and (selected := _keyword_option(options, intent_choices(golden, intent))) is not None:
        return selected
    hints = golden.elicitation_hints or {}
    every = [choice for name in (hints.get("intents") or {}) for choice in intent_choices(golden, str(name))]
    return _keyword_option(options, every)


def _accept_default_option(options: list, golden: ConversationGolden) -> str | None:
    """With ``elicitation_hints.on_unmatched: accept``, answer like a satisfied user.

    Picks the single option marked recommended, else the single option that
    proceeds or accepts as-is. Returns None when neither is unambiguous.
    """
    hints = golden.elicitation_hints or {}
    if hints.get("on_unmatched") != "accept":
        return None
    recommended = [
        label
        for option in options
        if (label := _option_label(option))
        and "recommended" in f"{label} {option.get('description', '') if isinstance(option, dict) else ''}".lower()
    ]
    if len(recommended) == 1:
        return recommended[0]
    accepting = [
        label for option in options if (label := _option_label(option)) and _ACCEPT_OPTION_RE.match(label.strip())
    ]
    return accepting[0] if len(accepting) == 1 else None


_ACCEPT_OPTION_RE = re.compile(
    r"(?i)^(?:looks good|lgtm|proceed|continue|confirm|approve|accept|yes\b|no changes?\b|keep (?:it|as is|as-is)"
    r"|create (?:it|the pipeline|pipeline)\b)"
)


def _exact_option(options: list, desired: str) -> str | None:
    """Return the label of the option whose label or value equals ``desired`` (case-insensitive)."""
    desired_lower = desired.strip().lower()
    if not desired_lower:
        return None
    for option in options:
        label = _option_label(option)
        value = _option_value(option)
        if (label and label.lower() == desired_lower) or (value and value.lower() == desired_lower):
            return label or value
    return None


def _field_type(field: dict) -> str:
    """Return the emitted field type; fall back from options when type is absent."""
    raw = str(field.get("type") or "").strip().lower()
    if raw in {"text", "select", "multi_select", "toggle", "number"}:
        return raw
    if field.get("options"):
        return "select"
    return "text"


def _llm_on_miss(golden: ConversationGolden) -> bool:
    hints = golden.elicitation_hints or {}
    return bool(hints.get("llm_on_miss"))


def _hints_guidance_block(golden: ConversationGolden) -> str:
    hints = golden.elicitation_hints or {}
    intent_values = {k: v for k, v in (hints.get("intents") or {}).items() if v}
    if not intent_values:
        return ""
    return (
        "\nCurated answer intents (prefer these when the question maps to one; "
        "pick the closest offered option LABEL when options are shown — never numeric ids):\n"
        f"{json.dumps(intent_values, ensure_ascii=False, indent=2)}\n"
    )


def _option_label(option: object) -> str:
    if isinstance(option, dict):
        return str(option.get("label") or option.get("value") or option.get("id") or "")
    return str(option)


def _option_value(option: object) -> str:
    if isinstance(option, dict):
        return str(option.get("value") or option.get("id") or "")
    return str(option)


def _option_matches(options: list, selected: str) -> bool:
    selected_lower = selected.lower().strip()
    for option in options:
        label = _option_label(option).lower().strip()
        value = _option_value(option).lower().strip()
        if selected_lower and selected_lower in {label, value}:
            return True
    return False


def _contains_whole(text: str, part: str) -> bool:
    """True when ``part`` occurs in ``text`` not glued to identifier characters.

    ``github_testgo`` is not inside ``github_testgo_account``: picking it would
    answer with a different connector.
    """
    return re.search(rf"(?<![a-z0-9_]){re.escape(part)}(?![a-z0-9_])", text) is not None


def _select_option(options: list, desired: str, allow_custom: bool = True, specific: bool = True) -> str | None:
    """Match desired text to an option and return the option LABEL (never bare id).

    Matching order: exact label → exact value → case-insensitive label contains
    desired → case-insensitive desired contains label. Returns None when no
    confident match exists. Without ``specific`` the first substring match wins
    and no custom entry is typed.
    """
    desired_norm = desired.strip()
    if not desired_norm or not options:
        return None
    desired_lower = desired_norm.lower()

    # 1. Exact label (case-insensitive)
    for option in options:
        label = _option_label(option)
        if label and label.lower() == desired_lower:
            return label

    # 2. Exact value/id (case-insensitive) — still return the label
    for option in options:
        value = _option_value(option)
        label = _option_label(option)
        if value and value.lower() == desired_lower:
            return label or value

    if not specific:
        for option in options:
            label = _option_label(option)
            if label and (desired_lower in label.lower() or label.lower() in desired_lower):
                return label
        return None

    # 3. Desired is a substring of exactly one label (or vice versa). A label that is
    # only a fragment of a longer answer loses to an offered custom entry.
    has_custom = any(_is_custom_option(option) for option in options)
    partial = [
        label
        for option in options
        if (label := _option_label(option))
        and (
            _contains_whole(label.lower(), desired_lower)
            or (not has_custom and _contains_whole(desired_lower, label.lower()))
        )
    ]
    if len(partial) == 1:
        return partial[0]

    # 4. Type the exact answer when the field offers a custom/other entry
    if allow_custom and any(_is_custom_option(option) for option in options):
        return desired_norm

    return None


def _is_custom_option(option: object) -> bool:
    label = (_option_label(option) or "").lower()
    value = (_option_value(option) or "").lower()
    return value in {"__custom__", "custom", "other"} or label.startswith(("custom", "other"))


def _select_choice(options: list, golden: ConversationGolden, intent: str | None, desired: str) -> str | None:
    """Pick the first offered option among an intent's acceptable answers; type the primary as a last resort."""
    if not is_specific(golden):
        return _select_option(options, desired, specific=False)
    choices = intent_choices(golden, intent) or ([desired] if desired else [])
    for choice in choices:
        if (selected := _select_option(options, choice, allow_custom=False)) is not None:
            return selected
    return _select_option(options, choices[0]) if choices else None


def _select_choice_multi(options: list, golden: ConversationGolden, intent: str | None, desired: str) -> list[str]:
    """Multi-select variant of ``_select_choice``."""
    if not is_specific(golden):
        return _select_options_multi(options, desired, specific=False)
    choices = intent_choices(golden, intent) or ([desired] if desired else [])
    for choice in choices:
        if selected := _select_options_multi(options, choice, allow_custom=False):
            return selected
    return _select_options_multi(options, choices[0]) if choices else []


def _select_options_multi(options: list, desired: str, allow_custom: bool = True, specific: bool = True) -> list[str]:
    """Resolve one or more option labels from a comma/semicolon-separated desired string."""
    if not desired.strip():
        return []
    parts = [part.strip() for part in desired.replace(";", ",").split(",") if part.strip()]
    if not parts:
        parts = [desired.strip()]

    selected: list[str] = []
    for part in parts:
        matched = _select_option(options, part, allow_custom, specific)
        if matched and matched not in selected:
            selected.append(matched)

    # If the whole string matches one option (e.g. "All Providers"), prefer that.
    if not selected:
        whole = _select_option(options, desired, allow_custom, specific)
        if whole:
            return [whole]
    return selected
