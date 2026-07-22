"""ConversationalStreamingHttpTarget — drive a streaming HTTP agent across turns."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from harness_evals.conversation.context import get_conversation_key
from harness_evals.core.golden import Golden
from harness_evals.core.types import Message
from harness_evals.errors import TargetInvocationError
from harness_evals.logging_config import compact_json
from harness_evals.plugins import register_target
from harness_evals.targets.base import ConversationTarget
from harness_evals.targets.streaming_http import StreamingHttpTarget, _decode, _parse_sse
from harness_evals.targets.templating import render_headers

logger = logging.getLogger(__name__)

_DEFAULT_SESSION_KEY = "__default__"


@dataclass
class _ConversationSession:
    conversation_id: str | None = None
    session_id: str | None = None
    context: dict[str, str] = field(default_factory=dict)


@register_target("conversational_streaming_http")
@dataclass
class ConversationalStreamingHttpTarget(StreamingHttpTarget, ConversationTarget):
    """POST conversation turns to a streaming endpoint and return assistant messages.

    This target is for ``ConversationGolden`` evaluation. It keeps session IDs
    across calls within one conversation and supports configurable human-input
    continuation requests. Session state is keyed by the active conversation
    context so batch evaluation can run goldens concurrently on one target.
    """

    user_body_template: dict | None = None
    continue_body_template: dict | None = None
    human_input_body_template: dict | None = None
    system_event_body_template: dict | None = None

    human_input_events: list[str] | None = None
    completion_events: list[str] | None = None
    session_metadata_event: str | None = None
    session_fields: dict[str, str] | None = None
    correlation_id_field: str | None = None

    _sessions: dict[str, _ConversationSession] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.human_input_body_template is None and self.system_event_body_template is not None:
            self.human_input_body_template = self.system_event_body_template

    @property
    def conversation_id(self) -> str | None:
        return self._active_session().conversation_id

    @conversation_id.setter
    def conversation_id(self, value: str | None) -> None:
        self._active_session().conversation_id = value

    @property
    def session_id(self) -> str | None:
        return self._active_session().session_id

    @session_id.setter
    def session_id(self, value: str | None) -> None:
        self._active_session().session_id = value

    @property
    def _session_context(self) -> dict[str, str]:
        return self._active_session().context

    def _active_session(self) -> _ConversationSession:
        key = get_conversation_key() or _DEFAULT_SESSION_KEY
        session = self._sessions.get(key)
        if session is None:
            session = _ConversationSession()
            self._sessions[key] = session
        return session

    async def agenerate(
        self,
        messages: list[Message],
        human_input: dict | None = None,
        *,
        system_event: dict | None = None,
    ) -> Message:
        session = self._active_session()
        continuation = human_input if human_input is not None else system_event
        body = self._build_conversation_body(messages, human_input=continuation, session=session)
        header_golden = Golden(
            input=self._last_user_content(messages),
            metadata=self._template_context(continuation, session=session),
        )
        headers = render_headers(self.headers, header_golden)

        if self._async_client is not None:
            raw_body, content_type, latency_ms, error = await self._execute_async(body, headers)
        else:
            raw_body, content_type, latency_ms, error = await asyncio.to_thread(
                self._execute_with_retries, body, headers
            )

        if error is not None:
            raise TargetInvocationError(
                f"ConversationalStreamingHttpTarget invocation failed: {error}",
                latency_ms=latency_ms,
            )

        output, _kwargs, metadata_extra, _extract_source = self._process_response(
            raw_body,
            content_type,
            self._last_user_content(messages),
        )
        decoded = self._decode_stream(raw_body, content_type)
        self._update_session(decoded, session=session)
        pending = self._pending_human_input(decoded)
        interaction_id = self._last_metadata_value(decoded, "interaction_id")
        event_names = [name for name, _payload in decoded]
        request_kind = "human_input" if continuation is not None else "user"
        entity_mutations = [payload for name, payload in decoded if name == "entity_mutation"]
        logger.debug(
            "ConversationalStreamingHttpTarget %s: conversation_id=%s session_id=%s events=%s pending=%s",
            request_kind,
            session.conversation_id,
            session.session_id,
            event_names,
            (pending or {}).get("type"),
        )
        if entity_mutations:
            logger.debug(
                "ConversationalStreamingHttpTarget %s entity_mutation payload=%s",
                request_kind,
                compact_json(entity_mutations),
            )

        metadata = {
            **(metadata_extra or {}),
            **session.context,
            "conversation_id": session.conversation_id,
            "session_id": session.session_id,
            "interaction_id": interaction_id,
            "latency_ms": latency_ms,
            "pending_human_input": pending,
            "pending_elicitation": pending,
        }

        return Message(
            role="assistant",
            content=self._message_content(output),
            metadata=metadata,
        )

    def _build_conversation_body(
        self,
        messages: list[Message],
        *,
        human_input: dict | None,
        session: _ConversationSession,
    ) -> bytes:
        if human_input is not None:
            template = self.human_input_body_template or {"human_input": "{{human_input}}", "stream": True}
        elif session.conversation_id is None:
            template = (
                self.user_body_template or self.body_template or {"prompt": "{{last_user_content}}", "stream": True}
            )
        else:
            template = self.continue_body_template or {
                "prompt": "{{last_user_content}}",
                "conversation_id": "{{conversation_id}}",
                "session_id": "{{session_id}}",
                "stream": True,
            }

        payload = _render_template(
            template,
            self._template_context(human_input, messages=messages, session=session),
        )
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def _template_context(
        self,
        human_input: dict | None = None,
        *,
        messages: list[Message] | None = None,
        session: _ConversationSession | None = None,
    ) -> dict:
        active = session or self._active_session()
        last_user_content = self._last_user_content(messages or [])
        context = {
            "last_user_content": last_user_content,
            "conversation_id": active.conversation_id,
            "session_id": active.session_id,
            "human_input": human_input,
            "system_event": human_input,
            "elicitation_response": human_input,
            **active.context,
        }
        return context

    @staticmethod
    def _last_user_content(messages: list[Message]) -> str:
        for msg in reversed(messages):
            if msg.role == "user":
                return msg.content or ""
        return ""

    def _decode_stream(self, raw_body: object, content_type: str) -> list[tuple[str, object]]:
        if raw_body is None or "text/event-stream" not in content_type:
            return []
        return [(name, _decode(data)) for name, data in _parse_sse(str(raw_body))]

    def _update_session(self, decoded: list[tuple[str, object]], *, session: _ConversationSession) -> None:
        if not self.session_metadata_event or not self.session_fields:
            return
        latest: dict | None = None
        for name, payload in decoded:
            if name == self.session_metadata_event and isinstance(payload, dict):
                latest = payload
        if latest is None:
            return
        for context_key, payload_key in self.session_fields.items():
            value = latest.get(payload_key)
            if value is None:
                continue
            self._set_session_value(context_key, str(value), session=session)

    def _set_session_value(self, context_key: str, value: str, *, session: _ConversationSession) -> None:
        session.context[context_key] = value
        if context_key == "conversation_id":
            session.conversation_id = value
        if context_key == "session_id":
            session.session_id = value

    def _pending_human_input(self, decoded: list[tuple[str, object]]) -> dict | None:
        completion_events = self.completion_events
        if completion_events is None and self.output_event:
            completion_events = [self.output_event]
        completion_events = completion_events or []

        if any(name in completion_events for name, _ in decoded):
            return None

        if not self.human_input_events:
            return None

        for name, payload in reversed(decoded):
            if name in self.human_input_events and isinstance(payload, dict):
                correlation_id = None
                if self.correlation_id_field:
                    raw = payload.get(self.correlation_id_field)
                    correlation_id = str(raw) if raw is not None else None
                return {
                    "type": name,
                    "correlation_id": correlation_id,
                    "review_id": correlation_id,
                    "payload": payload,
                }
        return None

    @staticmethod
    def _last_metadata_value(decoded: list[tuple[str, object]], key: str) -> str | None:
        for name, payload in reversed(decoded):
            if name == "stream_metadata" and isinstance(payload, dict) and payload.get(key):
                return str(payload[key])
        return None

    @staticmethod
    def _message_content(output: object) -> str:
        if isinstance(output, str):
            return output
        return json.dumps(output, ensure_ascii=False)


def _render_template(node: Any, context: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        return {key: _render_template(value, context) for key, value in node.items()}
    if isinstance(node, list):
        return [_render_template(item, context) for item in node]
    if isinstance(node, str):
        return _render_string(node, context)
    return node


def _render_string(value: str, context: dict[str, Any]) -> Any:
    if value.startswith("{{") and value.endswith("}}") and value.count("{{") == 1 and value.count("}}") == 1:
        key = value[2:-2].strip()
        return context.get(key)
    rendered = value
    for key, replacement in context.items():
        rendered = rendered.replace("{{" + key + "}}", "" if replacement is None else str(replacement))
    return rendered
