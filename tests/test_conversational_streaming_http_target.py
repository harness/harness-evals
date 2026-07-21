"""Tests for ConversationalStreamingHttpTarget."""

import json

import pytest

from harness_evals.core.types import Message
from harness_evals.targets.conversational_streaming_http import ConversationalStreamingHttpTarget


def _sse(events: list[tuple[str, dict]]) -> str:
    return "\n\n".join(f"event: {name}\ndata: {json.dumps(payload)}" for name, payload in events) + "\n\n"


def _harness_target(**overrides) -> ConversationalStreamingHttpTarget:
    defaults = {
        "url": "http://example.test/stream",
        "output_event": "assistant_message",
        "output_path": "$.v",
        "capture_events": ["elicitation_form", "assistant_tool_request"],
        "human_input_events": ["elicitation_form"],
        "session_metadata_event": "stream_metadata",
        "session_fields": {"conversation_id": "conversation_id", "session_id": "session_id"},
        "correlation_id_field": "review_id",
        "human_input_body_template": {
            "system_event": "{{human_input}}",
            "conversation_id": "{{conversation_id}}",
            "session_id": "{{session_id}}",
            "stream": True,
        },
    }
    defaults.update(overrides)
    return ConversationalStreamingHttpTarget(**defaults)


@pytest.mark.unit
async def test_conversational_streaming_target_detects_pending_elicitation(monkeypatch):
    target = _harness_target()
    captured: dict[str, dict] = {}

    def fake_execute(body: bytes, headers: dict[str, str]):
        captured["body"] = json.loads(body.decode("utf-8"))
        return (
            _sse(
                [
                    (
                        "stream_metadata",
                        {
                            "conversation_id": "conv-1",
                            "session_id": "sess-1",
                            "interaction_id": "int-1",
                        },
                    ),
                    (
                        "elicitation_form",
                        {
                            "review_id": "ask-1",
                            "content": {
                                "fields": [
                                    {
                                        "label": "Auth method",
                                        "options": [{"label": "Inherit", "value": "Inherit"}],
                                    }
                                ]
                            },
                        },
                    ),
                ]
            ),
            "text/event-stream",
            12.0,
            None,
        )

    monkeypatch.setattr(target, "_execute_with_retries", fake_execute)

    msg = await target.agenerate([Message(role="user", content="Create a k8s connector")])

    assert captured["body"] == {"prompt": "Create a k8s connector", "stream": True}
    assert target.conversation_id == "conv-1"
    assert target.session_id == "sess-1"
    assert msg.content == ""
    assert msg.metadata["interaction_id"] == "int-1"
    assert msg.metadata["pending_human_input"]["type"] == "elicitation_form"
    assert msg.metadata["pending_human_input"]["correlation_id"] == "ask-1"
    assert "elicitation_form" in msg.metadata["sse_events"]


@pytest.mark.unit
async def test_conversational_streaming_target_uses_latest_stream_metadata(monkeypatch):
    target = _harness_target()

    def fake_execute(body: bytes, headers: dict[str, str]):
        return (
            _sse(
                [
                    (
                        "stream_metadata",
                        {
                            "conversation_id": "conv-1",
                            "session_id": "sess-initial",
                            "interaction_id": "int-1",
                        },
                    ),
                    (
                        "stream_metadata",
                        {
                            "conversation_id": "conv-1",
                            "session_id": "conv-1",
                            "interaction_id": "int-1",
                        },
                    ),
                    ("elicitation_form", {"review_id": "ask-1", "content": {"fields": []}}),
                ]
            ),
            "text/event-stream",
            5.0,
            None,
        )

    monkeypatch.setattr(target, "_execute_with_retries", fake_execute)

    await target.agenerate([Message(role="user", content="hello")])

    assert target.conversation_id == "conv-1"
    assert target.session_id == "conv-1"


@pytest.mark.unit
async def test_conversational_streaming_target_sends_human_input_and_completes(monkeypatch):
    target = _harness_target(capture_events=None)
    target.conversation_id = "conv-1"
    target.session_id = "sess-1"
    captured: dict[str, dict] = {}

    def fake_execute(body: bytes, headers: dict[str, str]):
        captured["body"] = json.loads(body.decode("utf-8"))
        return (
            _sse(
                [
                    (
                        "stream_metadata",
                        {
                            "conversation_id": "conv-1",
                            "session_id": "conv-1",
                            "interaction_id": "int-2",
                        },
                    ),
                    (
                        "entity_mutation",
                        {"action": "create", "resource_type": "connector", "identifier": "testconnector"},
                    ),
                    ("assistant_message", {"v": "Connector created."}),
                ]
            ),
            "text/event-stream",
            18.0,
            None,
        )

    monkeypatch.setattr(target, "_execute_with_retries", fake_execute)
    human_input = {
        "event_type": "action_completed",
        "capability_id": "ask-1",
        "result": {"success": True, "action_id": "respond", "free_text": "testconnector"},
    }

    msg = await target.agenerate([Message(role="user", content="Create a k8s connector")], human_input=human_input)

    assert captured["body"] == {
        "system_event": human_input,
        "conversation_id": "conv-1",
        "session_id": "sess-1",
        "stream": True,
    }
    assert target.session_id == "conv-1"
    assert msg.content == "Connector created."
    assert msg.metadata["pending_human_input"] is None
    assert msg.metadata["sse_events"]["entity_mutation"][0]["identifier"] == "testconnector"


@pytest.mark.unit
async def test_conversational_streaming_target_supports_custom_human_input_events(monkeypatch):
    target = ConversationalStreamingHttpTarget(
        url="http://example.test/stream",
        output_event="done",
        output_path="$.text",
        human_input_events=["needs_input"],
        correlation_id_field="request_id",
    )
    captured: dict[str, dict] = {}

    def fake_execute(body: bytes, headers: dict[str, str]):
        captured["body"] = json.loads(body.decode("utf-8"))
        return (
            _sse([("needs_input", {"request_id": "req-9", "question": "Continue?"})]),
            "text/event-stream",
            5.0,
            None,
        )

    monkeypatch.setattr(target, "_execute_with_retries", fake_execute)

    msg = await target.agenerate([Message(role="user", content="hello")])

    assert captured["body"] == {"prompt": "hello", "stream": True}
    assert msg.metadata["pending_human_input"] == {
        "type": "needs_input",
        "correlation_id": "req-9",
        "review_id": "req-9",
        "payload": {"request_id": "req-9", "question": "Continue?"},
    }
