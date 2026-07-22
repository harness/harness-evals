"""Tests that ConversationalStreamingHttpTarget isolates session state per golden."""

from __future__ import annotations

import json

import pytest

from harness_evals.conversation import ConversationGolden, ConversationMode, ConversationSimulator
from harness_evals.core.types import Message
from harness_evals.targets.conversational_streaming_http import ConversationalStreamingHttpTarget


def _sse(events: list[tuple[str, dict]]) -> str:
    return "\n\n".join(f"event: {name}\ndata: {json.dumps(payload)}" for name, payload in events) + "\n\n"


def _harness_target(**overrides) -> ConversationalStreamingHttpTarget:
    defaults = {
        "url": "http://example.test/stream",
        "output_event": "assistant_message",
        "output_path": "$.v",
        "session_metadata_event": "stream_metadata",
        "session_fields": {"conversation_id": "conversation_id", "session_id": "session_id"},
    }
    defaults.update(overrides)
    return ConversationalStreamingHttpTarget(**defaults)


def _scripted_golden(*, golden_id: str, prompt: str) -> ConversationGolden:
    return ConversationGolden(
        id=golden_id,
        scenario=f"Scenario for {golden_id}",
        expected_outcome="done",
        mode=ConversationMode.SCRIPTED,
        turns=[Message(role="user", content=prompt)],
        max_turns=1,
    )


@pytest.mark.unit
async def test_simulate_batch_isolates_first_turn_requests_across_goldens(monkeypatch):
    """Golden B's first request must not reuse golden A's conversation/session IDs."""
    target = _harness_target()
    requests: list[dict] = []

    def fake_execute(body: bytes, headers: dict[str, str]):
        payload = json.loads(body.decode("utf-8"))
        requests.append(payload)
        prompt = payload.get("prompt", "")
        if "connector" in prompt:
            conv_id, sess_id = "conv-a", "sess-a"
        else:
            conv_id, sess_id = "conv-b", "sess-b"
        return (
            _sse(
                [
                    ("stream_metadata", {"conversation_id": conv_id, "session_id": sess_id}),
                    ("assistant_message", {"v": f"done:{prompt}"}),
                ]
            ),
            "text/event-stream",
            1.0,
            None,
        )

    monkeypatch.setattr(target, "_execute_with_retries", fake_execute)

    goldens = [
        _scripted_golden(golden_id="golden-a", prompt="Create connector A"),
        _scripted_golden(golden_id="golden-b", prompt="Create pipeline B"),
    ]
    simulator = ConversationSimulator(simulator_llm=None, max_concurrent=1)
    await simulator.simulate_batch(goldens, target.agenerate)

    assert len(requests) == 2
    assert requests[0] == {"prompt": "Create connector A", "stream": True}
    assert requests[1] == {"prompt": "Create pipeline B", "stream": True}
    assert "conversation_id" not in requests[1]
    assert "session_id" not in requests[1]

    session_a = target._sessions["golden-a"]
    session_b = target._sessions["golden-b"]
    assert session_a.conversation_id == "conv-a"
    assert session_a.session_id == "sess-a"
    assert session_b.conversation_id == "conv-b"
    assert session_b.session_id == "sess-b"


@pytest.mark.unit
async def test_simulate_batch_concurrent_requests_keep_per_golden_session_ids(monkeypatch):
    """Concurrent simulations must not cross-contaminate session IDs."""
    target = _harness_target()
    requests: list[dict] = []

    def fake_execute(body: bytes, headers: dict[str, str]):
        payload = json.loads(body.decode("utf-8"))
        requests.append(payload)
        prompt = payload.get("prompt", "")
        if "connector" in prompt:
            conv_id, sess_id = "conv-a", "sess-a"
        elif "pipeline" in prompt:
            conv_id, sess_id = "conv-b", "sess-b"
        else:
            conv_id = payload.get("conversation_id", "unknown")
            sess_id = payload.get("session_id", "unknown")
        return (
            _sse(
                [
                    ("stream_metadata", {"conversation_id": conv_id, "session_id": sess_id}),
                    ("assistant_message", {"v": f"done:{prompt or conv_id}"}),
                ]
            ),
            "text/event-stream",
            1.0,
            None,
        )

    monkeypatch.setattr(target, "_execute_with_retries", fake_execute)

    goldens = [
        _scripted_golden(golden_id="golden-a", prompt="Create connector A"),
        _scripted_golden(golden_id="golden-b", prompt="Create pipeline B"),
    ]
    simulator = ConversationSimulator(simulator_llm=None, max_concurrent=2)
    await simulator.simulate_batch(goldens, target.agenerate)

    first_turn_bodies = [body for body in requests if "prompt" in body]
    assert len(first_turn_bodies) == 2
    for body in first_turn_bodies:
        assert "conversation_id" not in body
        assert "session_id" not in body

    assert target._sessions["golden-a"].conversation_id == "conv-a"
    assert target._sessions["golden-a"].session_id == "sess-a"
    assert target._sessions["golden-b"].conversation_id == "conv-b"
    assert target._sessions["golden-b"].session_id == "sess-b"
