"""Replay captured Harness SSE fixtures for the k8s connector conversation flow."""

from __future__ import annotations

import json

import pytest

from harness_evals.conversation import ConversationGolden, ConversationSimulator
from harness_evals.core.types import Message
from harness_evals.llm.base import BaseLLM
from harness_evals.targets.conversational_streaming_http import ConversationalStreamingHttpTarget
from tests.conversation.k8s_connector_sse_fixtures import k8s_connector_turn_responses


class StopLLM(BaseLLM):
    async def generate(self, prompt: str, **kwargs) -> str:
        return "Create a k8s connector"

    async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
        return {"achieved": True, "reasoning": "done"}


def _harness_target() -> ConversationalStreamingHttpTarget:
    return ConversationalStreamingHttpTarget(
        url="http://example.test/stream",
        output_event="assistant_message",
        output_path="$.v",
        capture_events=[
            "elicitation_form",
            "elicitation_free_text",
            "elicitation_yaml",
            "entity_mutation",
            "assistant_message",
        ],
        human_input_events=["elicitation_form", "elicitation_free_text", "elicitation_yaml"],
        completion_events=["assistant_message"],
        session_metadata_event="stream_metadata",
        session_fields={"conversation_id": "conversation_id", "session_id": "session_id"},
        correlation_id_field="review_id",
        human_input_body_template={
            "system_event": "{{human_input}}",
            "conversation_id": "{{conversation_id}}",
            "session_id": "{{session_id}}",
            "stream": True,
        },
    )


@pytest.mark.unit
async def test_replay_harness_data_turns_complete_k8s_connector_flow(monkeypatch):
    turn_bodies = k8s_connector_turn_responses()
    responses = iter(turn_bodies)
    requests: list[dict] = []

    target = _harness_target()

    def fake_execute(self, body: bytes, headers: dict[str, str]):
        requests.append(json.loads(body.decode("utf-8")))
        return next(responses), "text/event-stream", 1.0, None

    async def fake_execute_async(self, body: bytes, headers: dict[str, str]):
        return fake_execute(self, body, headers)

    monkeypatch.setattr(ConversationalStreamingHttpTarget, "_execute_with_retries", fake_execute)
    monkeypatch.setattr(ConversationalStreamingHttpTarget, "_execute_async", fake_execute_async)

    golden = ConversationGolden(
        scenario="Create a Kubernetes connector in the AICHAT project",
        expected_outcome="Connector 'testconnector' created",
        max_turns=1,
        max_elicitation_rounds=6,
        initial_prompt="Create a k8s connector",
        elicitation_hints={
            "intents": {
                "auth_method": "Inherit from Delegate",
                "scope": "Project (Recommended)",
                "connector_name": "testconnector",
                "delegate_selector": "hello",
            },
            "matchers": [
                {"intent": "auth_method", "question_contains": ["auth", "authentication"]},
                {"intent": "scope", "question_contains": ["scope", "project"]},
                {"intent": "delegate_selector", "question_contains": ["delegate", "selector", "tag"]},
                {
                    "intent": "connector_name",
                    "question_contains": ["name would you like", "name for the", "identifier", "connector"],
                },
            ],
            "yaml": {"default_action": "accept"},
        },
        metadata={
            "sse_checks": [
                {"event": "elicitation_form", "exists": True},
                {"event": "elicitation_yaml", "exists": True},
                {"event": "entity_mutation", "path": "$.resource_type", "equals": "connector"},
                {"event": "entity_mutation", "path": "$.identifier", "equals": "testconnector"},
                {"event": "assistant_message", "exists": True},
            ]
        },
    )

    from examples.harness_sse_elicitation_adapter import HarnessSseElicitationAdapter

    from harness_evals.conversation.human_input import HumanInputSimulator

    simulator = ConversationSimulator(
        simulator_llm=StopLLM(),
        human_input_simulator=HumanInputSimulator(adapter=HarnessSseElicitationAdapter()),
    )

    async def agent_fn(messages: list[Message], *, human_input: dict | None = None) -> Message:
        return await target.agenerate(messages, human_input=human_input)

    result = await simulator.simulate(golden, agent_fn)

    assert len(requests) == 6
    assert requests[0]["prompt"] == "Create a k8s connector"
    assert "system_event" in requests[1]
    assert result.output == "Connector created."
    sse_events = result.metadata["sse_events"]
    assert sse_events["entity_mutation"][0]["identifier"] == "testconnector"
    assert "elicitation_form" in sse_events
    assert "elicitation_yaml" in sse_events
    assert "entity_mutation" in sse_events
    assert "assistant_message" in sse_events
