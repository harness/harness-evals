"""Tests for ConversationSimulator elicitation sub-loop."""

import pytest
from examples.harness_sse_elicitation_adapter import ElicitationSimulator

from harness_evals.conversation import ConversationGolden, ConversationSimulator
from harness_evals.core.types import Message
from harness_evals.llm.base import BaseLLM


class StopLLM(BaseLLM):
    async def generate(self, prompt: str, **kwargs) -> str:
        return ""

    async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
        return {"achieved": True, "reasoning": "done"}


@pytest.mark.unit
async def test_simulator_uses_initial_prompt_and_resolves_elicitation():
    golden = ConversationGolden(
        scenario="Create a k8s connector",
        expected_outcome="Connector created",
        max_turns=1,
        max_elicitation_rounds=3,
        initial_prompt="Create a k8s connector",
        elicitation_hints={
            "intents": {"connector_name": "testconnector"},
            "matchers": [{"intent": "connector_name", "question_contains": ["name", "connector"]}],
        },
    )
    calls: list[dict | None] = []

    async def agent_fn(messages: list[Message], system_event: dict | None = None) -> Message:
        calls.append(system_event)
        assert messages[-1].content == "Create a k8s connector"
        if system_event is None:
            return Message(
                role="assistant",
                metadata={
                    "pending_elicitation": {
                        "type": "elicitation_free_text",
                        "payload": {
                            "review_id": "ask-name",
                            "content": {"question": "What name would you like for the connector?"},
                        },
                    }
                },
            )
        assert system_event["capability_id"] == "ask-name"
        assert system_event["result"]["free_text"] == "testconnector"
        return Message(role="assistant", content="Connector created.")

    simulator = ConversationSimulator(simulator_llm=StopLLM(), elicitation_simulator=ElicitationSimulator())
    eval_case = await simulator.simulate(golden, agent_fn)

    assert len(calls) == 2
    assert eval_case.output == "Connector created."
    assert eval_case.messages == [
        Message(role="user", content="Create a k8s connector"),
        Message(role="assistant", content="Connector created."),
    ]


@pytest.mark.unit
async def test_simulator_stops_elicitation_loop_at_round_cap():
    golden = ConversationGolden(
        scenario="Create a k8s connector",
        expected_outcome="Connector created",
        max_turns=1,
        max_elicitation_rounds=2,
        initial_prompt="Create a k8s connector",
        elicitation_hints={
            "intents": {"connector_name": "testconnector"},
            "matchers": [{"intent": "connector_name", "question_contains": ["name", "connector"]}],
        },
    )

    async def agent_fn(messages: list[Message], system_event: dict | None = None) -> Message:
        return Message(
            role="assistant",
            metadata={
                "pending_elicitation": {
                    "type": "elicitation_free_text",
                    "payload": {
                        "review_id": "ask-name",
                        "content": {"question": "What name would you like for the connector?"},
                    },
                }
            },
        )

    simulator = ConversationSimulator(simulator_llm=StopLLM(), elicitation_simulator=ElicitationSimulator())
    eval_case = await simulator.simulate(golden, agent_fn)

    final = eval_case.messages[-1]
    assert final.metadata["elicitation_error"] == "max_elicitation_rounds_exceeded"
    assert final.metadata["elicitation_rounds"] == 2
