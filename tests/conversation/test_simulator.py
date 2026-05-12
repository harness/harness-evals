"""Tests for ConversationSimulator."""

import pytest

from harness_evals.conversation import ConversationGolden, ConversationSimulator
from harness_evals.core.types import Message
from tests.conftest import MockLLM


class SimulatorMockLLM(MockLLM):
    """Mock LLM that returns user messages via generate() and stop checks via generate_json()."""

    def __init__(self, user_messages: list[str], stop_after: int = 2):
        self._user_messages = user_messages
        self._user_idx = 0
        self._json_call_count = 0
        self._stop_after = stop_after
        super().__init__()

    async def generate(self, prompt: str, **kwargs) -> str:
        if self._user_idx < len(self._user_messages):
            msg = self._user_messages[self._user_idx]
            self._user_idx += 1
            return msg
        return "Thank you, that's all."

    async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
        self._json_call_count += 1
        achieved = self._json_call_count >= self._stop_after
        return {"achieved": achieved, "reasoning": "test"}


@pytest.mark.unit
class TestConversationSimulator:
    async def test_basic_simulation(self):
        llm = SimulatorMockLLM(
            user_messages=["What is your refund policy?", "How long does it take?"],
            stop_after=2,
        )

        async def agent_fn(messages: list[Message]) -> Message:
            return Message(role="assistant", content=f"Response to turn {len(messages)}")

        golden = ConversationGolden(
            scenario="Ask about refund policy",
            expected_outcome="Agent explains refund process",
        )

        simulator = ConversationSimulator(simulator_llm=llm)
        result = await simulator.simulate(golden, agent_fn)

        assert result.messages is not None
        assert len(result.messages) >= 4  # At least 2 user + 2 assistant turns
        assert result.input == "Ask about refund policy"
        assert result.metadata["scenario"] == "Ask about refund policy"
        assert result.metadata["expected_outcome"] == "Agent explains refund process"

    async def test_max_turns_cap(self):
        llm = SimulatorMockLLM(
            user_messages=["msg"] * 20,
            stop_after=100,  # Never stops naturally
        )

        async def agent_fn(messages: list[Message]) -> Message:
            return Message(role="assistant", content="response")

        golden = ConversationGolden(
            scenario="Test",
            expected_outcome="Outcome",
            max_turns=4,  # Cap at 4 turns total (2 exchanges)
        )

        simulator = ConversationSimulator(simulator_llm=llm)
        result = await simulator.simulate(golden, agent_fn)

        assert len(result.messages) <= 8  # max_turns iterations, each adds 2 messages

    async def test_replay_mode(self):
        llm = MockLLM()  # Should not be called
        turns = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="How are you?"),
            Message(role="assistant", content="I'm doing well, thanks!"),
        ]

        golden = ConversationGolden(
            scenario="Greeting",
            expected_outcome="Polite exchange",
            turns=turns,
        )

        simulator = ConversationSimulator(simulator_llm=llm)
        result = await simulator.simulate(golden, agent_fn=lambda x: None)

        assert result.messages == turns
        assert result.output == "I'm doing well, thanks!"
        assert result.metadata["n_turns"] == 4

    async def test_output_is_last_assistant_message(self):
        llm = SimulatorMockLLM(user_messages=["question"], stop_after=1)

        async def agent_fn(messages: list[Message]) -> Message:
            return Message(role="assistant", content="final answer")

        golden = ConversationGolden(
            scenario="Test",
            expected_outcome="Done",
        )

        simulator = ConversationSimulator(simulator_llm=llm)
        result = await simulator.simulate(golden, agent_fn)

        assert result.output == "final answer"

    async def test_simulate_batch(self):
        llm = SimulatorMockLLM(
            user_messages=["q1", "q2", "q3", "q4"],
            stop_after=1,
        )

        async def agent_fn(messages: list[Message]) -> Message:
            return Message(role="assistant", content="answer")

        goldens = [ConversationGolden(scenario=f"S{i}", expected_outcome=f"O{i}") for i in range(3)]

        simulator = ConversationSimulator(simulator_llm=llm, max_concurrent=2)
        results = await simulator.simulate_batch(goldens, agent_fn)

        assert len(results) == 3
        for r in results:
            assert r.messages is not None

    async def test_metadata_from_golden(self):
        llm = SimulatorMockLLM(user_messages=["hi"], stop_after=1)

        async def agent_fn(messages: list[Message]) -> Message:
            return Message(role="assistant", content="hello")

        golden = ConversationGolden(
            scenario="Test",
            expected_outcome="Done",
            metadata={"custom_key": "custom_value"},
            tags={"env": "test"},
        )

        simulator = ConversationSimulator(simulator_llm=llm)
        result = await simulator.simulate(golden, agent_fn)

        assert result.metadata["custom_key"] == "custom_value"
        assert result.tags == {"env": "test"}
