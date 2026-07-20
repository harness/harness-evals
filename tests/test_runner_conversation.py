"""Tests for evaluate_dataset() with ConversationGolden inputs."""

import pytest

from harness_evals.conversation.golden import ConversationGolden, ConversationMode
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.golden import Golden
from harness_evals.core.runner import evaluate_dataset
from harness_evals.core.types import Message
from harness_evals.metrics.deterministic.exact_match import ExactMatchMetric
from tests.conftest import MockLLM


class SimulatorMockLLM(MockLLM):
    async def generate(self, prompt: str, **kwargs) -> str:
        return "What is the refund policy?"

    async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
        return {"achieved": True, "reasoning": "done"}


async def mock_agent_fn_single(golden: Golden) -> EvalCase:
    return EvalCase(input=str(golden.input), output="answer")


async def mock_agent_fn_conv(messages: list[Message]) -> Message:
    return Message(role="assistant", content="Here is your answer.")


@pytest.mark.unit
class TestEvaluateDatasetWithConversationGolden:
    async def test_single_turn_goldens_unchanged(self):
        goldens = [Golden(input="q1", expected="a1"), Golden(input="q2", expected="a2")]
        metrics = [ExactMatchMetric()]
        results = await evaluate_dataset(goldens, mock_agent_fn_single, metrics)
        assert len(results) == 2
        assert all(isinstance(scores, list) for scores in results)

    async def test_conversation_goldens_require_simulator_llm(self):
        goldens = [ConversationGolden(scenario="test", expected_outcome="done")]
        with pytest.raises(ValueError, match="simulator_llm"):
            await evaluate_dataset(goldens, mock_agent_fn_conv, [])

    async def test_scripted_mode_runs_without_simulator_llm(self):
        """SCRIPTED mode never calls the simulator LLM, so simulator_llm=None is allowed."""
        goldens = [
            ConversationGolden(
                scenario="Greeting",
                expected_outcome="Polite exchange",
                mode=ConversationMode.SCRIPTED,
                turns=[Message(role="user", content="Hi")],
            )
        ]
        results = await evaluate_dataset(goldens, mock_agent_fn_conv, [ExactMatchMetric()], simulator_llm=None)
        assert len(results) == 1
        assert isinstance(results[0], list)

    async def test_replay_mode_runs_without_simulator_llm(self):
        """REPLAY mode replays stored turns with no LLM, so simulator_llm=None is allowed."""
        goldens = [
            ConversationGolden(
                scenario="Greeting",
                expected_outcome="Polite exchange",
                mode=ConversationMode.REPLAY,
                turns=[
                    Message(role="user", content="Hi"),
                    Message(role="assistant", content="Hello!"),
                ],
            )
        ]
        results = await evaluate_dataset(goldens, mock_agent_fn_conv, [ExactMatchMetric()], simulator_llm=None)
        assert len(results) == 1
        assert isinstance(results[0], list)

    async def test_mixed_scripted_and_replay_run_without_simulator_llm(self):
        """A batch of only SCRIPTED/REPLAY goldens needs no simulator LLM."""
        goldens = [
            ConversationGolden(
                scenario="a",
                expected_outcome="o",
                mode=ConversationMode.SCRIPTED,
                turns=[Message(role="user", content="Hi")],
            ),
            ConversationGolden(
                scenario="b",
                expected_outcome="o",
                mode=ConversationMode.REPLAY,
                turns=[Message(role="user", content="Hi"), Message(role="assistant", content="Yo")],
            ),
        ]
        results = await evaluate_dataset(goldens, mock_agent_fn_conv, [ExactMatchMetric()], simulator_llm=None)
        assert len(results) == 2

    async def test_graph_mode_requires_simulator_llm(self):
        """GRAPH mode drives LLM user turns, so simulator_llm=None must raise."""
        goldens = [
            ConversationGolden(
                scenario="test",
                expected_outcome="done",
                mode=ConversationMode.GRAPH,
            )
        ]
        with pytest.raises(ValueError, match="simulator_llm"):
            await evaluate_dataset(goldens, mock_agent_fn_conv, [], simulator_llm=None)

    async def test_batch_with_one_simulate_golden_requires_llm(self):
        """If any golden needs an LLM (SIMULATE), the whole batch requires simulator_llm."""
        goldens = [
            ConversationGolden(
                scenario="scripted",
                expected_outcome="o",
                mode=ConversationMode.SCRIPTED,
                turns=[Message(role="user", content="Hi")],
            ),
            ConversationGolden(scenario="simulate", expected_outcome="o"),  # default SIMULATE
        ]
        with pytest.raises(ValueError, match="simulator_llm"):
            await evaluate_dataset(goldens, mock_agent_fn_conv, [], simulator_llm=None)

    async def test_conversation_goldens_with_simulator_llm(self):
        """SIMULATE mode generates one user turn then stops (mock returns achieved=True)."""
        llm = SimulatorMockLLM()
        call_count = 0

        async def counting_agent(messages: list[Message]) -> Message:
            nonlocal call_count
            call_count += 1
            return Message(role="assistant", content="Here is your answer.")

        goldens = [
            ConversationGolden(scenario="Ask about refund", expected_outcome="Explained", max_turns=5),
        ]
        metrics = [ExactMatchMetric()]
        results = await evaluate_dataset(
            goldens,
            counting_agent,
            metrics,
            simulator_llm=llm,
        )
        assert len(results) == 1
        assert isinstance(results[0], list)
        # SimulatorMockLLM returns achieved=True immediately, so only 1 turn should execute
        assert call_count == 1

    async def test_mixed_goldens_raises(self):
        """Mixed lists (Golden + ConversationGolden) should raise TypeError."""
        mixed = [Golden(input="q", expected="a"), ConversationGolden(scenario="s", expected_outcome="o")]
        with pytest.raises(TypeError, match="mixed"):
            await evaluate_dataset(mixed, mock_agent_fn_conv, [], simulator_llm=MockLLM())

    async def test_conversation_golden_replay_mode(self):
        """Pre-scripted turns skip simulation and use replay mode."""
        llm = MockLLM()  # Should not be called for simulation
        turns = [
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello!"),
        ]
        goldens = [
            ConversationGolden(
                scenario="Greeting",
                expected_outcome="Polite exchange",
                turns=turns,
            )
        ]
        metrics = [ExactMatchMetric()]
        results = await evaluate_dataset(
            goldens,
            mock_agent_fn_conv,
            metrics,
            simulator_llm=llm,
        )
        assert len(results) == 1

    async def test_scripted_mode_calls_agent(self):
        """SCRIPTED mode feeds user turns to agent and collects responses."""
        llm = SimulatorMockLLM()
        call_count = 0

        async def counting_agent(messages: list[Message]) -> Message:
            nonlocal call_count
            call_count += 1
            return Message(role="assistant", content=f"Response {call_count}")

        user_turns = [
            Message(role="user", content="Hello"),
            Message(role="user", content="How are you?"),
            Message(role="user", content="Goodbye"),
        ]
        goldens = [
            ConversationGolden(
                scenario="Multi-turn greeting",
                expected_outcome="Polite exchange",
                turns=user_turns,
                mode=ConversationMode.SCRIPTED,
            )
        ]
        metrics = [ExactMatchMetric()]
        results = await evaluate_dataset(
            goldens,
            counting_agent,
            metrics,
            simulator_llm=llm,
        )
        assert len(results) == 1
        assert call_count == 3

    async def test_scripted_mode_requires_turns(self):
        """SCRIPTED mode without turns raises ValueError."""
        with pytest.raises(ValueError, match="requires 'turns'"):
            ConversationGolden(
                scenario="test",
                expected_outcome="done",
                mode=ConversationMode.SCRIPTED,
            )

    async def test_replay_mode_requires_turns(self):
        """REPLAY mode without turns raises ValueError."""
        with pytest.raises(ValueError, match="requires 'turns'"):
            ConversationGolden(
                scenario="test",
                expected_outcome="done",
                mode=ConversationMode.REPLAY,
            )

    async def test_mode_enum_serialization(self):
        """ConversationMode round-trips through to_dict/from_dict."""
        turns = [Message(role="user", content="Hi")]
        golden = ConversationGolden(
            scenario="test",
            expected_outcome="done",
            turns=turns,
            mode=ConversationMode.SCRIPTED,
        )
        d = golden.to_dict()
        assert d["mode"] == "scripted"

        restored = ConversationGolden.from_dict(d)
        assert restored.mode == ConversationMode.SCRIPTED
