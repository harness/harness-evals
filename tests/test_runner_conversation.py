"""Tests for evaluate_dataset() with ConversationGolden inputs."""

import pytest

from harness_evals.conversation.golden import ConversationGolden
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.golden import Golden
from harness_evals.core.runner import evaluate_dataset
from harness_evals.core.score import Score
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

    async def test_conversation_goldens_with_simulator_llm(self):
        llm = SimulatorMockLLM()
        goldens = [
            ConversationGolden(scenario="Ask about refund", expected_outcome="Explained"),
        ]
        metrics = [ExactMatchMetric()]
        results = await evaluate_dataset(
            goldens,
            mock_agent_fn_conv,
            metrics,
            simulator_llm=llm,
        )
        assert len(results) == 1
        assert isinstance(results[0], list)

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
