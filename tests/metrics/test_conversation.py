"""Tests for conversation metrics: Coherence, Resolution, TurnEfficiency."""

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.types import Message
from harness_evals.metrics.conversation.coherence import ConversationCoherenceMetric
from harness_evals.metrics.conversation.resolution import ConversationResolutionMetric
from harness_evals.metrics.conversation.turn_efficiency import TurnEfficiencyMetric
from tests.conftest import MockLLM

COHERENT_MESSAGES = [
    Message(role="user", content="What is the capital of France?"),
    Message(role="assistant", content="The capital of France is Paris."),
    Message(role="user", content="What is its population?"),
    Message(role="assistant", content="Paris has a population of about 2.1 million."),
]

INCOHERENT_MESSAGES = [
    Message(role="user", content="What is the capital of France?"),
    Message(role="assistant", content="I like pizza."),
    Message(role="user", content="That doesn't answer my question."),
    Message(role="assistant", content="The weather is nice today."),
]


# ---------------------------------------------------------------------------
# ConversationCoherenceMetric
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConversationCoherence:
    async def test_coherent_conversation(self):
        llm = MockLLM(default={"reasoning": "Conversation stays on topic", "score": 0.95})
        metric = ConversationCoherenceMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="q", output="a", messages=COHERENT_MESSAGES)
        score = await metric.a_measure(ec)
        assert score.passed
        assert score.value == 0.95
        assert score.metadata["n_turns"] == 4

    async def test_incoherent_conversation(self):
        llm = MockLLM(default={"reasoning": "Off-topic responses", "score": 0.2})
        metric = ConversationCoherenceMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="q", output="a", messages=INCOHERENT_MESSAGES)
        score = await metric.a_measure(ec)
        assert not score.passed
        assert score.value == 0.2

    async def test_missing_messages(self):
        llm = MockLLM()
        metric = ConversationCoherenceMetric(llm=llm)
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "missing" in score.reason

    async def test_single_turn(self):
        llm = MockLLM()
        metric = ConversationCoherenceMetric(llm=llm)
        ec = EvalCase(
            input="q",
            output="a",
            messages=[Message(role="user", content="hello")],
        )
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "fewer than 2" in score.reason

    async def test_score_clamped(self):
        llm = MockLLM(default={"reasoning": "edge", "score": 1.5})
        metric = ConversationCoherenceMetric(llm=llm)
        ec = EvalCase(input="q", output="a", messages=COHERENT_MESSAGES)
        score = await metric.a_measure(ec)
        assert score.value == 1.0

    def test_sync_measure(self):
        llm = MockLLM(default={"reasoning": "ok", "score": 0.8})
        metric = ConversationCoherenceMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="q", output="a", messages=COHERENT_MESSAGES)
        score = metric.measure(ec)
        assert score.passed


# ---------------------------------------------------------------------------
# ConversationResolutionMetric
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConversationResolution:
    async def test_resolved_conversation(self):
        llm = MockLLM(default={"reasoning": "User need fully addressed", "score": 1.0})
        metric = ConversationResolutionMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="q", output="a", messages=COHERENT_MESSAGES)
        score = await metric.a_measure(ec)
        assert score.passed
        assert score.value == 1.0

    async def test_unresolved_conversation(self):
        llm = MockLLM(default={"reasoning": "Question not answered", "score": 0.1})
        metric = ConversationResolutionMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="q", output="a", messages=INCOHERENT_MESSAGES)
        score = await metric.a_measure(ec)
        assert not score.passed
        assert score.value == 0.1

    async def test_missing_messages(self):
        llm = MockLLM()
        metric = ConversationResolutionMetric(llm=llm)
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "missing" in score.reason

    async def test_single_turn(self):
        llm = MockLLM()
        metric = ConversationResolutionMetric(llm=llm)
        ec = EvalCase(
            input="q",
            output="a",
            messages=[Message(role="user", content="hi")],
        )
        score = await metric.a_measure(ec)
        assert score.value == 0.0

    def test_sync_measure(self):
        llm = MockLLM(default={"reasoning": "resolved", "score": 0.9})
        metric = ConversationResolutionMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="q", output="a", messages=COHERENT_MESSAGES)
        score = metric.measure(ec)
        assert score.passed


# ---------------------------------------------------------------------------
# TurnEfficiencyMetric
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTurnEfficiency:
    def _make_messages(self, n: int) -> list[Message]:
        return [Message(role="user" if i % 2 == 0 else "assistant", content=f"msg{i}") for i in range(n)]

    def test_exact_turns(self):
        ec = EvalCase(input="q", output="a", messages=self._make_messages(5), metadata={"expected_turns": 5})
        score = TurnEfficiencyMetric().measure(ec)
        assert score.value == 1.0

    def test_extra_turns_penalized(self):
        ec = EvalCase(input="q", output="a", messages=self._make_messages(10), metadata={"expected_turns": 5})
        score = TurnEfficiencyMetric().measure(ec)
        assert score.value == pytest.approx(0.5)

    def test_fewer_turns_capped(self):
        ec = EvalCase(input="q", output="a", messages=self._make_messages(3), metadata={"expected_turns": 5})
        score = TurnEfficiencyMetric().measure(ec)
        assert score.value == 1.0

    def test_no_messages(self):
        ec = EvalCase(input="q", output="a", metadata={"expected_turns": 5})
        score = TurnEfficiencyMetric().measure(ec)
        assert score.value == 0.0
        assert "messages" in score.reason

    def test_missing_expected_turns(self):
        ec = EvalCase(input="q", output="a", messages=self._make_messages(5))
        score = TurnEfficiencyMetric().measure(ec)
        assert score.value == 0.0
        assert "expected_turns" in score.reason

    def test_no_metadata_no_messages(self):
        ec = EvalCase(input="q", output="a")
        score = TurnEfficiencyMetric().measure(ec)
        assert score.value == 0.0

    def test_zero_expected_turns(self):
        ec = EvalCase(input="q", output="a", messages=self._make_messages(5), metadata={"expected_turns": 0})
        score = TurnEfficiencyMetric().measure(ec)
        assert score.value == 0.0

    def test_threshold_applied(self):
        ec = EvalCase(input="q", output="a", messages=self._make_messages(10), metadata={"expected_turns": 5})
        score = TurnEfficiencyMetric(threshold=0.8).measure(ec)
        assert not score.passed

    def test_metadata_fields(self):
        ec = EvalCase(input="q", output="a", messages=self._make_messages(8), metadata={"expected_turns": 4})
        score = TurnEfficiencyMetric().measure(ec)
        assert score.metadata["actual_turns"] == 8
        assert score.metadata["expected_turns"] == 4

    def test_non_numeric_expected_turns(self):
        ec = EvalCase(input="q", output="a", messages=self._make_messages(5), metadata={"expected_turns": "five"})
        score = TurnEfficiencyMetric().measure(ec)
        assert score.value == 0.0
        assert "numeric" in score.reason
