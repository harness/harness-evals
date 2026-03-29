"""Tests for conversation metrics."""

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.types import Message, ToolCall
from harness_evals.metrics.conversation.coherence import ConversationCoherenceMetric
from harness_evals.metrics.conversation.conversation_completeness import (
    ConversationCompletenessMetric,
)
from harness_evals.metrics.conversation.goal_accuracy import GoalAccuracyMetric
from harness_evals.metrics.conversation.knowledge_retention import (
    KnowledgeRetentionMetric,
)
from harness_evals.metrics.conversation.resolution import ConversationResolutionMetric
from harness_evals.metrics.conversation.role_adherence import RoleAdherenceMetric
from harness_evals.metrics.conversation.tool_use import ToolUseMetric
from harness_evals.metrics.conversation.topic_adherence import TopicAdherenceMetric
from harness_evals.metrics.conversation.turn_efficiency import TurnEfficiencyMetric
from harness_evals.metrics.conversation.turn_relevancy import TurnRelevancyMetric
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


# ---------------------------------------------------------------------------
# ConversationCompletenessMetric
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConversationCompleteness:
    async def test_complete_conversation(self):
        llm = MockLLM(default={"reasoning": "All intents addressed", "score": 0.95})
        metric = ConversationCompletenessMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="q", output="a", messages=COHERENT_MESSAGES)
        score = await metric.a_measure(ec)
        assert score.passed
        assert score.value == 0.95
        assert score.metadata["n_turns"] == 4

    async def test_incomplete_conversation(self):
        llm = MockLLM(default={"reasoning": "User question unanswered", "score": 0.2})
        metric = ConversationCompletenessMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="q", output="a", messages=INCOHERENT_MESSAGES)
        score = await metric.a_measure(ec)
        assert not score.passed

    async def test_missing_messages(self):
        llm = MockLLM()
        metric = ConversationCompletenessMetric(llm=llm)
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "missing" in score.reason

    async def test_single_turn(self):
        llm = MockLLM()
        metric = ConversationCompletenessMetric(llm=llm)
        ec = EvalCase(input="q", output="a", messages=[Message(role="user", content="hi")])
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "fewer than 2" in score.reason

    def test_sync_measure(self):
        llm = MockLLM(default={"reasoning": "ok", "score": 0.8})
        metric = ConversationCompletenessMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="q", output="a", messages=COHERENT_MESSAGES)
        score = metric.measure(ec)
        assert score.passed


# ---------------------------------------------------------------------------
# TurnRelevancyMetric
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTurnRelevancy:
    async def test_all_relevant(self):
        llm = MockLLM(default={"reasoning": "All responses on topic", "score": 1.0})
        metric = TurnRelevancyMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="q", output="a", messages=COHERENT_MESSAGES)
        score = await metric.a_measure(ec)
        assert score.passed
        assert score.value == 1.0

    async def test_irrelevant_responses(self):
        llm = MockLLM(default={"reasoning": "Off-topic replies", "score": 0.1})
        metric = TurnRelevancyMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="q", output="a", messages=INCOHERENT_MESSAGES)
        score = await metric.a_measure(ec)
        assert not score.passed

    async def test_missing_messages(self):
        llm = MockLLM()
        metric = TurnRelevancyMetric(llm=llm)
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "missing" in score.reason

    async def test_single_turn(self):
        llm = MockLLM()
        metric = TurnRelevancyMetric(llm=llm)
        ec = EvalCase(input="q", output="a", messages=[Message(role="user", content="hi")])
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "fewer than 2" in score.reason

    def test_sync_measure(self):
        llm = MockLLM(default={"reasoning": "ok", "score": 0.9})
        metric = TurnRelevancyMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="q", output="a", messages=COHERENT_MESSAGES)
        score = metric.measure(ec)
        assert score.passed


# ---------------------------------------------------------------------------
# KnowledgeRetentionMetric
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKnowledgeRetention:
    async def test_good_retention(self):
        llm = MockLLM(default={"reasoning": "Remembers all context", "score": 0.95})
        metric = KnowledgeRetentionMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="q", output="a", messages=COHERENT_MESSAGES)
        score = await metric.a_measure(ec)
        assert score.passed
        assert score.value == 0.95

    async def test_poor_retention(self):
        llm = MockLLM(default={"reasoning": "Forgot earlier info", "score": 0.2})
        metric = KnowledgeRetentionMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="q", output="a", messages=INCOHERENT_MESSAGES)
        score = await metric.a_measure(ec)
        assert not score.passed

    async def test_missing_messages(self):
        llm = MockLLM()
        metric = KnowledgeRetentionMetric(llm=llm)
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "missing" in score.reason

    async def test_single_turn(self):
        llm = MockLLM()
        metric = KnowledgeRetentionMetric(llm=llm)
        ec = EvalCase(input="q", output="a", messages=[Message(role="user", content="hi")])
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "fewer than 2" in score.reason

    def test_sync_measure(self):
        llm = MockLLM(default={"reasoning": "ok", "score": 0.8})
        metric = KnowledgeRetentionMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="q", output="a", messages=COHERENT_MESSAGES)
        score = metric.measure(ec)
        assert score.passed


# ---------------------------------------------------------------------------
# RoleAdherenceMetric
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRoleAdherence:
    async def test_good_adherence(self):
        llm = MockLLM(default={"reasoning": "Stays in character", "score": 0.95})
        metric = RoleAdherenceMetric(llm=llm, threshold=0.7)
        ec = EvalCase(
            input="q", output="a",
            messages=COHERENT_MESSAGES,
            metadata={"chatbot_role": "A helpful geography tutor"},
        )
        score = await metric.a_measure(ec)
        assert score.passed
        assert score.value == 0.95
        assert score.metadata["chatbot_role"] == "A helpful geography tutor"

    async def test_poor_adherence(self):
        llm = MockLLM(default={"reasoning": "Broke character", "score": 0.2})
        metric = RoleAdherenceMetric(llm=llm, threshold=0.7)
        ec = EvalCase(
            input="q", output="a",
            messages=INCOHERENT_MESSAGES,
            metadata={"chatbot_role": "A geography tutor"},
        )
        score = await metric.a_measure(ec)
        assert not score.passed

    async def test_missing_messages(self):
        llm = MockLLM()
        metric = RoleAdherenceMetric(llm=llm)
        ec = EvalCase(input="q", output="a", metadata={"chatbot_role": "tutor"})
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "missing" in score.reason

    async def test_missing_role(self):
        llm = MockLLM()
        metric = RoleAdherenceMetric(llm=llm)
        ec = EvalCase(input="q", output="a", messages=COHERENT_MESSAGES)
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "chatbot_role" in score.reason

    async def test_single_turn(self):
        llm = MockLLM()
        metric = RoleAdherenceMetric(llm=llm)
        ec = EvalCase(
            input="q", output="a",
            messages=[Message(role="user", content="hi")],
            metadata={"chatbot_role": "tutor"},
        )
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "fewer than 2" in score.reason

    def test_sync_measure(self):
        llm = MockLLM(default={"reasoning": "ok", "score": 0.8})
        metric = RoleAdherenceMetric(llm=llm, threshold=0.7)
        ec = EvalCase(
            input="q", output="a",
            messages=COHERENT_MESSAGES,
            metadata={"chatbot_role": "tutor"},
        )
        score = metric.measure(ec)
        assert score.passed


# ---------------------------------------------------------------------------
# TopicAdherenceMetric
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTopicAdherence:
    async def test_on_topic(self):
        llm = MockLLM(default={"reasoning": "All on allowed topics", "score": 1.0})
        metric = TopicAdherenceMetric(llm=llm, allowed_topics=["geography", "demographics"], threshold=0.7)
        ec = EvalCase(input="q", output="a", messages=COHERENT_MESSAGES)
        score = await metric.a_measure(ec)
        assert score.passed
        assert score.value == 1.0
        assert score.metadata["allowed_topics"] == ["geography", "demographics"]

    async def test_off_topic(self):
        llm = MockLLM(default={"reasoning": "Answered off-topic questions", "score": 0.3})
        metric = TopicAdherenceMetric(llm=llm, allowed_topics=["math"], threshold=0.7)
        ec = EvalCase(input="q", output="a", messages=COHERENT_MESSAGES)
        score = await metric.a_measure(ec)
        assert not score.passed

    async def test_missing_messages(self):
        llm = MockLLM()
        metric = TopicAdherenceMetric(llm=llm, allowed_topics=["math"])
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "missing" in score.reason

    async def test_single_turn(self):
        llm = MockLLM()
        metric = TopicAdherenceMetric(llm=llm, allowed_topics=["math"])
        ec = EvalCase(input="q", output="a", messages=[Message(role="user", content="hi")])
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "fewer than 2" in score.reason

    def test_sync_measure(self):
        llm = MockLLM(default={"reasoning": "ok", "score": 0.9})
        metric = TopicAdherenceMetric(llm=llm, allowed_topics=["geography"], threshold=0.7)
        ec = EvalCase(input="q", output="a", messages=COHERENT_MESSAGES)
        score = metric.measure(ec)
        assert score.passed


# ---------------------------------------------------------------------------
# GoalAccuracyMetric
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGoalAccuracy:
    async def test_goal_achieved(self):
        llm = MockLLM(default={"reasoning": "Goal fully achieved", "score": 1.0})
        metric = GoalAccuracyMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="q", output="a", messages=COHERENT_MESSAGES)
        score = await metric.a_measure(ec)
        assert score.passed
        assert score.value == 1.0

    async def test_goal_not_achieved(self):
        llm = MockLLM(default={"reasoning": "Goal not met", "score": 0.1})
        metric = GoalAccuracyMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="q", output="a", messages=INCOHERENT_MESSAGES)
        score = await metric.a_measure(ec)
        assert not score.passed

    async def test_missing_messages(self):
        llm = MockLLM()
        metric = GoalAccuracyMetric(llm=llm)
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "missing" in score.reason

    async def test_single_turn(self):
        llm = MockLLM()
        metric = GoalAccuracyMetric(llm=llm)
        ec = EvalCase(input="q", output="a", messages=[Message(role="user", content="hi")])
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "fewer than 2" in score.reason

    def test_sync_measure(self):
        llm = MockLLM(default={"reasoning": "ok", "score": 0.85})
        metric = GoalAccuracyMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="q", output="a", messages=COHERENT_MESSAGES)
        score = metric.measure(ec)
        assert score.passed


# ---------------------------------------------------------------------------
# ToolUseMetric
# ---------------------------------------------------------------------------

MESSAGES_WITH_TOOLS = [
    Message(role="user", content="Search for flights to Paris"),
    Message(
        role="assistant",
        content="Searching for flights...",
        tool_calls=[ToolCall(name="flight_search", input={"destination": "Paris"})],
    ),
    Message(role="user", content="Book the cheapest one"),
    Message(
        role="assistant",
        content="Booking flight AA123",
        tool_calls=[ToolCall(name="book_flight", input={"flight_id": "AA123"})],
    ),
]


@pytest.mark.unit
class TestToolUse:
    async def test_good_tool_use(self):
        llm = MockLLM(default={"reasoning": "Tools used correctly", "score": 0.95})
        metric = ToolUseMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="q", output="a", messages=MESSAGES_WITH_TOOLS)
        score = await metric.a_measure(ec)
        assert score.passed
        assert score.value == 0.95
        assert score.metadata["n_tool_calls"] == 2

    async def test_poor_tool_use(self):
        llm = MockLLM(default={"reasoning": "Wrong tool selected", "score": 0.2})
        metric = ToolUseMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="q", output="a", messages=MESSAGES_WITH_TOOLS)
        score = await metric.a_measure(ec)
        assert not score.passed

    async def test_no_tool_calls_in_messages(self):
        llm = MockLLM()
        metric = ToolUseMetric(llm=llm)
        ec = EvalCase(input="q", output="a", messages=COHERENT_MESSAGES)
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "tool calls" in score.reason.lower()

    async def test_missing_messages(self):
        llm = MockLLM()
        metric = ToolUseMetric(llm=llm)
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "missing" in score.reason

    async def test_single_turn(self):
        llm = MockLLM()
        metric = ToolUseMetric(llm=llm)
        ec = EvalCase(input="q", output="a", messages=[Message(role="user", content="hi")])
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "fewer than 2" in score.reason

    def test_sync_measure(self):
        llm = MockLLM(default={"reasoning": "ok", "score": 0.9})
        metric = ToolUseMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="q", output="a", messages=MESSAGES_WITH_TOOLS)
        score = metric.measure(ec)
        assert score.passed
