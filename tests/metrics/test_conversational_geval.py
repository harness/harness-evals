"""Tests for ConversationalGEvalMetric."""

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.types import Message
from harness_evals.metrics.conversation.conversational_geval import (
    ConversationalGEvalMetric,
    MultiTurnView,
)
from tests.conftest import MockLLM

MESSAGES = [
    Message(role="user", content="What is Python?"),
    Message(role="assistant", content="Python is a programming language."),
    Message(role="user", content="What about its typing?"),
    Message(role="assistant", content="Python supports dynamic and optional static typing."),
]


@pytest.mark.unit
class TestConversationalGEval:
    async def test_basic_scoring(self):
        llm = MockLLM(default={"reasoning": "Good response", "score": 0.9})
        metric = ConversationalGEvalMetric(
            llm=llm,
            criteria="Is the response accurate and helpful?",
        )
        ec = EvalCase(input="q", output="a", messages=MESSAGES)
        score = await metric.a_measure(ec)

        assert score.passed
        assert score.value == 0.9
        assert score.metadata["n_assistant_turns"] == 2
        assert len(score.metadata["turn_scores"]) == 2

    async def test_per_turn_breakdown(self):
        llm = MockLLM(
            responses=[
                {"reasoning": "First turn good", "score": 0.8},
                {"reasoning": "Second turn excellent", "score": 1.0},
            ]
        )
        metric = ConversationalGEvalMetric(
            llm=llm,
            criteria="Accuracy",
        )
        ec = EvalCase(input="q", output="a", messages=MESSAGES)
        score = await metric.a_measure(ec)

        assert score.value == 0.9  # mean of 0.8 and 1.0
        assert score.metadata["turn_scores"][0]["score"] == 0.8
        assert score.metadata["turn_scores"][1]["score"] == 1.0
        assert score.metadata["turn_scores"][0]["reasoning"] == "First turn good"

    async def test_missing_messages(self):
        llm = MockLLM()
        metric = ConversationalGEvalMetric(llm=llm, criteria="test")
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "missing" in score.reason

    async def test_single_turn(self):
        llm = MockLLM()
        metric = ConversationalGEvalMetric(llm=llm, criteria="test")
        ec = EvalCase(
            input="q",
            output="a",
            messages=[Message(role="user", content="hi")],
        )
        score = await metric.a_measure(ec)
        assert score.value == 0.0

    async def test_no_assistant_turns(self):
        llm = MockLLM()
        metric = ConversationalGEvalMetric(llm=llm, criteria="test")
        ec = EvalCase(
            input="q",
            output="a",
            messages=[
                Message(role="user", content="hi"),
                Message(role="user", content="hello?"),
            ],
        )
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "no assistant turns" in score.reason

    async def test_sliding_window_view(self):
        long_messages = []
        for i in range(10):
            long_messages.append(Message(role="user", content=f"Question {i}"))
            long_messages.append(Message(role="assistant", content=f"Answer {i}"))

        llm = MockLLM(default={"reasoning": "ok", "score": 0.85})
        metric = ConversationalGEvalMetric(
            llm=llm,
            criteria="Relevancy",
            view=MultiTurnView.SLIDING_WINDOW,
            window_size=3,
        )
        ec = EvalCase(input="q", output="a", messages=long_messages)
        score = await metric.a_measure(ec)

        assert score.passed
        assert score.metadata["n_assistant_turns"] == 10

    async def test_evaluation_steps(self):
        llm = MockLLM(default={"reasoning": "followed steps", "score": 0.75})
        metric = ConversationalGEvalMetric(
            llm=llm,
            criteria="Helpfulness",
            evaluation_steps=["Check accuracy", "Check completeness"],
        )
        ec = EvalCase(input="q", output="a", messages=MESSAGES)
        score = await metric.a_measure(ec)
        assert score.value == 0.75

    async def test_score_clamping(self):
        llm = MockLLM(default={"reasoning": "test", "score": 1.5})
        metric = ConversationalGEvalMetric(llm=llm, criteria="test")
        ec = EvalCase(input="q", output="a", messages=MESSAGES)
        score = await metric.a_measure(ec)
        assert score.value == 1.0

    async def test_negative_score_clamping(self):
        llm = MockLLM(default={"reasoning": "test", "score": -0.5})
        metric = ConversationalGEvalMetric(llm=llm, criteria="test")
        ec = EvalCase(input="q", output="a", messages=MESSAGES)
        score = await metric.a_measure(ec)
        assert score.value == 0.0

    def test_sync_measure(self):
        llm = MockLLM(default={"reasoning": "sync works", "score": 0.8})
        metric = ConversationalGEvalMetric(llm=llm, criteria="test")
        ec = EvalCase(input="q", output="a", messages=MESSAGES)
        score = metric.measure(ec)
        assert score.passed
        assert score.value == 0.8

    async def test_threshold_customization(self):
        llm = MockLLM(default={"reasoning": "ok", "score": 0.6})
        metric = ConversationalGEvalMetric(llm=llm, criteria="test", threshold=0.5)
        ec = EvalCase(input="q", output="a", messages=MESSAGES)
        score = await metric.a_measure(ec)
        assert score.passed  # 0.6 > 0.5
