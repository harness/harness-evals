"""Tests for per-turn operational metrics."""

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.types import Message
from harness_evals.metrics.operational.turn_latency import TurnLatencyMetric
from harness_evals.metrics.operational.turn_token_cost import TurnTokenCostMetric

MESSAGES_WITH_LATENCY = [
    Message(role="user", content="q1"),
    Message(role="assistant", content="a1", latency_ms=200.0),
    Message(role="user", content="q2"),
    Message(role="assistant", content="a2", latency_ms=400.0),
]

MESSAGES_WITH_TOKENS = [
    Message(role="user", content="q1"),
    Message(role="assistant", content="a1", token_count=50),
    Message(role="user", content="q2"),
    Message(role="assistant", content="a2", token_count=150),
]


@pytest.mark.unit
class TestTurnLatencyMetric:
    def test_within_budget(self):
        ec = EvalCase(input="q", output="a", messages=MESSAGES_WITH_LATENCY)
        score = TurnLatencyMetric(max_ms_per_turn=500, threshold=0.3).measure(ec)
        assert score.passed
        assert score.value == pytest.approx(0.4)  # mean of (1-200/500, 1-400/500) = mean(0.6, 0.2) = 0.4
        assert score.metadata["turn_latencies"] == [200.0, 400.0]
        assert score.metadata["mean_latency_ms"] == 300.0

    def test_over_budget(self):
        ec = EvalCase(input="q", output="a", messages=MESSAGES_WITH_LATENCY)
        score = TurnLatencyMetric(max_ms_per_turn=300, threshold=0.8).measure(ec)
        assert not score.passed  # mean = (1-200/300 + 0.0) / 2 = 0.333
        assert score.value < 0.5

    def test_no_messages(self):
        ec = EvalCase(input="q", output="a")
        score = TurnLatencyMetric().measure(ec)
        assert score.value == 0.0
        assert "no messages" in score.reason

    def test_no_assistant_latency_data(self):
        ec = EvalCase(input="q", output="a", messages=[
            Message(role="user", content="q"),
            Message(role="assistant", content="a"),  # no latency_ms
        ])
        score = TurnLatencyMetric().measure(ec)
        assert score.value == 0.0
        assert "no latency" in score.reason

    def test_partial_latency_data_skips_missing(self):
        messages = [
            Message(role="user", content="q1"),
            Message(role="assistant", content="a1", latency_ms=200.0),
            Message(role="user", content="q2"),
            Message(role="assistant", content="a2"),  # missing latency
        ]
        ec = EvalCase(input="q", output="a", messages=messages)
        score = TurnLatencyMetric(max_ms_per_turn=500).measure(ec)
        # Only turn 1 scored: (1 - 200/500) = 0.6
        assert score.value == pytest.approx(0.6)
        assert score.metadata["turn_latencies"] == [200.0]

    def test_negative_latency_skipped(self):
        messages = [
            Message(role="user", content="q"),
            Message(role="assistant", content="a", latency_ms=-100.0),
        ]
        ec = EvalCase(input="q", output="a", messages=messages)
        score = TurnLatencyMetric(max_ms_per_turn=500).measure(ec)
        assert score.value == 0.0
        assert score.metadata is None or score.metadata.get("n_turns_scored", 0) == 0


@pytest.mark.unit
class TestTurnTokenCostMetric:
    def test_within_budget(self):
        ec = EvalCase(input="q", output="a", messages=MESSAGES_WITH_TOKENS)
        score = TurnTokenCostMetric(max_tokens_per_turn=200, threshold=0.4).measure(ec)
        assert score.passed
        assert score.value == pytest.approx(0.5)  # mean of (1-50/200, 1-150/200) = mean(0.75, 0.25) = 0.5
        assert score.metadata["turn_token_counts"] == [50, 150]
        assert score.metadata["mean_token_count"] == pytest.approx(100.0)

    def test_over_budget(self):
        ec = EvalCase(input="q", output="a", messages=MESSAGES_WITH_TOKENS)
        score = TurnTokenCostMetric(max_tokens_per_turn=100, threshold=0.8).measure(ec)
        assert not score.passed
        # scores = [1-50/100, 0.0] = [0.5, 0.0], mean = 0.25
        assert score.value == pytest.approx(0.25)

    def test_no_messages(self):
        ec = EvalCase(input="q", output="a")
        score = TurnTokenCostMetric().measure(ec)
        assert score.value == 0.0
        assert "no messages" in score.reason

    def test_no_token_data(self):
        ec = EvalCase(input="q", output="a", messages=[
            Message(role="user", content="q"),
            Message(role="assistant", content="a"),  # no token_count
        ])
        score = TurnTokenCostMetric().measure(ec)
        assert score.value == 0.0
        assert "no token" in score.reason

    def test_partial_token_data_skips_missing(self):
        messages = [
            Message(role="user", content="q1"),
            Message(role="assistant", content="a1", token_count=500),
            Message(role="user", content="q2"),
            Message(role="assistant", content="a2"),  # missing token_count
        ]
        ec = EvalCase(input="q", output="a", messages=messages)
        score = TurnTokenCostMetric(max_tokens_per_turn=1000).measure(ec)
        # Only turn 1 scored: (1 - 500/1000) = 0.5
        assert score.value == pytest.approx(0.5)
        assert score.metadata["n_turns_scored"] == 1

    def test_negative_token_count_skipped(self):
        messages = [
            Message(role="user", content="q"),
            Message(role="assistant", content="a", token_count=-100),
        ]
        ec = EvalCase(input="q", output="a", messages=messages)
        score = TurnTokenCostMetric(max_tokens_per_turn=1000).measure(ec)
        assert score.value == 0.0
        assert score.metadata is None or score.metadata.get("n_turns_scored", 0) == 0
