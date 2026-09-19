"""Tests for ScoreMetric."""

from __future__ import annotations

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.decision.types import ScoreAnswer
from harness_evals.metrics.decision.score import ScoreMetric
from tests.metrics.decision.conftest import FakeDecisionProvider

_CRITERIA = ["low", "medium", "high"]


@pytest.mark.unit
class TestCorrectnessMode:
    async def test_exact_level_match_scores_one(self):
        provider = FakeDecisionProvider(answer=ScoreAnswer(score=2.0, confidence=0.9, legend={}, probabilities={}))
        metric = ScoreMetric(provider=provider, instructions="?", criteria=_CRITERIA)
        ec = EvalCase(input="x", output="y", expected=2)
        score = await metric.a_measure(ec)
        assert score.value == 1.0

    async def test_non_integer_raw_score_gives_partial_credit(self):
        provider = FakeDecisionProvider(answer=ScoreAnswer(score=1.3, confidence=0.7, legend={}, probabilities={}))
        metric = ScoreMetric(provider=provider, instructions="?", criteria=_CRITERIA)
        ec = EvalCase(input="x", output="y", expected=1)
        score = await metric.a_measure(ec)
        assert score.value == pytest.approx(0.85)

    async def test_max_distance_scores_zero(self):
        provider = FakeDecisionProvider(answer=ScoreAnswer(score=0.0, confidence=0.9, legend={}, probabilities={}))
        metric = ScoreMetric(provider=provider, instructions="?", criteria=_CRITERIA)
        ec = EvalCase(input="x", output="y", expected=2)
        score = await metric.a_measure(ec)
        assert score.value == 0.0

    async def test_non_numeric_expected_raises(self):
        provider = FakeDecisionProvider(answer=ScoreAnswer(score=1.0, confidence=0.9, legend={}, probabilities={}))
        metric = ScoreMetric(provider=provider, instructions="?", criteria=_CRITERIA)
        ec = EvalCase(input="x", output="y", expected="high")
        with pytest.raises(ValueError, match="numeric rubric level"):
            await metric.a_measure(ec)

    async def test_out_of_range_expected_raises(self):
        provider = FakeDecisionProvider(answer=ScoreAnswer(score=1.0, confidence=0.9, legend={}, probabilities={}))
        metric = ScoreMetric(provider=provider, instructions="?", criteria=_CRITERIA)
        ec = EvalCase(input="x", output="y", expected=5)
        with pytest.raises(ValueError, match="out of range"):
            await metric.a_measure(ec)

    async def test_mode_correctness_without_expected_raises(self):
        provider = FakeDecisionProvider(answer=ScoreAnswer(score=1.0, confidence=0.9, legend={}, probabilities={}))
        metric = ScoreMetric(provider=provider, instructions="?", criteria=_CRITERIA, mode="correctness")
        ec = EvalCase(input="x", output="y")
        with pytest.raises(ValueError, match="mode='correctness'"):
            await metric.a_measure(ec)


@pytest.mark.unit
class TestConfidenceMode:
    async def test_value_is_confidence(self):
        provider = FakeDecisionProvider(
            answer=ScoreAnswer(
                score=1.3, confidence=0.7, legend={0: "low", 1: "medium", 2: "high"}, probabilities={1: 0.7}
            )
        )
        metric = ScoreMetric(provider=provider, instructions="?", criteria=_CRITERIA)
        ec = EvalCase(input="x", output="y")
        score = await metric.a_measure(ec)
        assert score.value == 0.7
        assert score.metadata["score"] == 1.3
        assert score.metadata["legend"] == {0: "low", 1: "medium", 2: "high"}
        assert score.metadata["mode"] == "confidence"

    async def test_out_of_range_confidence_is_clamped(self):
        provider = FakeDecisionProvider(answer=ScoreAnswer(score=1.0, confidence=-0.2, legend={}, probabilities={}))
        metric = ScoreMetric(provider=provider, instructions="?", criteria=_CRITERIA)
        ec = EvalCase(input="x", output="y")
        score = await metric.a_measure(ec)
        assert score.value == 0.0


@pytest.mark.unit
class TestErrorsAndEdgeCases:
    async def test_missing_state_field_returns_zero(self):
        provider = FakeDecisionProvider(answer=ScoreAnswer(score=1.0, confidence=0.9, legend={}, probabilities={}))
        metric = ScoreMetric(provider=provider, instructions="?", criteria=_CRITERIA, state_field="nonexistent")
        ec = EvalCase(input="x", output="y")
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert provider.calls == []

    async def test_provider_error_propagates(self):
        provider = FakeDecisionProvider(error=RuntimeError("upstream failure"))
        metric = ScoreMetric(provider=provider, instructions="?", criteria=_CRITERIA)
        ec = EvalCase(input="x", output="y")
        with pytest.raises(RuntimeError, match="upstream failure"):
            await metric.a_measure(ec)

    def test_sync_measure_delegates_to_async(self):
        provider = FakeDecisionProvider(answer=ScoreAnswer(score=2.0, confidence=0.9, legend={}, probabilities={}))
        metric = ScoreMetric(provider=provider, instructions="?", criteria=_CRITERIA)
        ec = EvalCase(input="x", output="y", expected=2)
        score = metric.measure(ec)
        assert score.value == 1.0
