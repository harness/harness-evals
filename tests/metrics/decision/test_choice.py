"""Tests for ChoiceMetric."""

from __future__ import annotations

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.decision.types import ChoiceAnswer
from harness_evals.metrics.decision.choice import ChoiceMetric
from tests.metrics.decision.conftest import FakeDecisionProvider

_CRITERIA = {"billing": None, "returns": None}


@pytest.mark.unit
class TestCorrectnessMode:
    async def test_exact_match_scores_one(self):
        provider = FakeDecisionProvider(answer=ChoiceAnswer(choice="billing", confidence=0.8, probabilities={}))
        metric = ChoiceMetric(provider=provider, instructions="Which team?", criteria=_CRITERIA)
        ec = EvalCase(input="x", output="I was charged twice", expected="billing")
        score = await metric.a_measure(ec)
        assert score.value == 1.0

    async def test_mismatch_scores_zero(self):
        provider = FakeDecisionProvider(answer=ChoiceAnswer(choice="billing", confidence=0.8, probabilities={}))
        metric = ChoiceMetric(provider=provider, instructions="Which team?", criteria=_CRITERIA)
        ec = EvalCase(input="x", output="Where is my package", expected="returns")
        score = await metric.a_measure(ec)
        assert score.value == 0.0

    async def test_mode_correctness_without_expected_raises(self):
        provider = FakeDecisionProvider(answer=ChoiceAnswer(choice="billing", confidence=0.8, probabilities={}))
        metric = ChoiceMetric(provider=provider, instructions="?", criteria=_CRITERIA, mode="correctness")
        ec = EvalCase(input="x", output="y")
        with pytest.raises(ValueError, match="mode='correctness'"):
            await metric.a_measure(ec)


@pytest.mark.unit
class TestConfidenceMode:
    async def test_value_is_confidence(self):
        provider = FakeDecisionProvider(
            answer=ChoiceAnswer(choice="billing", confidence=0.65, probabilities={"billing": 0.65, "returns": 0.35})
        )
        metric = ChoiceMetric(provider=provider, instructions="?", criteria=_CRITERIA)
        ec = EvalCase(input="x", output="y")
        score = await metric.a_measure(ec)
        assert score.value == 0.65
        assert score.metadata["choice"] == "billing"
        assert score.metadata["probabilities"] == {"billing": 0.65, "returns": 0.35}
        assert score.metadata["mode"] == "confidence"

    async def test_out_of_range_confidence_is_clamped(self):
        provider = FakeDecisionProvider(answer=ChoiceAnswer(choice="billing", confidence=1.4, probabilities={}))
        metric = ChoiceMetric(provider=provider, instructions="?", criteria=_CRITERIA)
        ec = EvalCase(input="x", output="y")
        score = await metric.a_measure(ec)
        assert score.value == 1.0


@pytest.mark.unit
class TestErrorsAndEdgeCases:
    async def test_missing_state_field_returns_zero(self):
        provider = FakeDecisionProvider(answer=ChoiceAnswer(choice="billing", confidence=0.8, probabilities={}))
        metric = ChoiceMetric(provider=provider, instructions="?", criteria=_CRITERIA, state_field="nonexistent")
        ec = EvalCase(input="x", output="y")
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert provider.calls == []

    async def test_provider_error_propagates(self):
        provider = FakeDecisionProvider(error=RuntimeError("upstream failure"))
        metric = ChoiceMetric(provider=provider, instructions="?", criteria=_CRITERIA)
        ec = EvalCase(input="x", output="y")
        with pytest.raises(RuntimeError, match="upstream failure"):
            await metric.a_measure(ec)

    def test_sync_measure_delegates_to_async(self):
        provider = FakeDecisionProvider(answer=ChoiceAnswer(choice="billing", confidence=0.8, probabilities={}))
        metric = ChoiceMetric(provider=provider, instructions="?", criteria=_CRITERIA)
        ec = EvalCase(input="x", output="y", expected="billing")
        score = metric.measure(ec)
        assert score.value == 1.0
