"""Tests for NoulMetric."""

from __future__ import annotations

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.decision.types import NoulAnswer
from harness_evals.metrics.decision.noul import NoulMetric
from tests.metrics.decision.conftest import FakeDecisionProvider


@pytest.mark.unit
class TestCorrectnessMode:
    async def test_matches_expected_true(self):
        provider = FakeDecisionProvider(answer=NoulAnswer(noul=0.9))
        metric = NoulMetric(provider=provider, instructions="Is this an escalation?")
        ec = EvalCase(input="x", output="I want a human", expected=True)
        score = await metric.a_measure(ec)
        assert score.value == 1.0

    async def test_mismatch_expected_false(self):
        provider = FakeDecisionProvider(answer=NoulAnswer(noul=0.9))
        metric = NoulMetric(provider=provider, instructions="Is this an escalation?")
        ec = EvalCase(input="x", output="ok", expected=False)
        score = await metric.a_measure(ec)
        assert score.value == 0.0

    @pytest.mark.parametrize("expected", [True, False, 1, 0, "true", "FALSE"])
    async def test_accepted_expected_values(self, expected):
        provider = FakeDecisionProvider(answer=NoulAnswer(noul=0.6))
        metric = NoulMetric(provider=provider, instructions="?")
        ec = EvalCase(input="x", output="y", expected=expected)
        score = await metric.a_measure(ec)
        assert score.value in (0.0, 1.0)

    @pytest.mark.parametrize("expected", ["yes", 2, "maybe"])
    async def test_rejected_expected_values_raise(self, expected):
        provider = FakeDecisionProvider(answer=NoulAnswer(noul=0.6))
        metric = NoulMetric(provider=provider, instructions="?")
        ec = EvalCase(input="x", output="y", expected=expected)
        with pytest.raises(ValueError, match="Noul expected"):
            await metric.a_measure(ec)

    async def test_mode_correctness_without_expected_raises(self):
        provider = FakeDecisionProvider(answer=NoulAnswer(noul=0.6))
        metric = NoulMetric(provider=provider, instructions="?", mode="correctness")
        ec = EvalCase(input="x", output="y")
        with pytest.raises(ValueError, match="mode='correctness'"):
            await metric.a_measure(ec)


@pytest.mark.unit
class TestConfidenceMode:
    async def test_value_is_raw_noul(self):
        provider = FakeDecisionProvider(answer=NoulAnswer(noul=0.73))
        metric = NoulMetric(provider=provider, instructions="?")
        ec = EvalCase(input="x", output="y")
        score = await metric.a_measure(ec)
        assert score.value == 0.73
        assert score.metadata["mode"] == "confidence"
        assert score.metadata["model"] == "fake-model"
        assert score.metadata["input_tokens"] == 11
        assert score.metadata["output_tokens"] == 3

    async def test_invert_flips_value(self):
        provider = FakeDecisionProvider(answer=NoulAnswer(noul=0.73))
        metric = NoulMetric(provider=provider, instructions="?", invert=True)
        ec = EvalCase(input="x", output="y")
        score = await metric.a_measure(ec)
        assert score.value == pytest.approx(0.27)

    async def test_out_of_range_provider_response_is_clamped(self):
        provider = FakeDecisionProvider(answer=NoulAnswer(noul=1.5))
        metric = NoulMetric(provider=provider, instructions="?")
        ec = EvalCase(input="x", output="y")
        score = await metric.a_measure(ec)
        assert score.value == 1.0


@pytest.mark.unit
class TestErrorsAndEdgeCases:
    async def test_missing_state_field_returns_zero_not_exception(self):
        provider = FakeDecisionProvider(answer=NoulAnswer(noul=0.9))
        metric = NoulMetric(provider=provider, instructions="?", state_field="nonexistent")
        ec = EvalCase(input="x", output="y")
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "nonexistent" in score.reason
        assert provider.calls == []

    async def test_provider_error_propagates(self):
        provider = FakeDecisionProvider(error=RuntimeError("upstream failure"))
        metric = NoulMetric(provider=provider, instructions="?")
        ec = EvalCase(input="x", output="y")
        with pytest.raises(RuntimeError, match="upstream failure"):
            await metric.a_measure(ec)

    def test_sync_measure_delegates_to_async(self):
        provider = FakeDecisionProvider(answer=NoulAnswer(noul=0.42))
        metric = NoulMetric(provider=provider, instructions="?")
        ec = EvalCase(input="x", output="y")
        score = metric.measure(ec)
        assert score.value == 0.42
