"""Tests for operational metrics."""

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.metrics.operational import (
    CostEfficiencyMetric,
    LatencyMetric,
    RetryCountMetric,
    TokenCostMetric,
    TurnLatencyMetric,
    TurnTokenCostMetric,
)


@pytest.mark.unit
class TestConstructorValidation:
    def test_latency_non_positive_max_ms(self):
        with pytest.raises(ValueError):
            LatencyMetric(max_ms=0)

    def test_cost_efficiency_non_positive_max_cost(self):
        with pytest.raises(ValueError):
            CostEfficiencyMetric(max_cost_usd=0)

    def test_token_cost_non_positive_max_tokens(self):
        with pytest.raises(ValueError):
            TokenCostMetric(max_tokens=0)

    def test_retry_count_non_positive_max_retries(self):
        with pytest.raises(ValueError):
            RetryCountMetric(max_retries=0)

    def test_turn_latency_non_positive_max_ms_per_turn(self):
        with pytest.raises(ValueError):
            TurnLatencyMetric(max_ms_per_turn=0)

    def test_turn_token_cost_non_positive_max_tokens_per_turn(self):
        with pytest.raises(ValueError):
            TurnTokenCostMetric(max_tokens_per_turn=0)


@pytest.mark.unit
class TestLatency:
    def test_fast(self, operational_eval_case):
        score = LatencyMetric(max_ms=5000, threshold=0.5).measure(operational_eval_case)
        assert score.passed
        assert score.value == pytest.approx(0.76, abs=0.01)

    @pytest.mark.parametrize(
        "latency_ms, max_ms, expected_value",
        [
            (10000, 5000, 0.0),
            (0, 5000, 1.0),
            (2500, 5000, 0.5),
        ],
        ids=["over_max", "zero", "half"],
    )
    def test_latency_values(self, latency_ms, max_ms, expected_value):
        ec = EvalCase(input="q", output="a", latency_ms=latency_ms)
        score = LatencyMetric(max_ms=max_ms).measure(ec)
        assert score.value == pytest.approx(expected_value)

    def test_missing(self):
        ec = EvalCase(input="q", output="a")
        score = LatencyMetric().measure(ec)
        assert not score.passed
        assert "not provided" in score.reason


@pytest.mark.unit
class TestTokenCost:
    def test_low(self, operational_eval_case):
        score = TokenCostMetric(max_tokens=10000).measure(operational_eval_case)
        assert score.passed
        assert score.value > 0.9

    def test_missing(self):
        ec = EvalCase(input="q", output="a")
        assert not TokenCostMetric().measure(ec).passed


@pytest.mark.unit
class TestCostEfficiency:
    def test_cheap(self, operational_eval_case):
        score = CostEfficiencyMetric(max_cost_usd=0.10).measure(operational_eval_case)
        assert score.passed
        assert score.value > 0.9


@pytest.mark.unit
class TestRetryCount:
    @pytest.mark.parametrize(
        "retry_count, max_retries, expected_value",
        [
            (0, 5, 1.0),
            (3, 5, 0.4),
            (5, 5, 0.0),
        ],
        ids=["no_retries", "some_retries", "max_retries"],
    )
    def test_retry_values(self, retry_count, max_retries, expected_value):
        ec = EvalCase(input="q", output="a", retry_count=retry_count)
        score = RetryCountMetric(max_retries=max_retries).measure(ec)
        assert score.value == pytest.approx(expected_value)
