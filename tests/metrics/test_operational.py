"""Tests for operational metrics."""

import pytest

from harness_evals.core.test_case import TestCase
from harness_evals.metrics.operational import (
    CostEfficiencyMetric,
    LatencyMetric,
    RetryCountMetric,
    TokenCostMetric,
)


@pytest.mark.unit
class TestLatency:
    def test_fast(self, operational_test_case):
        score = LatencyMetric(max_ms=5000, threshold=0.5).measure(operational_test_case)
        assert score.success
        assert score.value == pytest.approx(0.76, abs=0.01)

    def test_slow(self):
        tc = TestCase(input="q", actual_output="a", metadata={"latency_ms": 10000})
        score = LatencyMetric(max_ms=5000).measure(tc)
        assert score.value == 0.0

    def test_missing(self):
        tc = TestCase(input="q", actual_output="a")
        score = LatencyMetric().measure(tc)
        assert not score.success
        assert "not provided" in score.reason


@pytest.mark.unit
class TestTokenCost:
    def test_low(self, operational_test_case):
        score = TokenCostMetric(max_tokens=10000).measure(operational_test_case)
        assert score.success
        assert score.value > 0.9

    def test_missing(self):
        tc = TestCase(input="q", actual_output="a")
        assert not TokenCostMetric().measure(tc).success


@pytest.mark.unit
class TestCostEfficiency:
    def test_cheap(self, operational_test_case):
        score = CostEfficiencyMetric(max_cost_usd=0.10).measure(operational_test_case)
        assert score.success
        assert score.value > 0.9


@pytest.mark.unit
class TestRetryCount:
    def test_no_retries(self, operational_test_case):
        score = RetryCountMetric().measure(operational_test_case)
        assert score.value == 1.0

    def test_some_retries(self):
        tc = TestCase(input="q", actual_output="a", metadata={"retry_count": 3})
        score = RetryCountMetric(max_retries=5).measure(tc)
        assert score.value == pytest.approx(0.4)
