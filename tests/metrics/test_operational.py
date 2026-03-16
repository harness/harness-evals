"""Tests for operational metrics."""

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.metrics.operational import (
    CostEfficiencyMetric,
    LatencyMetric,
    RetryCountMetric,
    TokenCostMetric,
)


@pytest.mark.unit
class TestLatency:
    def test_fast(self, operational_eval_case):
        score = LatencyMetric(max_ms=5000, threshold=0.5).measure(operational_eval_case)
        assert score.passed
        assert score.value == pytest.approx(0.76, abs=0.01)

    def test_slow(self):
        ec = EvalCase(input="q", output="a", latency_ms=10000)
        score = LatencyMetric(max_ms=5000).measure(ec)
        assert score.value == 0.0

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
    def test_no_retries(self, operational_eval_case):
        score = RetryCountMetric().measure(operational_eval_case)
        assert score.value == 1.0

    def test_some_retries(self):
        ec = EvalCase(input="q", output="a", retry_count=3)
        score = RetryCountMetric(max_retries=5).measure(ec)
        assert score.value == pytest.approx(0.4)
