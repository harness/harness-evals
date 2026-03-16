"""Tests for reliability metrics."""

import pytest

from harness_evals.core.test_case import TestCase
from harness_evals.metrics.reliability import (
    OutcomeConsistencyMetric,
    ResourceConsistencyMetric,
)


@pytest.mark.unit
class TestOutcomeConsistency:
    def test_all_same(self):
        runs = [TestCase(input="q", actual_output="same") for _ in range(5)]
        tc = TestCase(input="q", actual_output="same", runs=runs)
        score = OutcomeConsistencyMetric().measure(tc)
        assert score.value == 1.0
        assert score.success

    def test_mixed(self, multi_run_test_case):
        score = OutcomeConsistencyMetric(threshold=0.8).measure(multi_run_test_case)
        assert score.value == 0.8  # 4 out of 5 are "result_a"
        assert score.success

    def test_no_runs(self):
        tc = TestCase(input="q", actual_output="a")
        score = OutcomeConsistencyMetric().measure(tc)
        assert not score.success
        assert "No runs" in score.reason


@pytest.mark.unit
class TestResourceConsistency:
    def test_consistent(self):
        runs = [
            TestCase(input="q", actual_output="a", metadata={"token_usage": 100}),
            TestCase(input="q", actual_output="a", metadata={"token_usage": 102}),
            TestCase(input="q", actual_output="a", metadata={"token_usage": 98}),
        ]
        tc = TestCase(input="q", actual_output="a", runs=runs)
        score = ResourceConsistencyMetric(threshold=0.9).measure(tc)
        assert score.success
        assert score.value > 0.95

    def test_inconsistent(self):
        runs = [
            TestCase(input="q", actual_output="a", metadata={"token_usage": 100}),
            TestCase(input="q", actual_output="a", metadata={"token_usage": 500}),
            TestCase(input="q", actual_output="a", metadata={"token_usage": 1000}),
        ]
        tc = TestCase(input="q", actual_output="a", runs=runs)
        score = ResourceConsistencyMetric(threshold=0.9).measure(tc)
        assert not score.success

    def test_missing_metadata(self):
        runs = [
            TestCase(input="q", actual_output="a"),
            TestCase(input="q", actual_output="a"),
        ]
        tc = TestCase(input="q", actual_output="a", runs=runs)
        score = ResourceConsistencyMetric().measure(tc)
        assert not score.success
