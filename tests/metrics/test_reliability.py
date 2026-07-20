"""Tests for reliability metrics."""

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.metrics.reliability import (
    OutcomeConsistencyMetric,
    ResourceConsistencyMetric,
)


@pytest.mark.unit
class TestOutcomeConsistency:
    def test_all_same(self):
        runs = [EvalCase(input="q", output="same") for _ in range(5)]
        ec = EvalCase(input="q", output="same", runs=runs)
        score = OutcomeConsistencyMetric().measure(ec)
        assert score.value == 1.0
        assert score.passed
        assert "5 of 5" in score.reason

    def test_mixed(self, multi_run_eval_case):
        score = OutcomeConsistencyMetric(threshold=0.8).measure(multi_run_eval_case)
        assert score.value == 0.8
        assert score.passed

    def test_no_runs(self):
        ec = EvalCase(input="q", output="a")
        score = OutcomeConsistencyMetric().measure(ec)
        assert not score.passed
        assert "No runs" in score.reason


@pytest.mark.unit
class TestResourceConsistency:
    def test_consistent(self):
        runs = [
            EvalCase(input="q", output="a", token_count=100),
            EvalCase(input="q", output="a", token_count=102),
            EvalCase(input="q", output="a", token_count=98),
        ]
        ec = EvalCase(input="q", output="a", runs=runs)
        score = ResourceConsistencyMetric(threshold=0.9).measure(ec)
        assert score.passed
        assert score.value > 0.95
        assert "coefficient of variation" in score.reason

    def test_inconsistent(self):
        runs = [
            EvalCase(input="q", output="a", token_count=100),
            EvalCase(input="q", output="a", token_count=500),
            EvalCase(input="q", output="a", token_count=1000),
        ]
        ec = EvalCase(input="q", output="a", runs=runs)
        score = ResourceConsistencyMetric(threshold=0.9).measure(ec)
        assert not score.passed

    def test_missing_typed_field(self):
        runs = [
            EvalCase(input="q", output="a"),
            EvalCase(input="q", output="a"),
        ]
        ec = EvalCase(input="q", output="a", runs=runs)
        score = ResourceConsistencyMetric().measure(ec)
        assert not score.passed

    def test_custom_resource_key_via_metadata(self):
        """Custom keys like gpu_memory fall back to metadata."""
        runs = [
            EvalCase(input="q", output="a", metadata={"gpu_memory": 1024}),
            EvalCase(input="q", output="a", metadata={"gpu_memory": 1030}),
            EvalCase(input="q", output="a", metadata={"gpu_memory": 1020}),
        ]
        ec = EvalCase(input="q", output="a", runs=runs)
        score = ResourceConsistencyMetric(resource_key="gpu_memory", threshold=0.9).measure(ec)
        assert score.passed

    def test_latency_as_resource_key(self):
        """Typed fields work as resource_key too."""
        runs = [
            EvalCase(input="q", output="a", latency_ms=100),
            EvalCase(input="q", output="a", latency_ms=102),
            EvalCase(input="q", output="a", latency_ms=98),
        ]
        ec = EvalCase(input="q", output="a", runs=runs)
        score = ResourceConsistencyMetric(resource_key="latency_ms", threshold=0.9).measure(ec)
        assert score.passed
