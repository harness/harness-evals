"""Tests for FaultRobustnessMetric."""

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.metrics.reliability.fault_robustness import FaultRobustnessMetric


@pytest.mark.unit
class TestFaultRobustness:
    def test_all_pass(self):
        ec = EvalCase(
            input="q",
            output="a",
            metadata={"nominal_passed": True, "perturbed_results": [True, True, True]},
        )
        score = FaultRobustnessMetric().measure(ec)
        assert score.value == 1.0
        assert score.passed

    def test_all_fail(self):
        ec = EvalCase(
            input="q",
            output="a",
            metadata={"nominal_passed": True, "perturbed_results": [False, False, False]},
        )
        score = FaultRobustnessMetric().measure(ec)
        assert score.value == 0.0
        assert not score.passed

    def test_nominal_failed(self):
        ec = EvalCase(
            input="q",
            output="a",
            metadata={"nominal_passed": False, "perturbed_results": [False, False]},
        )
        score = FaultRobustnessMetric().measure(ec)
        assert score.value == 1.0

    def test_missing_metadata(self):
        ec = EvalCase(input="q", output="a")
        score = FaultRobustnessMetric().measure(ec)
        assert score.value == 0.0

    def test_name_and_threshold(self):
        metric = FaultRobustnessMetric(threshold=0.5)
        assert metric.name == "fault_robustness"
        assert metric.threshold == 0.5

    def test_dataset_level(self):
        metric = FaultRobustnessMetric()
        score = metric.measure_robustness(
            nominal_passed=[True, True, True],
            perturbed_passed=[[True, True], [True, False], [False, False]],
        )
        assert 0.0 < score.value < 1.0
