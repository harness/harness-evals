"""Tests for predictability metrics (Calibration, Discrimination)."""

import pytest

from harness_evals import EvalCase
from harness_evals.metrics.reliability.calibration import CalibrationMetric
from harness_evals.metrics.reliability.discrimination import DiscriminationMetric


@pytest.mark.unit
class TestCalibrationMetric:
    def test_single_case_returns_zero(self):
        metric = CalibrationMetric()
        ec = EvalCase(input="q", output="a", confidence=0.9)
        score = metric.measure(ec)
        assert score.value == 0.0
        assert "measure_dataset" in score.reason

    def test_perfect_calibration(self):
        """When confidence exactly matches accuracy in each bin."""
        metric = CalibrationMetric(n_bins=2, threshold=0.5)
        # Low confidence bin (0.0-0.5): all fail -> conf~0.3, acc=0.0
        # High confidence bin (0.5-1.0): all pass -> conf~0.9, acc=1.0
        cases = [
            EvalCase(input="q", output="a", confidence=0.0),
            EvalCase(input="q", output="a", confidence=0.0),
            EvalCase(input="q", output="a", confidence=1.0),
            EvalCase(input="q", output="a", confidence=1.0),
        ]
        outcomes = [False, False, True, True]
        score = metric.measure_dataset(cases, outcomes)
        # ECE should be 0.0 for perfect calibration
        assert score.value == 1.0

    def test_poor_calibration(self):
        """High confidence but all failures -> high ECE."""
        metric = CalibrationMetric(n_bins=5, threshold=0.5)
        cases = [EvalCase(input="q", output="a", confidence=0.95) for _ in range(10)]
        outcomes = [False] * 10
        score = metric.measure_dataset(cases, outcomes)
        # ECE should be ~0.95, so value ~0.05
        assert score.value < 0.2

    def test_mismatched_lengths(self):
        metric = CalibrationMetric()
        cases = [EvalCase(input="q", output="a", confidence=0.5)]
        outcomes = [True, False]
        score = metric.measure_dataset(cases, outcomes)
        assert score.value == 0.0

    def test_insufficient_confidence(self):
        metric = CalibrationMetric()
        cases = [EvalCase(input="q", output="a")]  # no confidence
        outcomes = [True]
        score = metric.measure_dataset(cases, outcomes)
        assert score.value == 0.0
        assert "at least 2" in score.reason

    def test_metadata_contains_ece(self):
        metric = CalibrationMetric(n_bins=5)
        cases = [
            EvalCase(input="q", output="a", confidence=0.8),
            EvalCase(input="q", output="a", confidence=0.2),
        ]
        outcomes = [True, False]
        score = metric.measure_dataset(cases, outcomes)
        assert "ece" in score.metadata


@pytest.mark.unit
class TestDiscriminationMetric:
    def test_single_case_returns_zero(self):
        metric = DiscriminationMetric()
        ec = EvalCase(input="q", output="a", confidence=0.9)
        score = metric.measure(ec)
        assert score.value == 0.0

    def test_perfect_discrimination(self):
        """All successes have higher confidence than all failures."""
        metric = DiscriminationMetric(threshold=0.5)
        cases = [
            EvalCase(input="q", output="a", confidence=0.9),
            EvalCase(input="q", output="a", confidence=0.8),
            EvalCase(input="q", output="a", confidence=0.3),
            EvalCase(input="q", output="a", confidence=0.1),
        ]
        outcomes = [True, True, False, False]
        score = metric.measure_dataset(cases, outcomes)
        assert score.value == 1.0

    def test_no_discrimination(self):
        """Confidence is random relative to outcome -> AUC ~0.5."""
        metric = DiscriminationMetric(threshold=0.9)
        cases = [
            EvalCase(input="q", output="a", confidence=0.9),
            EvalCase(input="q", output="a", confidence=0.1),
            EvalCase(input="q", output="a", confidence=0.8),
            EvalCase(input="q", output="a", confidence=0.2),
        ]
        # Alternate: high conf fail, low conf pass
        outcomes = [False, True, False, True]
        score = metric.measure_dataset(cases, outcomes)
        # AUC should be close to 0.0 (inverted)
        assert score.value < 0.5

    def test_all_same_outcome(self):
        metric = DiscriminationMetric()
        cases = [
            EvalCase(input="q", output="a", confidence=0.9),
            EvalCase(input="q", output="a", confidence=0.5),
        ]
        outcomes = [True, True]
        score = metric.measure_dataset(cases, outcomes)
        assert score.value == 0.0
        assert "both successes and failures" in score.reason

    def test_mismatched_lengths(self):
        metric = DiscriminationMetric()
        cases = [EvalCase(input="q", output="a", confidence=0.5)]
        outcomes = [True, False]
        score = metric.measure_dataset(cases, outcomes)
        assert score.value == 0.0

    def test_metadata_contains_counts(self):
        metric = DiscriminationMetric()
        cases = [
            EvalCase(input="q", output="a", confidence=0.9),
            EvalCase(input="q", output="a", confidence=0.1),
        ]
        outcomes = [True, False]
        score = metric.measure_dataset(cases, outcomes)
        assert score.metadata["n_positive"] == 1
        assert score.metadata["n_negative"] == 1
