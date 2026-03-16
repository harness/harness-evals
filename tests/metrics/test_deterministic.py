"""Tests for deterministic metrics."""

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.metrics.deterministic import (
    ContainsMetric,
    ExactMatchMetric,
    NumericDiffMetric,
    RegexMetric,
)


@pytest.mark.unit
class TestExactMatch:
    def test_match(self):
        ec = EvalCase(input="q", output="hello", expected="hello")
        assert ExactMatchMetric().measure(ec).passed

    def test_mismatch(self):
        ec = EvalCase(input="q", output="hello", expected="world")
        assert not ExactMatchMetric().measure(ec).passed

    def test_case_insensitive(self):
        ec = EvalCase(input="q", output="Hello", expected="hello")
        assert not ExactMatchMetric().measure(ec).passed
        assert ExactMatchMetric(case_sensitive=False).measure(ec).passed


@pytest.mark.unit
class TestContains:
    def test_contained(self):
        ec = EvalCase(input="q", output="hello world", expected="world")
        assert ContainsMetric().measure(ec).passed

    def test_not_contained(self):
        ec = EvalCase(input="q", output="hello", expected="world")
        assert not ContainsMetric().measure(ec).passed

    def test_case_insensitive(self):
        ec = EvalCase(input="q", output="Hello World", expected="hello")
        assert ContainsMetric(case_sensitive=False).measure(ec).passed


@pytest.mark.unit
class TestRegex:
    def test_match(self):
        ec = EvalCase(input="q", output="error code: 404", expected=r"error code: \d+")
        assert RegexMetric().measure(ec).passed

    def test_no_match(self):
        ec = EvalCase(input="q", output="success", expected=r"error code: \d+")
        assert not RegexMetric().measure(ec).passed

    def test_invalid_regex(self):
        ec = EvalCase(input="q", output="test", expected="[invalid")
        score = RegexMetric().measure(ec)
        assert not score.passed
        assert "Invalid regex" in score.reason


@pytest.mark.unit
class TestNumericDiff:
    def test_exact(self):
        ec = EvalCase(input="q", output="42", expected="42")
        assert NumericDiffMetric().measure(ec).value == 1.0

    def test_close(self):
        ec = EvalCase(input="q", output="41", expected="42")
        score = NumericDiffMetric(threshold=0.9).measure(ec)
        assert score.value > 0.95

    def test_far(self):
        ec = EvalCase(input="q", output="0", expected="100")
        score = NumericDiffMetric().measure(ec)
        assert score.value == 0.0

    def test_non_numeric(self):
        ec = EvalCase(input="q", output="abc", expected="42")
        score = NumericDiffMetric().measure(ec)
        assert not score.passed
        assert "Cannot parse" in score.reason
