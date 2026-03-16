"""Tests for deterministic metrics."""

import pytest

from harness_evals.core.test_case import TestCase
from harness_evals.metrics.deterministic import (
    ContainsMetric,
    ExactMatchMetric,
    NumericDiffMetric,
    RegexMetric,
)


@pytest.mark.unit
class TestExactMatch:
    def test_match(self):
        tc = TestCase(input="q", actual_output="hello", expected_output="hello")
        assert ExactMatchMetric().measure(tc).success

    def test_mismatch(self):
        tc = TestCase(input="q", actual_output="hello", expected_output="world")
        assert not ExactMatchMetric().measure(tc).success

    def test_case_insensitive(self):
        tc = TestCase(input="q", actual_output="Hello", expected_output="hello")
        assert not ExactMatchMetric().measure(tc).success
        assert ExactMatchMetric(case_sensitive=False).measure(tc).success


@pytest.mark.unit
class TestContains:
    def test_contained(self):
        tc = TestCase(input="q", actual_output="hello world", expected_output="world")
        assert ContainsMetric().measure(tc).success

    def test_not_contained(self):
        tc = TestCase(input="q", actual_output="hello", expected_output="world")
        assert not ContainsMetric().measure(tc).success

    def test_case_insensitive(self):
        tc = TestCase(input="q", actual_output="Hello World", expected_output="hello")
        assert ContainsMetric(case_sensitive=False).measure(tc).success


@pytest.mark.unit
class TestRegex:
    def test_match(self):
        tc = TestCase(input="q", actual_output="error code: 404", expected_output=r"error code: \d+")
        assert RegexMetric().measure(tc).success

    def test_no_match(self):
        tc = TestCase(input="q", actual_output="success", expected_output=r"error code: \d+")
        assert not RegexMetric().measure(tc).success

    def test_invalid_regex(self):
        tc = TestCase(input="q", actual_output="test", expected_output="[invalid")
        score = RegexMetric().measure(tc)
        assert not score.success
        assert "Invalid regex" in score.reason


@pytest.mark.unit
class TestNumericDiff:
    def test_exact(self):
        tc = TestCase(input="q", actual_output="42", expected_output="42")
        assert NumericDiffMetric().measure(tc).value == 1.0

    def test_close(self):
        tc = TestCase(input="q", actual_output="41", expected_output="42")
        score = NumericDiffMetric(threshold=0.9).measure(tc)
        assert score.value > 0.95

    def test_far(self):
        tc = TestCase(input="q", actual_output="0", expected_output="100")
        score = NumericDiffMetric().measure(tc)
        assert score.value == 0.0

    def test_non_numeric(self):
        tc = TestCase(input="q", actual_output="abc", expected_output="42")
        score = NumericDiffMetric().measure(tc)
        assert not score.success
        assert "Cannot parse" in score.reason
