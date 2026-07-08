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
    @pytest.mark.parametrize(
        "output, expected, case_sensitive, should_pass",
        [
            ("hello", "hello", True, True),
            ("hello", "world", True, False),
            ("Hello", "hello", True, False),
            ("Hello", "hello", False, True),
        ],
        ids=["match", "mismatch", "case_sensitive_fail", "case_insensitive_pass"],
    )
    def test_exact_match(self, output, expected, case_sensitive, should_pass):
        ec = EvalCase(input="q", output=output, expected=expected)
        score = ExactMatchMetric(case_sensitive=case_sensitive).measure(ec)
        assert score.passed == should_pass
        assert score.reason


@pytest.mark.unit
class TestContains:
    @pytest.mark.parametrize(
        "output, expected, case_sensitive, should_pass",
        [
            ("hello world", "world", True, True),
            ("hello", "world", True, False),
            ("Hello World", "hello", False, True),
        ],
        ids=["contained", "not_contained", "case_insensitive"],
    )
    def test_contains(self, output, expected, case_sensitive, should_pass):
        ec = EvalCase(input="q", output=output, expected=expected)
        score = ContainsMetric(case_sensitive=case_sensitive).measure(ec)
        assert score.passed == should_pass
        assert score.reason


@pytest.mark.unit
class TestRegex:
    @pytest.mark.parametrize(
        "output, pattern, should_pass",
        [
            ("error code: 404", r"error code: \d+", True),
            ("success", r"error code: \d+", False),
        ],
        ids=["match", "no_match"],
    )
    def test_regex(self, output, pattern, should_pass):
        ec = EvalCase(input="q", output=output, expected=pattern)
        score = RegexMetric().measure(ec)
        assert score.passed == should_pass
        assert score.reason

    def test_invalid_regex(self):
        ec = EvalCase(input="q", output="test", expected="[invalid")
        score = RegexMetric().measure(ec)
        assert not score.passed
        assert "Invalid regex" in score.reason


@pytest.mark.unit
class TestNumericDiff:
    @pytest.mark.parametrize(
        "output, expected, threshold, check",
        [
            ("42", "42", 1.0, lambda s: s.value == 1.0),
            ("41", "42", 0.9, lambda s: s.value > 0.95),
            ("0", "100", 1.0, lambda s: s.value == 0.0),
        ],
        ids=["exact", "close", "far"],
    )
    def test_numeric_diff(self, output, expected, threshold, check):
        ec = EvalCase(input="q", output=output, expected=expected)
        score = NumericDiffMetric(threshold=threshold).measure(ec)
        assert check(score)
        assert score.reason

    def test_non_numeric(self):
        ec = EvalCase(input="q", output="abc", expected="42")
        score = NumericDiffMetric().measure(ec)
        assert not score.passed
        assert "Cannot parse" in score.reason
