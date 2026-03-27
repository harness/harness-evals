"""Tests for ListContainsMetric."""

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.metrics.deterministic.list_contains import ListContainsMetric


@pytest.mark.unit
class TestListContainsMetric:
    def test_subset_all_present(self):
        ec = EvalCase(input="q", output=["a", "b", "c"], expected=["a", "b"])
        score = ListContainsMetric().measure(ec)
        assert score.value == 1.0
        assert score.passed

    def test_subset_partial(self):
        ec = EvalCase(input="q", output=["a", "c"], expected=["a", "b"])
        score = ListContainsMetric().measure(ec)
        assert score.value == 0.5

    def test_subset_none_present(self):
        ec = EvalCase(input="q", output=["x", "y"], expected=["a", "b"])
        score = ListContainsMetric().measure(ec)
        assert score.value == 0.0

    def test_exact_match(self):
        ec = EvalCase(input="q", output=["a", "b"], expected=["a", "b"])
        score = ListContainsMetric(mode="exact").measure(ec)
        assert score.value == 1.0

    def test_exact_mismatch(self):
        ec = EvalCase(input="q", output=["a", "b", "c"], expected=["a", "b"])
        score = ListContainsMetric(mode="exact").measure(ec)
        assert 0.0 < score.value < 1.0

    def test_case_insensitive(self):
        ec = EvalCase(input="q", output=["Apple", "Banana"], expected=["apple", "banana"])
        score = ListContainsMetric(case_sensitive=False).measure(ec)
        assert score.value == 1.0

    def test_case_sensitive_mismatch(self):
        ec = EvalCase(input="q", output=["Apple"], expected=["apple"])
        score = ListContainsMetric(case_sensitive=True).measure(ec)
        assert score.value == 0.0

    def test_json_string_input(self):
        ec = EvalCase(input="q", output='["a", "b", "c"]', expected='["a", "b"]')
        score = ListContainsMetric().measure(ec)
        assert score.value == 1.0

    def test_comma_separated_input(self):
        ec = EvalCase(input="q", output="a, b, c", expected="a, b")
        score = ListContainsMetric().measure(ec)
        assert score.value == 1.0

    def test_expected_none(self):
        ec = EvalCase(input="q", output=["a", "b"])
        score = ListContainsMetric().measure(ec)
        assert score.value == 0.0

    def test_empty_expected(self):
        ec = EvalCase(input="q", output=["a"], expected=[])
        score = ListContainsMetric().measure(ec)
        assert score.value == 1.0
