"""Tests for core types, evaluate(), and assert_test()."""

import pytest

from harness_evals import Score, TestCase, assert_test, evaluate
from harness_evals.metrics.deterministic import ExactMatchMetric


@pytest.mark.unit
class TestTestCase:
    def test_minimal(self):
        tc = TestCase(input="hello", actual_output="world")
        assert tc.input == "hello"
        assert tc.expected_output is None
        assert tc.runs is None

    def test_full(self):
        tc = TestCase(
            input="q",
            actual_output="a",
            expected_output="a",
            context=["doc1"],
            metadata={"latency_ms": 100},
            tags={"env": "ci"},
            runs=[TestCase(input="q", actual_output="a")],
        )
        assert len(tc.runs) == 1
        assert tc.tags["env"] == "ci"


@pytest.mark.unit
class TestScore:
    def test_pass(self):
        s = Score(name="test", value=0.9, threshold=0.8, success=True)
        assert s.success

    def test_fail(self):
        s = Score(name="test", value=0.5, threshold=0.8, success=False)
        assert not s.success


@pytest.mark.unit
class TestEvaluate:
    def test_evaluate_returns_scores(self, simple_test_case):
        scores = evaluate(simple_test_case, metrics=[ExactMatchMetric()])
        assert len(scores) == 1
        assert scores[0].success
        assert scores[0].value == 1.0

    def test_evaluate_does_not_raise(self):
        tc = TestCase(input="x", actual_output="wrong", expected_output="right")
        scores = evaluate(tc, metrics=[ExactMatchMetric()])
        assert len(scores) == 1
        assert not scores[0].success

    def test_evaluate_catches_exceptions(self):
        """Metric that raises still returns a Score."""

        class BrokenMetric(ExactMatchMetric):
            def measure(self, test_case):
                raise RuntimeError("broken")

        tc = TestCase(input="x", actual_output="y", expected_output="y")
        scores = evaluate(tc, metrics=[BrokenMetric()])
        assert len(scores) == 1
        assert not scores[0].success
        assert "broken" in scores[0].reason


@pytest.mark.unit
class TestAssertTest:
    def test_assert_test_passes(self, simple_test_case):
        scores = assert_test(simple_test_case, metrics=[ExactMatchMetric()])
        assert all(s.success for s in scores)

    def test_assert_test_raises(self):
        tc = TestCase(input="x", actual_output="wrong", expected_output="right")
        with pytest.raises(AssertionError, match="1 metric"):
            assert_test(tc, metrics=[ExactMatchMetric()])
