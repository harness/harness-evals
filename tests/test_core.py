"""Tests for core types, evaluate(), assert_test(), evaluate_cases(), and evaluate_dataset()."""

import pytest

from harness_evals import EvalCase, Golden, Score, assert_test, evaluate, evaluate_cases, evaluate_dataset
from harness_evals.metrics.deterministic import ExactMatchMetric


@pytest.mark.unit
class TestGolden:
    def test_minimal(self):
        g = Golden(input="hello")
        assert g.input == "hello"
        assert g.expected is None
        assert g.context is None

    def test_full(self):
        g = Golden(
            input="q",
            expected="a",
            context=["doc1"],
            metadata={"source": "test"},
            tags={"env": "ci"},
        )
        assert g.expected == "a"
        assert g.tags["env"] == "ci"

    def test_to_dict_omits_none(self):
        g = Golden(input="q", expected="a")
        d = g.to_dict()
        assert d == {"input": "q", "expected": "a"}
        assert "context" not in d

    def test_from_dict(self):
        g = Golden.from_dict({"input": "q", "expected": "a", "context": ["c1"]})
        assert g.input == "q"
        assert g.expected == "a"
        assert g.context == ["c1"]

    def test_from_dict_backward_compat(self):
        g = Golden.from_dict({"input": "q", "expected_output": "a"})
        assert g.expected == "a"

    def test_from_dict_ignores_unknown_keys(self):
        g = Golden.from_dict({"input": "q", "unknown_field": 42})
        assert g.input == "q"


@pytest.mark.unit
class TestEvalCase:
    def test_minimal(self):
        ec = EvalCase(input="hello", output="world")
        assert ec.input == "hello"
        assert ec.expected is None
        assert ec.runs is None
        assert ec.latency_ms is None

    def test_typed_fields(self):
        ec = EvalCase(
            input="q",
            output="a",
            latency_ms=320.5,
            token_count=150,
            cost_usd=0.003,
            retry_count=1,
            confidence=0.95,
        )
        assert ec.latency_ms == 320.5
        assert ec.token_count == 150
        assert ec.cost_usd == 0.003
        assert ec.retry_count == 1
        assert ec.confidence == 0.95

    def test_from_dict(self):
        ec = EvalCase.from_dict({"input": "q", "output": "a", "expected": "a", "latency_ms": 100})
        assert ec.output == "a"
        assert ec.latency_ms == 100

    def test_from_dict_backward_compat(self):
        ec = EvalCase.from_dict(
            {
                "input": "q",
                "actual_output": "a",
                "expected_output": "e",
                "token_usage": 500,
            }
        )
        assert ec.output == "a"
        assert ec.expected == "e"
        assert ec.token_count == 500

    def test_from_dict_ignores_unknown_keys(self):
        ec = EvalCase.from_dict({"input": "q", "output": "a", "unknown": 42})
        assert ec.output == "a"

    def test_from_golden(self):
        g = Golden(input="q", expected="a", context=["c1"], tags={"env": "ci"})
        ec = EvalCase.from_golden(g, output="a", latency_ms=200)
        assert ec.input == "q"
        assert ec.output == "a"
        assert ec.expected == "a"
        assert ec.context == ["c1"]
        assert ec.tags == {"env": "ci"}
        assert ec.latency_ms == 200

    def test_to_dict_omits_none(self):
        ec = EvalCase(input="q", output="a")
        d = ec.to_dict()
        assert d == {"input": "q", "output": "a"}
        assert "latency_ms" not in d


@pytest.mark.unit
class TestScore:
    def test_pass_auto_computed(self):
        s = Score(name="test", value=0.9, threshold=0.8)
        assert s.passed is True

    def test_fail_auto_computed(self):
        s = Score(name="test", value=0.5, threshold=0.8)
        assert s.passed is False

    def test_exact_threshold(self):
        s = Score(name="test", value=0.8, threshold=0.8)
        assert s.passed is True

    def test_created_at_set(self):
        s = Score(name="test", value=1.0, threshold=0.5)
        assert s.created_at is not None

    def test_to_dict(self):
        s = Score(name="test", value=0.9, threshold=0.8, reason="good")
        d = s.to_dict()
        assert d["name"] == "test"
        assert d["passed"] is True
        assert d["reason"] == "good"
        assert "created_at" in d


@pytest.mark.unit
class TestEvaluate:
    def test_evaluate_returns_scores(self, simple_eval_case):
        scores = evaluate(simple_eval_case, metrics=[ExactMatchMetric()])
        assert len(scores) == 1
        assert scores[0].passed
        assert scores[0].value == 1.0

    def test_evaluate_does_not_raise(self):
        ec = EvalCase(input="x", output="wrong", expected="right")
        scores = evaluate(ec, metrics=[ExactMatchMetric()])
        assert len(scores) == 1
        assert not scores[0].passed

    def test_evaluate_catches_exceptions(self):
        class BrokenMetric(ExactMatchMetric):
            def measure(self, eval_case):
                raise RuntimeError("broken")

        ec = EvalCase(input="x", output="y", expected="y")
        scores = evaluate(ec, metrics=[BrokenMetric()])
        assert len(scores) == 1
        assert not scores[0].passed
        assert "broken" in scores[0].reason


@pytest.mark.unit
class TestAssertTest:
    def test_assert_test_passes(self, simple_eval_case):
        scores = assert_test(simple_eval_case, metrics=[ExactMatchMetric()])
        assert all(s.passed for s in scores)

    def test_assert_test_raises(self):
        ec = EvalCase(input="x", output="wrong", expected="right")
        with pytest.raises(AssertionError, match="1 metric"):
            assert_test(ec, metrics=[ExactMatchMetric()])


@pytest.mark.unit
class TestEvaluateCases:
    def test_batch(self):
        cases = [
            EvalCase(input="q1", output="a", expected="a"),
            EvalCase(input="q2", output="b", expected="c"),
        ]
        results = evaluate_cases(cases, metrics=[ExactMatchMetric()])
        assert len(results) == 2
        assert results[0][0].passed
        assert not results[1][0].passed


@pytest.mark.unit
class TestEvaluateDataset:
    @pytest.mark.asyncio
    async def test_evaluate_dataset(self):
        goldens = [
            Golden(input="q1", expected="a"),
            Golden(input="q2", expected="b"),
        ]

        async def agent_fn(golden: Golden) -> EvalCase:
            return EvalCase.from_golden(golden, output=golden.expected)

        results = await evaluate_dataset(goldens, agent_fn, metrics=[ExactMatchMetric()])
        assert len(results) == 2
        assert all(r[0].passed for r in results)

    @pytest.mark.asyncio
    async def test_evaluate_dataset_with_failure(self):
        goldens = [Golden(input="q", expected="right")]

        async def bad_agent(golden: Golden) -> EvalCase:
            return EvalCase.from_golden(golden, output="wrong")

        results = await evaluate_dataset(goldens, bad_agent, metrics=[ExactMatchMetric()])
        assert not results[0][0].passed
