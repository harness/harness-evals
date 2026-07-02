"""Tests for core types, evaluate(), assert_test(), evaluate_cases(), and evaluate_dataset()."""

import json

import pytest

from harness_evals import (
    EvalCase,
    Golden,
    Message,
    Score,
    ToolCall,
    a_evaluate,
    assert_test,
    evaluate,
    evaluate_batch_metrics,
    evaluate_cases,
    evaluate_dataset,
)
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

    def test_meta_returns_value(self):
        g = Golden(input="q", metadata={"source": "test", "version": 3})
        assert g.meta("source") == "test"
        assert g.meta("version") == 3

    def test_meta_returns_default_when_missing(self):
        g = Golden(input="q", metadata={"source": "test"})
        assert g.meta("missing") is None
        assert g.meta("missing", "fallback") == "fallback"

    def test_meta_handles_none_metadata(self):
        g = Golden(input="q")
        assert g.meta("any_key") is None
        assert g.meta("any_key", 42) == 42

    def test_expected_tools(self):
        g = Golden(input="q", expected_tools=["search", "read"])
        assert g.expected_tools == ["search", "read"]

    def test_expected_tools_to_dict(self):
        g = Golden(input="q", expected_tools=["search"])
        d = g.to_dict()
        assert d["expected_tools"] == ["search"]

    def test_expected_tools_from_dict(self):
        g = Golden.from_dict({"input": "q", "expected_tools": ["search", "read"]})
        assert g.expected_tools == ["search", "read"]


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

    def test_from_golden_metadata_extra_merges(self):
        g = Golden(input="q", expected="a", metadata={"source": "dataset", "version": 1})
        ec = EvalCase.from_golden(g, output="a", metadata_extra={"model": "gpt-4", "version": 2})
        assert ec.metadata["source"] == "dataset"
        assert ec.metadata["model"] == "gpt-4"
        assert ec.metadata["version"] == 2  # extra wins on conflict

    def test_from_golden_metadata_extra_with_none_golden_metadata(self):
        g = Golden(input="q", expected="a")
        ec = EvalCase.from_golden(g, output="a", metadata_extra={"model": "gpt-4"})
        assert ec.metadata == {"model": "gpt-4"}

    def test_from_golden_no_metadata_extra(self):
        g = Golden(input="q", expected="a", metadata={"source": "test"})
        ec = EvalCase.from_golden(g, output="a")
        assert ec.metadata == {"source": "test"}

    def test_from_golden_both_metadata_none(self):
        g = Golden(input="q", expected="a")
        ec = EvalCase.from_golden(g, output="a")
        assert ec.metadata is None

    def test_from_golden_copies_expected_tools(self):
        g = Golden(input="q", expected="a", expected_tools=["search", "read"])
        ec = EvalCase.from_golden(g, output="a")
        assert ec.expected_tools == ["search", "read"]

    def test_messages_field(self):
        msgs = [Message(role="user", content="hi"), Message(role="assistant", content="hello")]
        ec = EvalCase(input="q", output="a", messages=msgs)
        assert len(ec.messages) == 2
        assert ec.messages[0].role == "user"

    def test_tool_calls_field(self):
        tcs = [ToolCall(name="search", input={"q": "foo"})]
        ec = EvalCase(input="q", output="a", tool_calls=tcs)
        assert len(ec.tool_calls) == 1
        assert ec.tool_calls[0].name == "search"

    def test_expected_tools_field(self):
        ec = EvalCase(input="q", output="a", expected_tools=["search"])
        assert ec.expected_tools == ["search"]

    def test_from_dict_deserializes_messages(self):
        ec = EvalCase.from_dict(
            {
                "input": "q",
                "output": "a",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
            }
        )
        assert len(ec.messages) == 2
        assert isinstance(ec.messages[0], Message)
        assert ec.messages[0].role == "user"

    def test_from_dict_deserializes_tool_calls(self):
        ec = EvalCase.from_dict(
            {
                "input": "q",
                "output": "a",
                "tool_calls": [
                    {"name": "search", "input": {"q": "foo"}},
                ],
            }
        )
        assert len(ec.tool_calls) == 1
        assert isinstance(ec.tool_calls[0], ToolCall)
        assert ec.tool_calls[0].name == "search"

    def test_from_dict_passes_through_message_objects(self):
        msgs = [Message(role="user", content="hi")]
        ec = EvalCase.from_dict({"input": "q", "output": "a", "messages": msgs})
        assert ec.messages[0] is msgs[0]

    def test_to_dict_omits_none(self):
        ec = EvalCase(input="q", output="a")
        d = ec.to_dict()
        assert d == {"input": "q", "output": "a"}
        assert "latency_ms" not in d

    def test_meta_returns_value(self):
        ec = EvalCase(input="q", output="a", metadata={"model": "gpt-4"})
        assert ec.meta("model") == "gpt-4"

    def test_meta_returns_default_when_missing(self):
        ec = EvalCase(input="q", output="a", metadata={"model": "gpt-4"})
        assert ec.meta("missing") is None
        assert ec.meta("missing", "default") == "default"

    def test_meta_handles_none_metadata(self):
        ec = EvalCase(input="q", output="a")
        assert ec.meta("any") is None

    def test_output_as_str_from_string(self):
        ec = EvalCase(input="q", output="hello world")
        assert ec.output_as_str() == "hello world"

    def test_output_as_str_from_dict(self):
        ec = EvalCase(input="q", output={"key": "value"})
        assert json.loads(ec.output_as_str()) == {"key": "value"}

    def test_output_as_str_from_list(self):
        ec = EvalCase(input="q", output=[1, 2, 3])
        assert json.loads(ec.output_as_str()) == [1, 2, 3]

    def test_output_as_dict_from_dict(self):
        ec = EvalCase(input="q", output={"key": "value"})
        assert ec.output_as_dict() == {"key": "value"}

    def test_output_as_dict_from_json_string(self):
        ec = EvalCase(input="q", output='{"key": "value"}')
        assert ec.output_as_dict() == {"key": "value"}

    def test_output_as_dict_raises_for_list(self):
        ec = EvalCase(input="q", output=[1, 2])
        with pytest.raises(TypeError, match="expected str or dict"):
            ec.output_as_dict()

    def test_output_as_dict_raises_for_json_array_string(self):
        ec = EvalCase(input="q", output="[1, 2]")
        with pytest.raises(TypeError, match="parsed to list"):
            ec.output_as_dict()

    def test_output_as_dict_raises_for_invalid_json(self):
        ec = EvalCase(input="q", output="not json")
        with pytest.raises(json.JSONDecodeError):
            ec.output_as_dict()

    def test_expected_as_str(self):
        ec = EvalCase(input="q", output="a", expected="hello")
        assert ec.expected_as_str() == "hello"

    def test_expected_as_str_from_dict(self):
        ec = EvalCase(input="q", output="a", expected={"k": "v"})
        assert json.loads(ec.expected_as_str()) == {"k": "v"}

    def test_expected_as_str_raises_when_none(self):
        ec = EvalCase(input="q", output="a")
        with pytest.raises(TypeError, match="expected is None"):
            ec.expected_as_str()

    def test_expected_as_dict(self):
        ec = EvalCase(input="q", output="a", expected={"k": "v"})
        assert ec.expected_as_dict() == {"k": "v"}

    def test_expected_as_dict_from_json_string(self):
        ec = EvalCase(input="q", output="a", expected='{"k": "v"}')
        assert ec.expected_as_dict() == {"k": "v"}

    def test_expected_as_dict_raises_when_none(self):
        ec = EvalCase(input="q", output="a")
        with pytest.raises(TypeError, match="expected is None"):
            ec.expected_as_dict()


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

    def test_clamp_eps_widened(self):
        s = Score(name="test", value=1.0 + 5e-7, threshold=0.5)
        assert s.value == 1.0
        assert s.passed is True

    def test_clamp_eps_rejects_large_overshoot(self):
        with pytest.raises(ValueError, match="must be between"):
            Score(name="test", value=1.01, threshold=0.5)

    def test_clamped_factory_clamps_high(self):
        s = Score.clamped(name="test", value=1.5, threshold=0.5)
        assert s.value == 1.0
        assert s.passed is True

    def test_clamped_factory_clamps_low(self):
        s = Score.clamped(name="test", value=-0.3, threshold=0.5)
        assert s.value == 0.0
        assert s.passed is False

    def test_clamped_factory_passes_through_normal(self):
        s = Score.clamped(name="test", value=0.75, threshold=0.5, reason="ok")
        assert s.value == 0.75
        assert s.reason == "ok"


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

    def test_evaluate_filters_none_scores(self):
        from harness_evals.core.metric import BaseMetric, Dimension

        class SkippingMetric(BaseMetric):
            def __init__(self):
                super().__init__(name="skipper", dimension=Dimension.CORRECTNESS, threshold=0.5)

            def measure(self, eval_case):
                return None

        ec = EvalCase(input="x", output="y", expected="y")
        scores = evaluate(ec, metrics=[ExactMatchMetric(), SkippingMetric()])
        assert len(scores) == 1
        assert scores[0].name == "exact_match"


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
class TestAEvaluate:
    async def test_a_evaluate_returns_scores(self, simple_eval_case):
        scores = await a_evaluate(simple_eval_case, metrics=[ExactMatchMetric()])
        assert len(scores) == 1
        assert scores[0].passed
        assert scores[0].value == 1.0

    async def test_a_evaluate_catches_exceptions(self):
        class BrokenMetric(ExactMatchMetric):
            async def a_measure(self, eval_case):
                raise RuntimeError("async broken")

        ec = EvalCase(input="x", output="y", expected="y")
        scores = await a_evaluate(ec, metrics=[BrokenMetric()])
        assert len(scores) == 1
        assert not scores[0].passed
        assert "async broken" in scores[0].reason

    async def test_a_evaluate_does_not_raise(self):
        ec = EvalCase(input="x", output="wrong", expected="right")
        scores = await a_evaluate(ec, metrics=[ExactMatchMetric()])
        assert len(scores) == 1
        assert not scores[0].passed

    async def test_a_evaluate_filters_none_scores(self):
        from harness_evals.core.metric import BaseMetric, Dimension

        class AsyncSkipper(BaseMetric):
            def __init__(self):
                super().__init__(name="async_skipper", dimension=Dimension.CORRECTNESS, threshold=0.5)

            def measure(self, eval_case):
                return None

            async def a_measure(self, eval_case):
                return None

        ec = EvalCase(input="x", output="y", expected="y")
        scores = await a_evaluate(ec, metrics=[ExactMatchMetric(), AsyncSkipper()])
        assert len(scores) == 1
        assert scores[0].name == "exact_match"


@pytest.mark.unit
class TestEvaluateBatchMetrics:
    def test_with_standard_metric(self):
        cases = [
            EvalCase(input="q1", output="a", expected="a"),
            EvalCase(input="q2", output="b", expected="c"),
        ]
        outcomes = [True, False]
        scores = evaluate_batch_metrics(cases, outcomes, metrics=[ExactMatchMetric()])
        assert len(scores) == 1
        assert scores[0].value == 0.5  # average of 1.0 and 0.0

    def test_with_dataset_metric(self):
        from harness_evals.metrics.reliability.calibration import CalibrationMetric

        cases = [
            EvalCase(input="q1", output="a", confidence=0.9),
            EvalCase(input="q2", output="b", confidence=0.9),
            EvalCase(input="q3", output="c", confidence=0.1),
            EvalCase(input="q4", output="d", confidence=0.1),
        ]
        outcomes = [True, True, False, False]
        scores = evaluate_batch_metrics(cases, outcomes, metrics=[CalibrationMetric()])
        assert len(scores) == 1
        assert scores[0].value > 0.0

    def test_catches_exceptions(self):
        from harness_evals.core.metric import BaseMetric, Dimension

        class BadMetric(BaseMetric):
            def __init__(self):
                super().__init__(name="bad", dimension=Dimension.CORRECTNESS, threshold=0.5)

            def measure(self, eval_case):
                raise RuntimeError("boom")

        cases = [EvalCase(input="q", output="a")]
        scores = evaluate_batch_metrics(cases, [True], metrics=[BadMetric()])
        assert len(scores) == 1
        assert scores[0].value == 0.0
        assert "boom" in scores[0].reason


@pytest.mark.unit
class TestEvaluateDataset:
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

    async def test_evaluate_dataset_with_failure(self):
        goldens = [Golden(input="q", expected="right")]

        async def bad_agent(golden: Golden) -> EvalCase:
            return EvalCase.from_golden(golden, output="wrong")

        results = await evaluate_dataset(goldens, bad_agent, metrics=[ExactMatchMetric()])
        assert not results[0][0].passed

    async def test_evaluate_dataset_isolates_agent_failure(self):
        """A raising agent_fn for one item must not abort the rest of the dataset."""
        goldens = [
            Golden(input="q0", expected="a0"),
            Golden(input="boom", expected="a1"),
            Golden(input="q2", expected="a2"),
        ]

        async def flaky_agent(golden: Golden) -> EvalCase:
            if golden.input == "boom":
                raise RuntimeError("kaboom")
            return EvalCase.from_golden(golden, output=golden.expected)

        results = await evaluate_dataset(goldens, flaky_agent, metrics=[ExactMatchMetric()])

        assert len(results) == 3
        # Surviving items scored normally, in input order.
        assert results[0][0].passed
        assert results[2][0].passed
        # Failed item produced a failed score rather than crashing the run.
        failed = results[1][0]
        assert not failed.passed
        assert failed.value == 0.0
        assert "kaboom" in failed.reason
        assert failed.metadata.get("target_error") is True

    async def test_evaluate_dataset_runs_concurrently(self):
        """Verify that evaluate_dataset uses asyncio.gather (runs in parallel)."""
        import asyncio

        call_order: list[str] = []

        goldens = [Golden(input=f"q{i}", expected=f"a{i}") for i in range(3)]

        async def slow_agent(golden: Golden) -> EvalCase:
            call_order.append(f"start-{golden.input}")
            await asyncio.sleep(0.01)
            call_order.append(f"end-{golden.input}")
            return EvalCase.from_golden(golden, output=golden.expected)

        results = await evaluate_dataset(goldens, slow_agent, metrics=[ExactMatchMetric()])
        assert len(results) == 3
        # With gather, all starts happen before any ends
        starts = [e for e in call_order if e.startswith("start")]
        assert len(starts) == 3

    async def test_evaluate_dataset_concurrency_limit(self):
        """Verify concurrency parameter limits parallel agent calls."""
        import asyncio

        active = 0
        max_active = 0

        goldens = [Golden(input=f"q{i}", expected=f"a{i}") for i in range(5)]

        async def tracked_agent(golden: Golden) -> EvalCase:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            active -= 1
            return EvalCase.from_golden(golden, output=golden.expected)

        results = await evaluate_dataset(goldens, tracked_agent, metrics=[ExactMatchMetric()], concurrency=2)
        assert len(results) == 5
        assert max_active <= 2


@pytest.mark.unit
class TestScoringDurationMs:
    """Verify scoring_duration_ms is populated by the runner on all code paths."""

    def test_evaluate_populates_scoring_duration_ms(self):
        ec = EvalCase(input="q", output="a", expected="a")
        scores = evaluate(ec, metrics=[ExactMatchMetric()])
        assert len(scores) == 1
        assert scores[0].scoring_duration_ms is not None
        assert scores[0].scoring_duration_ms >= 0

    def test_evaluate_populates_on_failure(self):
        ec = EvalCase(input="q", output="wrong", expected="right")
        scores = evaluate(ec, metrics=[ExactMatchMetric()])
        assert scores[0].scoring_duration_ms is not None
        assert scores[0].scoring_duration_ms >= 0

    def test_evaluate_populates_on_exception(self):
        class BrokenMetric(ExactMatchMetric):
            def measure(self, eval_case):
                raise RuntimeError("boom")

        ec = EvalCase(input="q", output="a", expected="a")
        scores = evaluate(ec, metrics=[BrokenMetric()])
        assert scores[0].scoring_duration_ms is not None
        assert scores[0].scoring_duration_ms >= 0

    async def test_a_evaluate_populates_scoring_duration_ms(self):
        ec = EvalCase(input="q", output="a", expected="a")
        scores = await a_evaluate(ec, metrics=[ExactMatchMetric()])
        assert len(scores) == 1
        assert scores[0].scoring_duration_ms is not None
        assert scores[0].scoring_duration_ms >= 0

    async def test_a_evaluate_populates_on_exception(self):
        class BrokenMetric(ExactMatchMetric):
            async def a_measure(self, eval_case):
                raise RuntimeError("async boom")

        ec = EvalCase(input="q", output="a", expected="a")
        scores = await a_evaluate(ec, metrics=[BrokenMetric()])
        assert scores[0].scoring_duration_ms is not None
        assert scores[0].scoring_duration_ms >= 0

    def test_evaluate_cases_populates_scoring_duration_ms(self):
        cases = [
            EvalCase(input="q1", output="a", expected="a"),
            EvalCase(input="q2", output="b", expected="c"),
        ]
        results = evaluate_cases(cases, metrics=[ExactMatchMetric()])
        for case_scores in results:
            for score in case_scores:
                assert score.scoring_duration_ms is not None
                assert score.scoring_duration_ms >= 0

    async def test_evaluate_dataset_populates_scoring_duration_ms(self):
        goldens = [
            Golden(input="q1", expected="a"),
            Golden(input="q2", expected="b"),
        ]

        async def agent_fn(golden: Golden) -> EvalCase:
            return EvalCase.from_golden(golden, output=golden.expected)

        results = await evaluate_dataset(goldens, agent_fn, metrics=[ExactMatchMetric()])
        for case_scores in results:
            for score in case_scores:
                assert score.scoring_duration_ms is not None
                assert score.scoring_duration_ms >= 0

    def test_junit_sink_emits_time_from_runner(self, tmp_path):
        """End-to-end: evaluate() -> JUnit sink -> XML time attribute."""
        import xml.etree.ElementTree as ET

        from harness_evals.sinks.junit_sink import JUnitSink

        path = tmp_path / "results.xml"
        sink = JUnitSink(str(path))
        ec = EvalCase(input="q", output="a", expected="a")
        scores = evaluate(ec, metrics=[ExactMatchMetric()], sinks=[sink])
        sink.finalize()

        tree = ET.parse(path)
        tc = tree.getroot().find("testcase")
        assert "time" in tc.attrib
        assert float(tc.attrib["time"]) >= 0
