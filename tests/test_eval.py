"""Tests for the run_eval() one-liner."""

from __future__ import annotations

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.golden import Golden
from harness_evals.core.score import Score
from harness_evals.eval import _CallableTarget, run_eval
from harness_evals.metrics.deterministic.exact_match import ExactMatchMetric

# ---------------------------------------------------------------------------
# _CallableTarget
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCallableTarget:
    async def test_wraps_async_callable(self) -> None:
        async def agent(golden: Golden) -> EvalCase:
            return EvalCase.from_golden(golden, output=golden.expected or "")

        target = _CallableTarget(agent)
        result = await target.ainvoke(Golden(input="hi", expected="hi"))
        assert result.output == "hi"

    async def test_wraps_sync_callable(self) -> None:
        def agent(golden: Golden) -> EvalCase:
            return EvalCase.from_golden(golden, output=golden.expected or "")

        target = _CallableTarget(agent)
        result = await target.ainvoke(Golden(input="hi", expected="hi"))
        assert result.output == "hi"

    async def test_context_manager(self) -> None:
        def agent(golden: Golden) -> EvalCase:
            return EvalCase.from_golden(golden, output="x")

        async with _CallableTarget(agent) as t:
            result = await t.ainvoke(Golden(input="hi"))
            assert result.output == "x"


# ---------------------------------------------------------------------------
# run_eval() with list[Golden]
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEvalWithGoldens:
    def test_eval_with_literal_goldens(self) -> None:
        goldens = [
            Golden(input="2+2", expected="4"),
            Golden(input="1+1", expected="2"),
        ]

        async def agent(golden: Golden) -> EvalCase:
            return EvalCase.from_golden(golden, output=golden.expected or "")

        scores = run_eval(
            "test-eval",
            data=goldens,
            target=agent,
            metrics=[ExactMatchMetric()],
            sinks=[],
        )

        assert len(scores) == 2
        assert all(s[0].passed for s in scores)

    def test_eval_with_sync_callable(self) -> None:
        goldens = [Golden(input="hi", expected="hi")]

        def agent(golden: Golden) -> EvalCase:
            return EvalCase.from_golden(golden, output=golden.expected or "")

        scores = run_eval("test", data=goldens, target=agent, metrics=[ExactMatchMetric()], sinks=[])
        assert len(scores) == 1
        assert scores[0][0].passed

    def test_eval_with_base_target(self) -> None:
        from harness_evals.targets.base import BaseTarget

        goldens = [Golden(input="hi", expected="hi")]

        class MyTarget(BaseTarget):
            async def ainvoke(self, golden: Golden) -> EvalCase:
                return EvalCase.from_golden(golden, output=golden.expected or "")

        scores = run_eval("test", data=goldens, target=MyTarget(), metrics=[ExactMatchMetric()], sinks=[])
        assert scores[0][0].passed


# ---------------------------------------------------------------------------
# run_eval() with ref string
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEvalWithRefString:
    def test_eval_with_local_file(self, tmp_path) -> None:
        dataset_path = tmp_path / "goldens.jsonl"
        dataset_path.write_text('{"input": "x", "expected": "x"}\n')

        async def agent(golden: Golden) -> EvalCase:
            return EvalCase.from_golden(golden, output=golden.expected or "")

        scores = run_eval(
            "test",
            data=str(dataset_path),
            target=agent,
            metrics=[ExactMatchMetric()],
            sinks=[],
        )

        assert len(scores) == 1
        assert scores[0][0].passed


# ---------------------------------------------------------------------------
# run_eval() with baseline
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEvalWithBaseline:
    def test_baseline_path_shorthand(self, tmp_path) -> None:
        from harness_evals.baseline.json_store import JsonBaselineStore

        baseline_dir = str(tmp_path / "baselines")
        store = JsonBaselineStore(baseline_dir=baseline_dir)
        store.save("prev", {"exact_match": [Score(name="exact_match", value=1.0, threshold=1.0)]})

        goldens = [Golden(input="x", expected="x")]

        async def agent(golden: Golden) -> EvalCase:
            return EvalCase.from_golden(golden, output="x")

        scores = run_eval(
            "test",
            data=goldens,
            target=agent,
            metrics=[ExactMatchMetric()],
            sinks=[],
            baseline=baseline_dir,
        )
        assert scores[0][0].passed

    def test_baseline_regression_raises(self, tmp_path) -> None:
        from harness_evals.baseline.json_store import JsonBaselineStore
        from harness_evals.errors import BaselineRegressionError

        baseline_dir = str(tmp_path / "baselines")
        store = JsonBaselineStore(baseline_dir=baseline_dir)
        store.save("prev", {"exact_match": [Score(name="exact_match", value=1.0, threshold=1.0)]})

        goldens = [Golden(input="x", expected="x")]

        async def agent(golden: Golden) -> EvalCase:
            return EvalCase.from_golden(golden, output="WRONG")

        with pytest.raises(BaselineRegressionError):
            run_eval(
                "test",
                data=goldens,
                target=agent,
                metrics=[ExactMatchMetric()],
                sinks=[],
                baseline=baseline_dir,
            )
