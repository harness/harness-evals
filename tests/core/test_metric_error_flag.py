"""A metric that raises yields a value-0 score flagged ``metric_error`` so sinks can tell it from a real low score."""

from __future__ import annotations

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.runner import a_evaluate, evaluate, evaluate_batch_metrics
from harness_evals.core.score import Score

pytestmark = pytest.mark.unit

CASE = EvalCase(input="q", output="a")


class RaisingMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__("raises", Dimension.CORRECTNESS)

    def measure(self, eval_case: EvalCase) -> Score:
        raise RuntimeError("judge failed")

    async def a_measure(self, eval_case: EvalCase) -> Score:
        raise RuntimeError("judge failed")


class LowMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__("low", Dimension.CORRECTNESS)

    def measure(self, eval_case: EvalCase) -> Score:
        return Score(name=self.name, value=0.0, threshold=self.threshold, reason="genuinely bad")

    async def a_measure(self, eval_case: EvalCase) -> Score:
        return self.measure(CASE)


def _assert_flagged(score: Score) -> None:
    assert score.value == 0.0
    assert score.reason.startswith("Metric raised:")
    assert score.metadata["metric_error"] is True


def test_evaluate_flags_raising_metric() -> None:
    (score,) = evaluate(CASE, [RaisingMetric()])
    _assert_flagged(score)


async def test_a_evaluate_flags_raising_metric() -> None:
    (score,) = await a_evaluate(CASE, [RaisingMetric()])
    _assert_flagged(score)


def test_evaluate_batch_metrics_flags_raising_metric() -> None:
    (score,) = evaluate_batch_metrics([CASE], [True], [RaisingMetric()])
    _assert_flagged(score)


async def test_low_score_is_not_flagged() -> None:
    (sync_score,) = evaluate(CASE, [LowMetric()])
    (async_score,) = await a_evaluate(CASE, [LowMetric()])
    for score in (sync_score, async_score):
        assert score.value == 0.0
        assert "metric_error" not in (score.metadata or {})


async def test_flag_is_per_metric() -> None:
    scores = await a_evaluate(CASE, [LowMetric(), RaisingMetric()])
    by_name = {s.name: s for s in scores}
    assert "metric_error" not in (by_name["low"].metadata or {})
    assert by_name["raises"].metadata["metric_error"] is True
