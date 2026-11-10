"""a_evaluate captures per-metric LLM usage into Score.metadata."""

from __future__ import annotations

import asyncio

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.runner import a_evaluate
from harness_evals.core.score import Score
from harness_evals.llm.usage import record_token_usage

pytestmark = pytest.mark.unit


class UsageMetric(BaseMetric):
    def __init__(self, name="usage", *, tokens=(4, 2), cost=0.001, metadata=None, delay=0.0, fail=False) -> None:
        super().__init__(name, Dimension.CORRECTNESS)
        self._tokens, self._cost, self._metadata, self._delay, self._fail = tokens, cost, metadata, delay, fail

    def measure(self, eval_case: EvalCase) -> Score:
        raise NotImplementedError

    async def a_measure(self, eval_case: EvalCase) -> Score:
        await asyncio.sleep(self._delay)
        if self._fail:
            raise RuntimeError("judge failed")
        record_token_usage(
            input_tokens=self._tokens[0], output_tokens=self._tokens[1], cost_usd=self._cost, model="test"
        )
        return Score(name=self.name, value=1.0, threshold=self.threshold, metadata=self._metadata)


CASE = EvalCase(input="q", output="a")


async def test_usage_attached_to_metadata() -> None:
    (score,) = await a_evaluate(CASE, [UsageMetric()])
    assert score.metadata["input_tokens"] == 4
    assert score.metadata["output_tokens"] == 2
    assert score.metadata["cost_usd"] == 0.001


async def test_metric_authored_metadata_wins() -> None:
    (score,) = await a_evaluate(CASE, [UsageMetric(metadata={"input_tokens": 99})])
    assert score.metadata["input_tokens"] == 99
    assert score.metadata["output_tokens"] == 2


async def test_concurrent_metrics_do_not_cross_contaminate() -> None:
    slow = UsageMetric("slow", tokens=(1, 1), cost=0.1, delay=0.03)
    fast = UsageMetric("fast", tokens=(50, 60), cost=0.5, delay=0.0)
    scores = await a_evaluate(CASE, [slow, fast])
    by_name = {s.name: s.metadata for s in scores}
    assert (by_name["slow"]["input_tokens"], by_name["slow"]["output_tokens"]) == (1, 1)
    assert (by_name["fast"]["input_tokens"], by_name["fast"]["output_tokens"]) == (50, 60)
    assert by_name["slow"]["cost_usd"] == 0.1
    assert by_name["fast"]["cost_usd"] == 0.5


async def test_exception_is_scored_zero() -> None:
    (score,) = await a_evaluate(CASE, [UsageMetric(fail=True)])
    assert score.value == 0.0
    assert score.reason.startswith("Metric raised:")
    usage_keys = {"input_tokens", "output_tokens", "cost_usd", "llm_spend_by_model", "llm_model"}
    assert not usage_keys & set(score.metadata or {})
