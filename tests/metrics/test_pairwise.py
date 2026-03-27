"""Tests for PairwiseMetric with mocked LLM."""

import pytest

from harness_evals import EvalCase
from harness_evals.llm.base import BaseLLM
from harness_evals.metrics.llm_judge.pairwise import PairwiseMetric


class MockLLM(BaseLLM):
    def __init__(self, json_response: dict):
        self._json_response = json_response

    async def generate(self, prompt: str, **kwargs) -> str:
        return ""

    async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
        return self._json_response


@pytest.mark.unit
class TestPairwiseMetric:
    async def test_a_wins(self):
        llm = MockLLM({"reasoning": "A is better", "winner": "A", "score": 0.9})
        metric = PairwiseMetric(llm=llm, threshold=0.5)
        ec = EvalCase(input="q", output="good answer", expected="reference")
        score = await metric.a_measure(ec)
        assert score.value == 0.9
        assert score.passed
        assert score.metadata["winner"] == "A"

    async def test_b_wins(self):
        llm = MockLLM({"reasoning": "B is better", "winner": "B", "score": 0.1})
        metric = PairwiseMetric(llm=llm, threshold=0.5)
        ec = EvalCase(input="q", output="bad", expected="good")
        score = await metric.a_measure(ec)
        assert score.value == 0.1
        assert not score.passed

    async def test_tie(self):
        llm = MockLLM({"reasoning": "Both are equal", "winner": "tie", "score": 0.5})
        metric = PairwiseMetric(llm=llm, threshold=0.5)
        ec = EvalCase(input="q", output="answer", expected="answer")
        score = await metric.a_measure(ec)
        assert score.value == 0.5
        assert score.passed

    async def test_expected_none(self):
        llm = MockLLM({"reasoning": "n/a", "winner": "n/a", "score": 0.0})
        metric = PairwiseMetric(llm=llm)
        ec = EvalCase(input="q", output="answer")
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "expected is required" in score.reason

    async def test_clamps_score(self):
        llm = MockLLM({"reasoning": "test", "winner": "A", "score": 1.5})
        metric = PairwiseMetric(llm=llm)
        ec = EvalCase(input="q", output="a", expected="b")
        score = await metric.a_measure(ec)
        assert score.value == 1.0

    def test_sync_measure(self):
        llm = MockLLM({"reasoning": "ok", "winner": "A", "score": 0.7})
        metric = PairwiseMetric(llm=llm, threshold=0.5)
        ec = EvalCase(input="q", output="a", expected="b")
        score = metric.measure(ec)
        assert score.value == 0.7
        assert score.passed
