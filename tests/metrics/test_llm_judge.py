"""Tests for LLM-judged metrics with mocked LLM."""

import pytest

from harness_evals import EvalCase
from harness_evals.llm.base import BaseLLM
from harness_evals.metrics.llm_judge.geval import GEvalMetric
from harness_evals.metrics.llm_judge.rubric_judge import RubricJudgeMetric


class MockLLM(BaseLLM):
    def __init__(self, json_response: dict):
        self._json_response = json_response

    async def generate(self, prompt: str, **kwargs) -> str:
        return ""

    async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
        return self._json_response


@pytest.mark.unit
class TestGEvalMetric:
    @pytest.mark.asyncio
    async def test_high_score(self):
        llm = MockLLM({"reasoning": "Accurate and complete", "score": 0.9})
        metric = GEvalMetric(llm=llm, criteria="accuracy", threshold=0.7)
        ec = EvalCase(input="What is 2+2?", output="4", expected="4")
        score = await metric.a_measure(ec)
        assert score.value == 0.9
        assert score.passed
        assert "Accurate" in score.reason

    @pytest.mark.asyncio
    async def test_low_score(self):
        llm = MockLLM({"reasoning": "Wrong answer", "score": 0.2})
        metric = GEvalMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="What is 2+2?", output="5", expected="4")
        score = await metric.a_measure(ec)
        assert score.value == 0.2
        assert not score.passed

    @pytest.mark.asyncio
    async def test_clamps_score(self):
        llm = MockLLM({"reasoning": "test", "score": 1.5})
        metric = GEvalMetric(llm=llm)
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_missing_score_defaults_zero(self):
        llm = MockLLM({"reasoning": "test"})
        metric = GEvalMetric(llm=llm)
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 0.0

    def test_sync_measure(self):
        llm = MockLLM({"reasoning": "ok", "score": 0.8})
        metric = GEvalMetric(llm=llm, threshold=0.5)
        ec = EvalCase(input="q", output="a")
        score = metric.measure(ec)
        assert score.value == 0.8
        assert score.passed


@pytest.mark.unit
class TestRubricJudgeMetric:
    @pytest.mark.asyncio
    async def test_top_level(self):
        llm = MockLLM({"reasoning": "Excellent work", "level": 5})
        metric = RubricJudgeMetric(llm=llm, threshold=0.5)
        ec = EvalCase(input="q", output="a", expected="a")
        score = await metric.a_measure(ec)
        assert score.value == 1.0  # 5 out of 5 -> 1.0
        assert score.passed
        assert score.metadata["level"] == 5

    @pytest.mark.asyncio
    async def test_mid_level(self):
        llm = MockLLM({"reasoning": "Acceptable", "level": 3})
        metric = RubricJudgeMetric(llm=llm, threshold=0.5)
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 0.5  # (3-1)/(5-1) = 0.5

    @pytest.mark.asyncio
    async def test_lowest_level(self):
        llm = MockLLM({"reasoning": "Very poor", "level": 1})
        metric = RubricJudgeMetric(llm=llm, threshold=0.5)
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 0.0  # (1-1)/(5-1) = 0.0

    @pytest.mark.asyncio
    async def test_clamps_level(self):
        llm = MockLLM({"reasoning": "test", "level": 10})
        metric = RubricJudgeMetric(llm=llm)
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 1.0  # clamped to max

    @pytest.mark.asyncio
    async def test_custom_rubric(self):
        custom = {1: "Bad", 2: "OK", 3: "Great"}
        llm = MockLLM({"reasoning": "OK", "level": 2})
        metric = RubricJudgeMetric(llm=llm, rubric=custom, threshold=0.3)
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 0.5  # (2-1)/(3-1) = 0.5
        assert score.passed
