"""Tests for RAG metrics with mocked LLM."""

import pytest

from harness_evals import EvalCase
from harness_evals.llm.base import BaseLLM
from harness_evals.metrics.rag.answer_relevancy import AnswerRelevancyMetric
from harness_evals.metrics.rag.context_precision import ContextPrecisionMetric
from harness_evals.metrics.rag.context_recall import ContextRecallMetric
from harness_evals.metrics.rag.faithfulness import FaithfulnessMetric


class MockLLM(BaseLLM):
    def __init__(self, responses: list[dict] | None = None, default: dict | None = None):
        self._responses = list(responses) if responses else []
        self._default = default or {}
        self._call_idx = 0

    async def generate(self, prompt: str, **kwargs) -> str:
        return ""

    async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
        if self._call_idx < len(self._responses):
            result = self._responses[self._call_idx]
            self._call_idx += 1
            return result
        return self._default


@pytest.mark.unit
class TestFaithfulnessMetric:
    @pytest.mark.asyncio
    async def test_all_supported(self):
        llm = MockLLM(
            responses=[
                {"claims": ["Paris is the capital", "It has the Eiffel Tower"]},
                {
                    "verdicts": [
                        {"claim": "Paris is the capital", "verdict": "supported"},
                        {"claim": "It has the Eiffel Tower", "verdict": "supported"},
                    ]
                },
            ]
        )
        metric = FaithfulnessMetric(llm=llm, threshold=0.7)
        ec = EvalCase(
            input="Tell me about Paris",
            output="Paris is the capital. It has the Eiffel Tower.",
            context=["Paris is the capital of France.", "The Eiffel Tower is in Paris."],
        )
        score = await metric.a_measure(ec)
        assert score.value == 1.0
        assert score.passed

    @pytest.mark.asyncio
    async def test_partial_support(self):
        llm = MockLLM(
            responses=[
                {"claims": ["claim1", "claim2", "claim3"]},
                {
                    "verdicts": [
                        {"claim": "claim1", "verdict": "supported"},
                        {"claim": "claim2", "verdict": "unsupported"},
                        {"claim": "claim3", "verdict": "supported"},
                    ]
                },
            ]
        )
        metric = FaithfulnessMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="q", output="a", context=["ctx"])
        score = await metric.a_measure(ec)
        assert abs(score.value - 2 / 3) < 0.01
        assert not score.passed  # 0.667 < 0.7

    @pytest.mark.asyncio
    async def test_no_context(self):
        llm = MockLLM()
        metric = FaithfulnessMetric(llm=llm)
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "No context" in score.reason

    @pytest.mark.asyncio
    async def test_no_claims(self):
        llm = MockLLM(responses=[{"claims": []}])
        metric = FaithfulnessMetric(llm=llm)
        ec = EvalCase(input="q", output="ok", context=["ctx"])
        score = await metric.a_measure(ec)
        assert score.value == 1.0


@pytest.mark.unit
class TestAnswerRelevancyMetric:
    @pytest.mark.asyncio
    async def test_relevant(self):
        llm = MockLLM(default={"reasoning": "Direct answer", "score": 0.95})
        metric = AnswerRelevancyMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="What is 2+2?", output="4")
        score = await metric.a_measure(ec)
        assert score.value == 0.95
        assert score.passed

    @pytest.mark.asyncio
    async def test_irrelevant(self):
        llm = MockLLM(default={"reasoning": "Off topic", "score": 0.1})
        metric = AnswerRelevancyMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="What is 2+2?", output="The sky is blue")
        score = await metric.a_measure(ec)
        assert score.value == 0.1
        assert not score.passed


@pytest.mark.unit
class TestContextPrecisionMetric:
    @pytest.mark.asyncio
    async def test_all_relevant(self):
        llm = MockLLM(
            default={
                "verdicts": [
                    {"chunk_index": 0, "relevant": True},
                    {"chunk_index": 1, "relevant": True},
                ]
            }
        )
        metric = ContextPrecisionMetric(llm=llm, threshold=0.5)
        ec = EvalCase(input="q", output="a", context=["c1", "c2"])
        score = await metric.a_measure(ec)
        assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_half_relevant(self):
        llm = MockLLM(
            default={
                "verdicts": [
                    {"chunk_index": 0, "relevant": True},
                    {"chunk_index": 1, "relevant": False},
                ]
            }
        )
        metric = ContextPrecisionMetric(llm=llm, threshold=0.5)
        ec = EvalCase(input="q", output="a", context=["c1", "c2"])
        score = await metric.a_measure(ec)
        assert score.value == 0.5

    @pytest.mark.asyncio
    async def test_no_context(self):
        llm = MockLLM()
        metric = ContextPrecisionMetric(llm=llm)
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 0.0


@pytest.mark.unit
class TestContextRecallMetric:
    @pytest.mark.asyncio
    async def test_full_recall(self):
        llm = MockLLM(
            default={
                "statements": [
                    {"statement": "s1", "attributed": True},
                    {"statement": "s2", "attributed": True},
                ]
            }
        )
        metric = ContextRecallMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="q", output="a", expected="expected", context=["ctx"])
        score = await metric.a_measure(ec)
        assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_partial_recall(self):
        llm = MockLLM(
            default={
                "statements": [
                    {"statement": "s1", "attributed": True},
                    {"statement": "s2", "attributed": False},
                    {"statement": "s3", "attributed": False},
                ]
            }
        )
        metric = ContextRecallMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="q", output="a", expected="expected", context=["ctx"])
        score = await metric.a_measure(ec)
        assert abs(score.value - 1 / 3) < 0.01

    @pytest.mark.asyncio
    async def test_no_context(self):
        llm = MockLLM()
        metric = ContextRecallMetric(llm=llm)
        ec = EvalCase(input="q", output="a", expected="e")
        score = await metric.a_measure(ec)
        assert score.value == 0.0

    @pytest.mark.asyncio
    async def test_no_expected(self):
        llm = MockLLM()
        metric = ContextRecallMetric(llm=llm)
        ec = EvalCase(input="q", output="a", context=["ctx"])
        score = await metric.a_measure(ec)
        assert score.value == 0.0
