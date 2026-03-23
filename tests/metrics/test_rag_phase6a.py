"""Tests for Phase 6a RAG metrics: ContextRelevancy, ContextEntityRecall, AnswerSimilarity, AnswerCorrectness."""

import pytest

from harness_evals import EvalCase
from harness_evals.llm.base import BaseLLM
from harness_evals.llm.embedding import BaseEmbedding
from harness_evals.metrics.rag.answer_correctness import AnswerCorrectnessMetric
from harness_evals.metrics.rag.answer_similarity import AnswerSimilarityMetric
from harness_evals.metrics.rag.context_entity_recall import ContextEntityRecallMetric
from harness_evals.metrics.rag.context_relevancy import ContextRelevancyMetric


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


class MockEmbedding(BaseEmbedding):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), float(len(t)) * 0.5, 1.0] for t in texts]


@pytest.mark.unit
class TestContextRelevancyMetric:
    async def test_all_relevant(self):
        llm = MockLLM(
            default={
                "verdicts": [
                    {"chunk_index": 0, "relevant": True},
                    {"chunk_index": 1, "relevant": True},
                ]
            }
        )
        metric = ContextRelevancyMetric(llm=llm, threshold=0.5)
        ec = EvalCase(
            input="What is Python?", output="a", context=["Python is a language", "Python was created by Guido"]
        )
        score = await metric.a_measure(ec)
        assert score.value == 1.0
        assert score.passed

    async def test_half_relevant(self):
        llm = MockLLM(
            default={
                "verdicts": [
                    {"chunk_index": 0, "relevant": True},
                    {"chunk_index": 1, "relevant": False},
                ]
            }
        )
        metric = ContextRelevancyMetric(llm=llm, threshold=0.5)
        ec = EvalCase(input="q", output="a", context=["relevant", "irrelevant"])
        score = await metric.a_measure(ec)
        assert score.value == 0.5
        assert score.passed

    async def test_no_context(self):
        llm = MockLLM()
        metric = ContextRelevancyMetric(llm=llm)
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "No context" in score.reason

    def test_sync_measure(self):
        llm = MockLLM(default={"verdicts": [{"chunk_index": 0, "relevant": True}]})
        metric = ContextRelevancyMetric(llm=llm, threshold=0.5)
        ec = EvalCase(input="q", output="a", context=["ctx"])
        score = metric.measure(ec)
        assert score.value == 1.0


@pytest.mark.unit
class TestContextEntityRecallMetric:
    async def test_full_recall(self):
        llm = MockLLM(
            responses=[
                {"entities": ["Paris", "France"]},
                {"entities": ["Paris", "France", "Europe"]},
            ]
        )
        metric = ContextEntityRecallMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="q", output="a", expected="Paris is in France", context=["Paris, France, Europe"])
        score = await metric.a_measure(ec)
        assert score.value == 1.0
        assert score.passed

    async def test_partial_recall(self):
        llm = MockLLM(
            responses=[
                {"entities": ["Paris", "France", "Berlin"]},
                {"entities": ["Paris"]},
            ]
        )
        metric = ContextEntityRecallMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="q", output="a", expected="Paris France Berlin", context=["Paris only"])
        score = await metric.a_measure(ec)
        assert abs(score.value - 1 / 3) < 0.01

    async def test_no_context(self):
        llm = MockLLM()
        metric = ContextEntityRecallMetric(llm=llm)
        ec = EvalCase(input="q", output="a", expected="e")
        score = await metric.a_measure(ec)
        assert score.value == 0.0

    async def test_no_expected(self):
        llm = MockLLM()
        metric = ContextEntityRecallMetric(llm=llm)
        ec = EvalCase(input="q", output="a", context=["ctx"])
        score = await metric.a_measure(ec)
        assert score.value == 0.0

    async def test_no_entities_in_expected(self):
        llm = MockLLM(responses=[{"entities": []}])
        metric = ContextEntityRecallMetric(llm=llm)
        ec = EvalCase(input="q", output="a", expected="the", context=["ctx"])
        score = await metric.a_measure(ec)
        assert score.value == 1.0


@pytest.mark.unit
class TestAnswerSimilarityMetric:
    async def test_identical(self):
        emb = MockEmbedding()
        ec = EvalCase(input="q", output="hello", expected="hello")
        score = await AnswerSimilarityMetric(embedding=emb).a_measure(ec)
        assert score.value == pytest.approx(1.0)

    async def test_expected_none(self):
        emb = MockEmbedding()
        ec = EvalCase(input="q", output="hello")
        score = await AnswerSimilarityMetric(embedding=emb).a_measure(ec)
        assert score.value == 0.0

    def test_sync_measure(self):
        emb = MockEmbedding()
        ec = EvalCase(input="q", output="test", expected="test")
        score = AnswerSimilarityMetric(embedding=emb).measure(ec)
        assert score.value == pytest.approx(1.0)


@pytest.mark.unit
class TestAnswerCorrectnessMetric:
    async def test_fully_correct(self):
        llm = MockLLM(default={"TP": ["stmt1", "stmt2"], "FP": [], "FN": []})
        emb = MockEmbedding()
        metric = AnswerCorrectnessMetric(llm=llm, embedding=emb, threshold=0.7)
        ec = EvalCase(input="q", output="correct answer", expected="correct answer")
        score = await metric.a_measure(ec)
        assert score.value > 0.9
        assert score.passed

    async def test_partially_correct(self):
        llm = MockLLM(default={"TP": ["s1"], "FP": ["s2"], "FN": ["s3"]})
        emb = MockEmbedding()
        metric = AnswerCorrectnessMetric(llm=llm, embedding=emb, threshold=0.5)
        ec = EvalCase(input="q", output="partial", expected="full answer")
        score = await metric.a_measure(ec)
        assert 0.0 < score.value < 1.0

    async def test_expected_none(self):
        llm = MockLLM()
        emb = MockEmbedding()
        metric = AnswerCorrectnessMetric(llm=llm, embedding=emb)
        ec = EvalCase(input="q", output="answer")
        score = await metric.a_measure(ec)
        assert score.value == 0.0

    async def test_empty_classification(self):
        llm = MockLLM(default={"TP": [], "FP": [], "FN": []})
        emb = MockEmbedding()
        metric = AnswerCorrectnessMetric(llm=llm, embedding=emb)
        ec = EvalCase(input="q", output="a", expected="a")
        score = await metric.a_measure(ec)
        assert score.metadata["f1"] == 1.0

    def test_sync_measure(self):
        llm = MockLLM(default={"TP": ["s1"], "FP": [], "FN": []})
        emb = MockEmbedding()
        metric = AnswerCorrectnessMetric(llm=llm, embedding=emb, threshold=0.5)
        ec = EvalCase(input="q", output="a", expected="a")
        score = metric.measure(ec)
        assert score.passed
