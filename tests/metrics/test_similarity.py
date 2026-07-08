"""Tests for similarity metrics: Levenshtein, BLEU, EmbeddingSimilarity."""

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.llm.embedding import BaseEmbedding
from harness_evals.metrics.similarity.bleu import BLEUMetric
from harness_evals.metrics.similarity.embedding_similarity import EmbeddingSimilarityMetric
from harness_evals.metrics.similarity.levenshtein import LevenshteinMetric


class MockEmbedding(BaseEmbedding):
    """Returns vectors based on text length for deterministic testing."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for t in texts:
            length = float(len(t))
            vectors.append([length, length * 0.5, 1.0])
        return vectors


@pytest.mark.unit
class TestLevenshteinMetric:
    def test_identical_strings(self):
        ec = EvalCase(input="q", output="hello", expected="hello")
        score = LevenshteinMetric().measure(ec)
        assert score.value == 1.0
        assert score.passed

    def test_completely_different(self):
        ec = EvalCase(input="q", output="abc", expected="xyz")
        score = LevenshteinMetric(threshold=0.5).measure(ec)
        assert score.value == 0.0

    def test_similar_strings(self):
        ec = EvalCase(input="q", output="kitten", expected="sitting")
        score = LevenshteinMetric(threshold=0.5).measure(ec)
        assert 0.0 < score.value < 1.0

    def test_both_empty(self):
        ec = EvalCase(input="q", output="", expected="")
        score = LevenshteinMetric().measure(ec)
        assert score.value == 1.0
        assert score.reason

    def test_one_empty(self):
        ec = EvalCase(input="q", output="hello", expected="")
        score = LevenshteinMetric().measure(ec)
        assert score.value == 0.0

    def test_expected_none(self):
        ec = EvalCase(input="q", output="hello")
        score = LevenshteinMetric().measure(ec)
        assert score.value == 0.0
        assert "None" in score.reason

    def test_edit_distance_in_metadata(self):
        ec = EvalCase(input="q", output="cat", expected="hat")
        score = LevenshteinMetric().measure(ec)
        assert score.metadata["edit_distance"] == 1


@pytest.mark.unit
class TestBLEUMetric:
    def test_identical(self):
        ec = EvalCase(input="q", output="the cat sat on the mat", expected="the cat sat on the mat")
        score = BLEUMetric().measure(ec)
        assert score.value > 0.9
        assert score.passed

    def test_completely_different(self):
        ec = EvalCase(input="q", output="xyz abc", expected="the cat sat on the mat")
        score = BLEUMetric().measure(ec)
        assert score.value < 0.1

    def test_expected_none(self):
        ec = EvalCase(input="q", output="hello world")
        score = BLEUMetric().measure(ec)
        assert score.value == 0.0

    def test_empty_expected(self):
        ec = EvalCase(input="q", output="hello world", expected="")
        score = BLEUMetric().measure(ec)
        assert score.value == 0.0

    def test_partial_overlap(self):
        ec = EvalCase(input="q", output="the cat sat on", expected="the cat sat on the mat")
        score = BLEUMetric(threshold=0.3).measure(ec)
        assert 0.0 < score.value < 1.0

    def test_short_hypothesis_with_low_n(self):
        ec = EvalCase(input="q", output="the cat sat", expected="the cat sat on the mat")
        score = BLEUMetric(threshold=0.3, max_n=2).measure(ec)
        assert 0.0 < score.value < 1.0

    def test_short_hypothesis_smoothing_nonzero(self):
        ec = EvalCase(input="q", output="the cat", expected="the cat sat on the mat")
        score = BLEUMetric().measure(ec)
        assert score.value > 0.0


@pytest.mark.unit
class TestEmbeddingSimilarityMetric:
    async def test_identical_texts(self):
        emb = MockEmbedding()
        ec = EvalCase(input="q", output="hello", expected="hello")
        score = await EmbeddingSimilarityMetric(embedding=emb).a_measure(ec)
        assert score.value == pytest.approx(1.0)
        assert score.passed

    async def test_different_texts(self):
        emb = MockEmbedding()
        ec = EvalCase(input="q", output="hi", expected="hello world foo bar baz")
        score = await EmbeddingSimilarityMetric(embedding=emb).a_measure(ec)
        assert score.value < 1.0

    async def test_expected_none(self):
        emb = MockEmbedding()
        ec = EvalCase(input="q", output="hello")
        score = await EmbeddingSimilarityMetric(embedding=emb).a_measure(ec)
        assert score.value == 0.0

    def test_sync_measure(self):
        emb = MockEmbedding()
        ec = EvalCase(input="q", output="test", expected="test")
        score = EmbeddingSimilarityMetric(embedding=emb).measure(ec)
        assert score.value == pytest.approx(1.0)
