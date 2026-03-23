"""Tests for embedding infrastructure: BaseEmbedding, _cosine_similarity, OpenAIEmbedding."""

import pytest

from harness_evals.llm.embedding import BaseEmbedding, _cosine_similarity


class MockEmbedding(BaseEmbedding):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0, 0.0] for t in texts]


@pytest.mark.unit
class TestCosineHelper:
    def test_identical_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_both_zero(self):
        assert _cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_similar_vectors(self):
        sim = _cosine_similarity([1.0, 1.0], [1.0, 0.9])
        assert sim > 0.99


@pytest.mark.unit
class TestBaseEmbedding:
    async def test_embed(self):
        emb = MockEmbedding()
        result = await emb.embed(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == 3

    def test_embed_sync(self):
        emb = MockEmbedding()
        result = emb.embed_sync(["hello"])
        assert len(result) == 1
        assert result[0][0] == 5.0
