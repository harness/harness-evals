"""Embedding similarity metric — cosine similarity of embedding vectors."""

from __future__ import annotations

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.llm.embedding import BaseEmbedding, _cosine_similarity


class EmbeddingSimilarityMetric(BaseMetric):
    """Cosine similarity between embedding vectors of output and expected.

    Score is the cosine similarity clamped to [0.0, 1.0].
    Requires an ``embedding`` provider for vectorization.
    """

    def __init__(self, embedding: BaseEmbedding, threshold: float = 0.8, **kwargs: object) -> None:
        super().__init__(name="embedding_similarity", dimension=Dimension.CORRECTNESS, threshold=threshold, **kwargs)
        self.embedding = embedding

    def measure(self, eval_case: EvalCase) -> Score:
        return _run_async(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        if eval_case.expected is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="No expected answer provided to compute similarity against (expected is None)",
            )

        vectors = await self.embedding.embed([str(eval_case.output), str(eval_case.expected)])
        similarity = _cosine_similarity(vectors[0], vectors[1])
        value = max(0.0, min(1.0, similarity))

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=f"Output is {value * 100:.0f}% semantically similar to the expected answer (cosine similarity = {similarity:.4f})",
            metadata={"cosine_similarity": similarity},
        )
