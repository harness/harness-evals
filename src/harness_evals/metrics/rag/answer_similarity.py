"""Answer Similarity metric — semantic similarity between output and expected via embeddings."""

from __future__ import annotations

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.llm.embedding import BaseEmbedding, _cosine_similarity


class AnswerSimilarityMetric(BaseMetric):
    """Cosine similarity between embeddings of output and expected answer.

    Named RAG metric for direct mapping to ``@builtin/answer_similarity``
    in the aiEvals catalog. Uses the same cosine similarity math as
    ``EmbeddingSimilarityMetric`` but lives in the RAG category.
    Requires ``eval_case.expected``.
    """

    def __init__(self, embedding: BaseEmbedding, threshold: float = 0.8, **kwargs: object) -> None:
        super().__init__(name="answer_similarity", threshold=threshold, **kwargs)
        self.embedding = embedding

    def measure(self, eval_case: EvalCase) -> Score:
        return _run_async(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        if eval_case.expected is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="expected is None — cannot compute similarity",
            )

        vectors = await self.embedding.embed([str(eval_case.output), str(eval_case.expected)])
        similarity = _cosine_similarity(vectors[0], vectors[1])
        value = max(0.0, min(1.0, similarity))

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            metadata={"cosine_similarity": similarity},
        )
