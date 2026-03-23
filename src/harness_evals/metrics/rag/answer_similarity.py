"""Answer Similarity metric — semantic similarity between output and expected via embeddings."""

from __future__ import annotations

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score
from harness_evals.llm.embedding import BaseEmbedding
from harness_evals.metrics.similarity.embedding_similarity import EmbeddingSimilarityMetric


class AnswerSimilarityMetric(EmbeddingSimilarityMetric):
    """Cosine similarity between embeddings of output and expected answer.

    Named RAG metric for direct mapping to ``@builtin/answer_similarity``
    in the aiEvals catalog. Delegates computation to
    ``EmbeddingSimilarityMetric`` but uses the name ``answer_similarity``.
    Requires ``eval_case.expected``.
    """

    def __init__(self, embedding: BaseEmbedding, threshold: float = 0.8, **kwargs: object) -> None:
        super().__init__(embedding=embedding, threshold=threshold, **kwargs)
        self.name = "answer_similarity"

    def measure(self, eval_case: EvalCase) -> Score:
        score = super().measure(eval_case)
        return Score(
            name=self.name,
            value=score.value,
            threshold=score.threshold,
            reason=score.reason,
            metadata=score.metadata,
        )

    async def a_measure(self, eval_case: EvalCase) -> Score:
        score = await super().a_measure(eval_case)
        return Score(
            name=self.name,
            value=score.value,
            threshold=score.threshold,
            reason=score.reason,
            metadata=score.metadata,
        )
