"""Answer Correctness metric — composite of factual correctness (F1) and semantic similarity."""

from __future__ import annotations

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM
from harness_evals.llm.embedding import BaseEmbedding, _cosine_similarity

_CLASSIFY_PROMPT = """Compare the statements in the output with those in the expected answer.
Classify each statement into one of three categories:

- **TP** (true positive): statement appears in both output and expected (correct)
- **FP** (false positive): statement appears in output but NOT in expected (extra/wrong)
- **FN** (false negative): statement appears in expected but NOT in output (missing)

**Output**: {output}

**Expected**: {expected}

Respond with JSON:
{{"TP": ["stmt", ...], "FP": ["stmt", ...], "FN": ["stmt", ...]}}
"""

_CLASSIFY_SCHEMA = {
    "type": "object",
    "required": ["TP", "FP", "FN"],
    "properties": {
        "TP": {"type": "array", "items": {"type": "string"}},
        "FP": {"type": "array", "items": {"type": "string"}},
        "FN": {"type": "array", "items": {"type": "string"}},
    },
}


class AnswerCorrectnessMetric(BaseMetric):
    """Weighted combination of factual F1 score and semantic similarity.

    Factuality component: LLM classifies statements as TP/FP/FN -> F1 score.
    Similarity component: cosine similarity of output and expected embeddings.
    Final score = factuality_weight * F1 + similarity_weight * cosine_sim.
    Requires ``eval_case.expected``.
    """

    def __init__(
        self,
        llm: BaseLLM,
        embedding: BaseEmbedding,
        factuality_weight: float = 0.75,
        similarity_weight: float = 0.25,
        threshold: float = 0.7,
        **kwargs: object,
    ) -> None:
        super().__init__(name="answer_correctness", threshold=threshold, **kwargs)
        self.llm = llm
        self.embedding = embedding
        self.factuality_weight = factuality_weight
        self.similarity_weight = similarity_weight

    def measure(self, eval_case: EvalCase) -> Score:
        return _run_async(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        if eval_case.expected is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="expected is None — cannot compute correctness",
            )

        classify_result = await self.llm.generate_json(
            _CLASSIFY_PROMPT.format(output=eval_case.output, expected=eval_case.expected),
            _CLASSIFY_SCHEMA,
        )
        tp = len(classify_result.get("TP", []))
        fp = len(classify_result.get("FP", []))
        fn = len(classify_result.get("FN", []))

        if tp + fp + fn == 0:
            f1 = 1.0
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        vectors = await self.embedding.embed([str(eval_case.output), str(eval_case.expected)])
        similarity = _cosine_similarity(vectors[0], vectors[1])
        similarity = max(0.0, min(1.0, similarity))

        value = self.factuality_weight * f1 + self.similarity_weight * similarity
        value = max(0.0, min(1.0, value))

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=f"F1={f1:.3f} (TP={tp}, FP={fp}, FN={fn}), similarity={similarity:.3f}",
            metadata={
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "cosine_similarity": similarity,
                "factuality_weight": self.factuality_weight,
                "similarity_weight": self.similarity_weight,
            },
        )
