"""Tests for RAG metrics with a mocked LLM/embedding.

Covers three families that share the same mocking helpers:

* Single-turn core: Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall.
* Phase 6a: ContextRelevancy, ContextEntityRecall, AnswerSimilarity, AnswerCorrectness.
* Turn-level (conversational): TurnFaithfulness, TurnContextual{Precision,Recall,Relevancy}.
"""

from unittest.mock import AsyncMock

import pytest

from harness_evals import EvalCase
from harness_evals.core.score import Score
from harness_evals.core.types import Message
from harness_evals.llm.base import BaseLLM
from harness_evals.llm.embedding import BaseEmbedding
from harness_evals.metrics.rag.answer_correctness import AnswerCorrectnessMetric
from harness_evals.metrics.rag.answer_relevancy import AnswerRelevancyMetric
from harness_evals.metrics.rag.answer_similarity import AnswerSimilarityMetric
from harness_evals.metrics.rag.context_entity_recall import ContextEntityRecallMetric
from harness_evals.metrics.rag.context_precision import ContextPrecisionMetric
from harness_evals.metrics.rag.context_recall import ContextRecallMetric
from harness_evals.metrics.rag.context_relevancy import ContextRelevancyMetric
from harness_evals.metrics.rag.conversational import (
    TurnContextualPrecisionMetric,
    TurnContextualRecallMetric,
    TurnContextualRelevancyMetric,
    TurnFaithfulnessMetric,
)
from harness_evals.metrics.rag.faithfulness import FaithfulnessMetric


class MockLLM(BaseLLM):
    """Returns queued responses in order, then falls back to ``default``."""

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


def _two_turn_case() -> EvalCase:
    """A 2-turn RAG conversation with per-turn retrieval_context + expected."""
    return EvalCase(
        input="What is RAG?",
        output="Chunking splits documents.",
        messages=[
            Message(role="user", content="What is RAG?"),
            Message(
                role="assistant",
                content="RAG combines retrieval and generation.",
                retrieval_context=["RAG combines retrieval with generation."],
                expected="RAG combines retrieval and generation.",
            ),
            Message(role="user", content="How does chunking help?"),
            Message(
                role="assistant",
                content="Chunking splits documents into pieces.",
                retrieval_context=["Chunking splits documents into smaller pieces."],
                expected="Chunking splits documents into pieces.",
            ),
        ],
    )


# ############################################################################
# Single-turn core RAG metrics
# ############################################################################


@pytest.mark.unit
class TestFaithfulnessMetric:
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

    async def test_no_context(self):
        llm = MockLLM()
        metric = FaithfulnessMetric(llm=llm)
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert "No context" in score.reason

    async def test_no_claims(self):
        llm = MockLLM(responses=[{"claims": []}])
        metric = FaithfulnessMetric(llm=llm)
        ec = EvalCase(input="q", output="ok", context=["ctx"])
        score = await metric.a_measure(ec)
        assert score.value == 1.0

    async def test_more_verdicts_than_claims_clamped(self):
        """Judge returns more supported verdicts than claims → clamp to 1.0, no error."""
        llm = MockLLM(
            responses=[
                {"claims": ["c1", "c2"]},
                {
                    "verdicts": [
                        {"claim": "c1", "verdict": "supported"},
                        {"claim": "c2", "verdict": "supported"},
                        {"claim": "c3", "verdict": "supported"},
                        {"claim": "c4", "verdict": "supported"},
                    ]
                },
            ]
        )
        metric = FaithfulnessMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="q", output="a", context=["ctx"])
        score = await metric.a_measure(ec)
        assert score.value == 1.0
        assert score.passed


@pytest.mark.unit
class TestAnswerRelevancyMetric:
    async def test_relevant(self):
        llm = MockLLM(default={"reasoning": "Direct answer", "score": 0.95})
        metric = AnswerRelevancyMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="What is 2+2?", output="4")
        score = await metric.a_measure(ec)
        assert score.value == 0.95
        assert score.passed

    async def test_irrelevant(self):
        llm = MockLLM(default={"reasoning": "Off topic", "score": 0.1})
        metric = AnswerRelevancyMetric(llm=llm, threshold=0.7)
        ec = EvalCase(input="What is 2+2?", output="The sky is blue")
        score = await metric.a_measure(ec)
        assert score.value == 0.1
        assert not score.passed


@pytest.mark.unit
class TestContextPrecisionMetric:
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

    async def test_no_context(self):
        llm = MockLLM()
        metric = ContextPrecisionMetric(llm=llm)
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 0.0

    async def test_more_verdicts_than_chunks_clamped(self):
        """Judge returns more relevant verdicts than chunks → clamp to 1.0, no error."""
        llm = MockLLM(
            default={
                "verdicts": [
                    {"chunk_index": 0, "relevant": True},
                    {"chunk_index": 1, "relevant": True},
                    {"chunk_index": 2, "relevant": True},
                ]
            }
        )
        metric = ContextPrecisionMetric(llm=llm, threshold=0.5)
        ec = EvalCase(input="q", output="a", context=["c1", "c2"])
        score = await metric.a_measure(ec)
        assert score.value == 1.0


@pytest.mark.unit
class TestContextRecallMetric:
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

    async def test_no_context(self):
        llm = MockLLM()
        metric = ContextRecallMetric(llm=llm)
        ec = EvalCase(input="q", output="a", expected="e")
        score = await metric.a_measure(ec)
        assert score.value == 0.0

    async def test_no_expected(self):
        llm = MockLLM()
        metric = ContextRecallMetric(llm=llm)
        ec = EvalCase(input="q", output="a", context=["ctx"])
        score = await metric.a_measure(ec)
        assert score.value == 0.0


# ############################################################################
# Phase 6a RAG metrics: ContextRelevancy, ContextEntityRecall, AnswerSimilarity, AnswerCorrectness
# ############################################################################


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

    async def test_score_clamped_with_hallucinated_verdicts(self):
        llm = MockLLM(
            default={
                "verdicts": [
                    {"chunk_index": 0, "relevant": True},
                    {"chunk_index": 1, "relevant": True},
                    {"chunk_index": 2, "relevant": True},
                    {"chunk_index": 3, "relevant": True},
                ]
            }
        )
        metric = ContextRelevancyMetric(llm=llm, threshold=0.5)
        ec = EvalCase(input="q", output="a", context=["c1", "c2"])
        score = await metric.a_measure(ec)
        assert score.value == 1.0
        assert score.value <= 1.0


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

    async def test_metadata_keys(self):
        llm = MockLLM(default={"TP": ["s1", "s2"], "FP": ["s3"], "FN": []})
        emb = MockEmbedding()
        metric = AnswerCorrectnessMetric(llm=llm, embedding=emb, threshold=0.7)
        ec = EvalCase(input="q", output="answer", expected="answer")
        score = await metric.a_measure(ec)
        expected_keys = {"f1", "tp", "fp", "fn", "cosine_similarity", "factuality_weight", "similarity_weight"}
        assert expected_keys == set(score.metadata.keys())
        assert score.metadata["tp"] == 2
        assert score.metadata["fp"] == 1
        assert score.metadata["fn"] == 0
        assert score.metadata["cosine_similarity"] >= 0.0

    def test_sync_measure(self):
        llm = MockLLM(default={"TP": ["s1"], "FP": [], "FN": []})
        emb = MockEmbedding()
        metric = AnswerCorrectnessMetric(llm=llm, embedding=emb, threshold=0.5)
        ec = EvalCase(input="q", output="a", expected="a")
        score = metric.measure(ec)
        assert score.passed


# ############################################################################
# Turn-level (conversational) RAG metrics
# ############################################################################


@pytest.mark.unit
class TestTurnFaithfulness:
    async def test_all_turns_supported(self):
        llm = MockLLM(
            responses=[
                {"claims": ["RAG combines retrieval and generation"]},
                {"verdicts": [{"claim": "c", "verdict": "supported"}]},
                {"claims": ["Chunking splits documents"]},
                {"verdicts": [{"claim": "c", "verdict": "supported"}]},
            ]
        )
        score = await TurnFaithfulnessMetric(llm=llm, threshold=0.7).a_measure(_two_turn_case())
        assert score.value == 1.0
        assert score.passed
        assert score.metadata["n_scored_turns"] == 2
        assert [t["score"] for t in score.metadata["turn_scores"]] == [1.0, 1.0]
        # per-turn breakdown carries the assistant-turn index and message index
        assert score.metadata["turn_scores"][0]["turn"] == 0
        assert score.metadata["turn_scores"][0]["message_index"] == 1
        assert score.metadata["turn_scores"][1]["message_index"] == 3

    async def test_one_turn_unfaithful_drags_mean(self):
        llm = MockLLM(
            responses=[
                {"claims": ["c1"]},
                {"verdicts": [{"claim": "c1", "verdict": "supported"}]},
                {"claims": ["c2", "c3"]},
                {
                    "verdicts": [
                        {"claim": "c2", "verdict": "supported"},
                        {"claim": "c3", "verdict": "unsupported"},
                    ]
                },
            ]
        )
        score = await TurnFaithfulnessMetric(llm=llm, threshold=0.9).a_measure(_two_turn_case())
        # turn 1 = 1.0, turn 2 = 0.5 -> mean 0.75
        assert abs(score.value - 0.75) < 1e-9
        assert not score.passed

    async def test_turn2_retriever_failure_does_not_pass(self):
        # Regression for the review finding: turn 1 is grounded, turn 2 has a
        # retriever/trace failure (no retrieval_context). It must NOT score 1.0.
        case = EvalCase(
            input="q",
            output="a",
            messages=[
                Message(role="user", content="q1"),
                Message(
                    role="assistant",
                    content="grounded answer",
                    retrieval_context=["ctx supporting the answer"],
                ),
                Message(role="user", content="q2"),
                Message(role="assistant", content="ungrounded answer"),  # retriever failure
            ],
        )
        llm = MockLLM(
            responses=[
                {"claims": ["c1"]},
                {"verdicts": [{"claim": "c1", "verdict": "supported"}]},
            ]
        )
        score = await TurnFaithfulnessMetric(llm=llm, threshold=0.7).a_measure(case)
        assert score.value == 0.5  # mean(1.0, 0.0), not 1.0
        assert not score.passed
        assert score.metadata["n_scored_turns"] == 2

    def test_sync_measure_wrapper(self):
        llm = MockLLM(
            responses=[
                {"claims": ["c1"]},
                {"verdicts": [{"claim": "c1", "verdict": "supported"}]},
                {"claims": ["c2"]},
                {"verdicts": [{"claim": "c2", "verdict": "supported"}]},
            ]
        )
        score = TurnFaithfulnessMetric(llm=llm).measure(_two_turn_case())
        assert score.value == 1.0

    async def test_single_turn_conversation(self):
        # Acceptance criterion: a one-turn conversation (single user->assistant
        # exchange) scores as a normal single scored turn.
        case = EvalCase(
            input="What is RAG?",
            output="RAG combines retrieval and generation.",
            messages=[
                Message(role="user", content="What is RAG?"),
                Message(
                    role="assistant",
                    content="RAG combines retrieval and generation.",
                    retrieval_context=["RAG combines retrieval with generation."],
                ),
            ],
        )
        llm = MockLLM(
            responses=[
                {"claims": ["RAG combines retrieval and generation"]},
                {"verdicts": [{"claim": "c", "verdict": "supported"}]},
            ]
        )
        score = await TurnFaithfulnessMetric(llm=llm, threshold=0.7).a_measure(case)
        assert score.value == 1.0
        assert score.passed
        assert score.metadata["n_scored_turns"] == 1
        assert score.metadata["n_skipped_turns"] == 0
        assert score.metadata["turn_scores"][0]["turn"] == 0
        assert score.metadata["turn_scores"][0]["message_index"] == 1


# --------------------------------------------------------------------------- #
# TurnContextualPrecision / Relevancy — 1 LLM call per turn (chunk verdicts).
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestTurnContextualPrecision:
    async def test_all_relevant(self):
        llm = MockLLM(
            responses=[
                {"verdicts": [{"chunk_index": 0, "relevant": True}]},
                {"verdicts": [{"chunk_index": 0, "relevant": True}]},
            ]
        )
        score = await TurnContextualPrecisionMetric(llm=llm, threshold=0.5).a_measure(_two_turn_case())
        assert score.value == 1.0
        assert score.passed
        assert score.metadata["n_scored_turns"] == 2

    async def test_irrelevant_chunk_lowers_score(self):
        llm = MockLLM(
            responses=[
                {"verdicts": [{"chunk_index": 0, "relevant": True}]},
                {"verdicts": [{"chunk_index": 0, "relevant": False}]},
            ]
        )
        score = await TurnContextualPrecisionMetric(llm=llm, threshold=0.75).a_measure(_two_turn_case())
        # turn 1 = 1.0, turn 2 = 0.0 -> mean 0.5
        assert score.value == 0.5
        assert not score.passed


@pytest.mark.unit
class TestTurnContextualRelevancy:
    async def test_all_relevant(self):
        llm = MockLLM(
            responses=[
                {"verdicts": [{"chunk_index": 0, "relevant": True}]},
                {"verdicts": [{"chunk_index": 0, "relevant": True}]},
            ]
        )
        score = await TurnContextualRelevancyMetric(llm=llm).a_measure(_two_turn_case())
        assert score.value == 1.0
        assert score.passed


# --------------------------------------------------------------------------- #
# TurnContextualRecall — 1 LLM call per turn (statement attribution).
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestTurnContextualRecall:
    async def test_all_attributed(self):
        llm = MockLLM(
            responses=[
                {"statements": [{"statement": "s1", "attributed": True}]},
                {"statements": [{"statement": "s2", "attributed": True}]},
            ]
        )
        score = await TurnContextualRecallMetric(llm=llm, threshold=0.7).a_measure(_two_turn_case())
        assert score.value == 1.0
        assert score.passed

    @staticmethod
    def _first_turn_only_expected() -> EvalCase:
        # Only the first assistant turn has an expected answer.
        return EvalCase(
            input="q",
            output="a",
            messages=[
                Message(role="user", content="q1"),
                Message(
                    role="assistant",
                    content="a1",
                    retrieval_context=["ctx1"],
                    expected="e1",
                ),
                Message(role="user", content="q2"),
                Message(role="assistant", content="a2", retrieval_context=["ctx2"]),  # no expected
            ],
        )

    async def test_missing_expected_scores_zero_by_default(self):
        # Turn 1 attributed (1.0); turn 2 has no expected -> localized failure (0.0).
        # A missing per-turn input must drag the mean down, not vanish from it.
        llm = MockLLM(responses=[{"statements": [{"statement": "s1", "attributed": True}]}])
        score = await TurnContextualRecallMetric(llm=llm).a_measure(self._first_turn_only_expected())
        assert score.value == 0.5  # mean(1.0, 0.0)
        assert score.metadata["n_scored_turns"] == 2
        assert score.metadata["n_skipped_turns"] == 0
        # the failing turn is recorded, with why
        failed = score.metadata["turn_scores"][1]
        assert failed["score"] == 0.0
        assert failed["skipped"] is False
        assert "expected" in failed["reasoning"].lower()

    async def test_missing_expected_excluded_when_allow_skips(self):
        # Opt-in legacy behaviour: skip the un-scorable turn, average the rest.
        llm = MockLLM(responses=[{"statements": [{"statement": "s1", "attributed": True}]}])
        score = await TurnContextualRecallMetric(llm=llm, allow_skips=True).a_measure(self._first_turn_only_expected())
        assert score.value == 1.0
        assert score.metadata["n_scored_turns"] == 1
        assert score.metadata["n_skipped_turns"] == 1
        assert score.metadata["turn_scores"][1]["skipped"] is True


# --------------------------------------------------------------------------- #
# Shared edge cases.
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestConversationalRAGEdgeCases:
    async def test_no_messages_returns_zero(self):
        llm = MockLLM(default={"claims": []})
        ec = EvalCase(input="q", output="a")
        score = await TurnFaithfulnessMetric(llm=llm).a_measure(ec)
        assert score.value == 0.0
        assert not score.passed
        assert "No messages" in score.reason

    async def test_no_retrieval_context_scores_zero_by_default(self):
        # An assistant turn with no retrieval context is a localized failure (0.0),
        # not an ignored turn — otherwise a retriever failure would pass silently.
        llm = MockLLM(default={"verdicts": []})
        ec = EvalCase(
            input="q",
            output="a",
            messages=[
                Message(role="user", content="q1"),
                Message(role="assistant", content="a1"),  # no retrieval_context
            ],
        )
        score = await TurnContextualPrecisionMetric(llm=llm).a_measure(ec)
        assert score.value == 0.0
        assert not score.passed
        assert score.metadata["n_scored_turns"] == 1
        assert score.metadata["turn_scores"][0]["reasoning"] == "No retrieval_context on this turn"

    async def test_no_retrieval_context_skipped_returns_zero_with_allow_skips(self):
        # With allow_skips, the un-scorable turn is excluded; nothing left to score.
        llm = MockLLM(default={"verdicts": []})
        ec = EvalCase(
            input="q",
            output="a",
            messages=[
                Message(role="user", content="q1"),
                Message(role="assistant", content="a1"),  # no retrieval_context
            ],
        )
        score = await TurnContextualPrecisionMetric(llm=llm, allow_skips=True).a_measure(ec)
        assert score.value == 0.0
        assert score.metadata["n_scored_turns"] == 0
        assert score.metadata["n_skipped_turns"] == 1
        assert "No turns with retrieval context" in score.reason

    async def test_precision_missing_query_scores_zero_by_default(self):
        # Assistant turn with retrieval context but no preceding user query.
        llm = MockLLM(default={"verdicts": [{"chunk_index": 0, "relevant": True}]})
        ec = EvalCase(
            input="q",
            output="a",
            messages=[
                Message(role="assistant", content="a1", retrieval_context=["ctx"]),
            ],
        )
        score = await TurnContextualPrecisionMetric(llm=llm).a_measure(ec)
        assert score.value == 0.0
        assert score.metadata["n_scored_turns"] == 1
        assert score.metadata["turn_scores"][0]["reasoning"] == "No preceding user query for this turn"


# --------------------------------------------------------------------------- #
# Delegation wiring — the per-turn (query, chunks, answer, expected) mapping is
# the core of these metrics, but MockLLM ignores the prompt, so assert the exact
# EvalCase handed to each delegate rather than trusting queued responses.
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestConversationalRAGDelegation:
    async def test_delegate_receives_exact_per_turn_eval_case(self):
        metric = TurnContextualRecallMetric(llm=MockLLM())
        # Spy on the single-turn delegate; capture every EvalCase it is given.
        metric._delegate.a_measure = AsyncMock(return_value=Score(name="context_recall", value=1.0, threshold=0.7))

        await metric.a_measure(_two_turn_case())

        calls = metric._delegate.a_measure.call_args_list
        assert len(calls) == 2

        turn1 = calls[0].args[0]
        assert turn1.input == "What is RAG?"  # preceding user query
        assert turn1.output == "RAG combines retrieval and generation."  # assistant answer
        assert turn1.expected == "RAG combines retrieval and generation."  # per-turn expected
        assert turn1.context == ["RAG combines retrieval with generation."]  # per-turn chunks

        turn2 = calls[1].args[0]
        assert turn2.input == "How does chunking help?"
        assert turn2.output == "Chunking splits documents into pieces."
        assert turn2.expected == "Chunking splits documents into pieces."
        assert turn2.context == ["Chunking splits documents into smaller pieces."]


@pytest.mark.unit
def test_metrics_exported_from_harness_evals_metrics():
    # Users import these from the package root; guard the public export surface.
    import harness_evals.metrics as metrics_pkg

    for name in [
        "TurnFaithfulnessMetric",
        "TurnContextualPrecisionMetric",
        "TurnContextualRecallMetric",
        "TurnContextualRelevancyMetric",
    ]:
        assert hasattr(metrics_pkg, name), f"{name} not importable from harness_evals.metrics"
        assert name in metrics_pkg.__all__, f"{name} missing from harness_evals.metrics.__all__"
