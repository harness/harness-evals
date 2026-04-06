"""Tests for Summarization metric."""

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import Dimension
from harness_evals.metrics.llm_judge.summarization import SummarizationMetric
from tests.conftest import MockLLM


@pytest.mark.unit
class TestSummarizationMetric:
    def test_dimension_is_correctness(self):
        llm = MockLLM()
        metric = SummarizationMetric(llm=llm)
        assert metric.dimension == Dimension.CORRECTNESS

    async def test_high_alignment_high_coverage(self):
        """Both alignment and coverage high → high score."""
        llm = MockLLM(
            responses=[
                {"reasoning": "Summary is factually accurate", "score": 0.95},  # alignment
                {"questions": ["Does summary mention X?", "Does summary mention Y?"]},  # generate questions
                {"answers": [{"question": "X?", "covered": True}, {"question": "Y?", "covered": True}], "score": 0.9},  # coverage
            ]
        )
        metric = SummarizationMetric(llm=llm)
        ec = EvalCase(
            input="Paris is the capital of France with a population of 2.1 million.",
            output="Paris, France's capital, has 2.1M people.",
        )
        score = await metric.a_measure(ec)
        assert score.value == 0.9  # min(0.95, 0.9)
        assert score.passed  # threshold 0.5
        assert score.metadata["alignment_score"] == 0.95
        assert score.metadata["coverage_score"] == 0.9

    async def test_low_alignment_caps_score(self):
        """Poor alignment (hallucination) caps the final score."""
        llm = MockLLM(
            responses=[
                {"reasoning": "Summary contains hallucinated facts", "score": 0.2},  # alignment
                {"questions": ["Q1?"]},
                {"answers": [{"question": "Q1?", "covered": True}], "score": 0.9},  # coverage
            ]
        )
        metric = SummarizationMetric(llm=llm)
        ec = EvalCase(input="original text", output="bad summary with hallucinations")
        score = await metric.a_measure(ec)
        assert score.value == 0.2  # min(0.2, 0.9)
        assert not score.passed

    async def test_low_coverage_caps_score(self):
        """Good alignment but missing details caps the score."""
        llm = MockLLM(
            responses=[
                {"reasoning": "Factually correct", "score": 0.95},  # alignment
                {"questions": ["Q1?", "Q2?", "Q3?"]},
                {"answers": [], "score": 0.3},  # coverage
            ]
        )
        metric = SummarizationMetric(llm=llm)
        ec = EvalCase(input="long detailed text", output="very short summary")
        score = await metric.a_measure(ec)
        assert score.value == 0.3  # min(0.95, 0.3)

    async def test_custom_assessment_questions(self):
        """User-provided questions skip the generation step."""
        llm = MockLLM(
            responses=[
                {"reasoning": "ok", "score": 0.8},  # alignment
                # No question generation call
                {"answers": [{"question": "Q?", "covered": True}], "score": 0.85},  # coverage
            ]
        )
        metric = SummarizationMetric(
            llm=llm,
            assessment_questions=["Does the summary preserve the main point?"],
        )
        ec = EvalCase(input="original", output="summary")
        score = await metric.a_measure(ec)
        assert score.value == 0.8  # min(0.8, 0.85)
        assert len(score.metadata["questions"]) == 1

    async def test_both_zero(self):
        """Both scores zero → 0.0."""
        llm = MockLLM(
            responses=[
                {"reasoning": "completely wrong", "score": 0.0},
                {"questions": ["Q?"]},
                {"answers": [], "score": 0.0},
            ]
        )
        metric = SummarizationMetric(llm=llm)
        ec = EvalCase(input="text", output="unrelated")
        score = await metric.a_measure(ec)
        assert score.value == 0.0
