"""Tests for PromptAlignment metric."""

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import Dimension
from harness_evals.metrics.llm_judge.prompt_alignment import PromptAlignmentMetric
from tests.conftest import MockLLM


@pytest.mark.unit
class TestPromptAlignmentMetric:
    def test_dimension_is_correctness(self):
        llm = MockLLM()
        metric = PromptAlignmentMetric(llm=llm, prompt_instructions=["Be polite"])
        assert metric.dimension == Dimension.CORRECTNESS

    def test_empty_instructions_raises(self):
        llm = MockLLM()
        with pytest.raises(ValueError, match="at least one instruction"):
            PromptAlignmentMetric(llm=llm, prompt_instructions=[])

    async def test_all_instructions_followed(self):
        """All 3 instructions followed → score 1.0."""
        llm = MockLLM(
            responses=[
                {
                    "results": [
                        {"instruction": "Reply in uppercase", "followed": True, "reasoning": "output is uppercase"},
                        {"instruction": "Be concise", "followed": True, "reasoning": "output is short"},
                        {"instruction": "Include a greeting", "followed": True, "reasoning": "starts with Hello"},
                    ]
                }
            ]
        )
        metric = PromptAlignmentMetric(
            llm=llm,
            prompt_instructions=["Reply in uppercase", "Be concise", "Include a greeting"],
        )
        ec = EvalCase(input="How are you?", output="HELLO! I'M FINE.")
        score = await metric.a_measure(ec)
        assert score.value == 1.0
        assert score.passed
        assert score.metadata["followed"] == 3

    async def test_partial_alignment(self):
        """2 of 3 instructions followed → score ~0.67."""
        llm = MockLLM(
            responses=[
                {
                    "results": [
                        {"instruction": "Reply in uppercase", "followed": False, "reasoning": "output is lowercase"},
                        {"instruction": "Be concise", "followed": True, "reasoning": "short"},
                        {"instruction": "Include a greeting", "followed": True, "reasoning": "has greeting"},
                    ]
                }
            ]
        )
        metric = PromptAlignmentMetric(
            llm=llm,
            prompt_instructions=["Reply in uppercase", "Be concise", "Include a greeting"],
        )
        ec = EvalCase(input="How are you?", output="hello! i'm fine.")
        score = await metric.a_measure(ec)
        assert abs(score.value - 2 / 3) < 0.01
        assert score.passed  # threshold 0.5
        assert "FAILED" in score.reason
        assert "uppercase" in score.reason

    async def test_no_instructions_followed(self):
        """0 of 2 → score 0.0."""
        llm = MockLLM(
            responses=[
                {
                    "results": [
                        {"instruction": "A", "followed": False, "reasoning": "nope"},
                        {"instruction": "B", "followed": False, "reasoning": "nope"},
                    ]
                }
            ]
        )
        metric = PromptAlignmentMetric(llm=llm, prompt_instructions=["A", "B"])
        ec = EvalCase(input="q", output="a")
        score = await metric.a_measure(ec)
        assert score.value == 0.0
        assert not score.passed

    async def test_single_instruction(self):
        """Single instruction followed → 1.0."""
        llm = MockLLM(
            responses=[
                {"results": [{"instruction": "Be JSON", "followed": True, "reasoning": "is json"}]}
            ]
        )
        metric = PromptAlignmentMetric(llm=llm, prompt_instructions=["Be JSON"])
        ec = EvalCase(input="q", output='{"key": "value"}')
        score = await metric.a_measure(ec)
        assert score.value == 1.0
