"""Tests for GSM8K benchmark."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from harness_evals.benchmarks.gsm8k import GSM8K, _extract_gsm8k_answer


@pytest.mark.unit
class TestGSM8K:
    def test_extract_gsm8k_answer(self):
        assert _extract_gsm8k_answer("Step 1: ...\n#### 42") == "42"
        assert _extract_gsm8k_answer("#### 1,234") == "1234"
        assert _extract_gsm8k_answer("The answer is 7") == "7"

    async def test_format_prompt(self):
        gsm = GSM8K()
        gsm._few_shot_examples = []
        item = {"question": "If John has 5 apples and gives 2 away, how many does he have?"}
        prompt = gsm.format_prompt(item, shots=0)
        assert "If John has 5 apples" in prompt
        assert "step by step" in prompt.lower()

    async def test_score_response_correct(self):
        gsm = GSM8K()
        item = {"answer": "Step by step...\n#### 42"}
        value, reason = gsm.score_response(item, "Let me think... #### 42")
        assert value == 1.0

    async def test_score_response_correct_different_format(self):
        gsm = GSM8K()
        item = {"answer": "#### 100"}
        value, reason = gsm.score_response(item, "The answer is 100")
        assert value == 1.0

    async def test_score_response_incorrect(self):
        gsm = GSM8K()
        item = {"answer": "#### 42"}
        value, reason = gsm.score_response(item, "The answer is 43")
        assert value == 0.0

    async def test_score_response_no_number(self):
        gsm = GSM8K()
        item = {"answer": "#### 42"}
        value, reason = gsm.score_response(item, "I don't know")
        assert value == 0.0
        assert "Could not extract" in reason

    async def test_run_with_mock(self, mock_llm_factory):
        gsm = GSM8K()
        sample_data = [
            {"question": "What is 5+3?", "answer": "#### 8"},
            {"question": "What is 2*6?", "answer": "#### 12"},
        ]

        with patch.object(gsm, "load_dataset", new_callable=AsyncMock, return_value=sample_data):
            gsm._few_shot_examples = []
            llm = mock_llm_factory(responses=["#### 8", "#### 12"])
            result = await gsm.run(llm, shots=0)

        assert result.num_total == 2
        assert result.num_correct == 2
        assert result.accuracy == 1.0
