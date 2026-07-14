"""Tests for MMLU benchmark."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from harness_evals.benchmarks.mmlu import MMLU


@pytest.mark.unit
class TestMMLU:
    def test_invalid_subject(self):
        with pytest.raises(ValueError, match="Unknown MMLU subjects"):
            MMLU(subjects=["not_a_real_subject"])

    def test_valid_subjects(self):
        mmlu = MMLU(subjects=["abstract_algebra", "anatomy"])
        assert mmlu.subjects == ["abstract_algebra", "anatomy"]

    def test_default_all_subjects(self):
        mmlu = MMLU()
        assert len(mmlu.subjects) == 57

    def test_default_shots(self):
        mmlu = MMLU()
        assert mmlu.default_shots == 5

    def test_custom_shots(self):
        mmlu = MMLU(shots=3)
        assert mmlu.default_shots == 3

    async def test_format_prompt(self):
        mmlu = MMLU(subjects=["abstract_algebra"])
        mmlu._few_shot_examples = {"abstract_algebra": []}

        item = {
            "question": "Find the degree of the extension Q(sqrt(2), sqrt(3)) over Q.",
            "choices": ["4", "2", "6", "8"],
            "answer": 0,
            "subject": "abstract_algebra",
        }
        prompt = mmlu.format_prompt(item, shots=0)
        assert "abstract_algebra" not in prompt
        assert "Abstract Algebra" in prompt
        assert "Find the degree" in prompt
        assert "A. 4" in prompt
        assert "B. 2" in prompt

    async def test_score_response_correct(self):
        mmlu = MMLU(subjects=["abstract_algebra"])
        item = {"choices": ["4", "2", "6", "8"], "answer": 0}
        value, reason = mmlu.score_response(item, "The answer is A")
        assert value == 1.0
        assert reason is None

    async def test_score_response_incorrect(self):
        mmlu = MMLU(subjects=["abstract_algebra"])
        item = {"choices": ["4", "2", "6", "8"], "answer": 0}
        value, reason = mmlu.score_response(item, "The answer is B")
        assert value == 0.0

    async def test_score_response_unparseable(self):
        mmlu = MMLU(subjects=["abstract_algebra"])
        item = {"choices": ["4", "2", "6", "8"], "answer": 0}
        value, reason = mmlu.score_response(item, "I have no idea")
        assert value == 0.0
        assert "Could not parse" in reason

    async def test_run_with_mock(self, mock_llm_factory):
        mmlu = MMLU(subjects=["abstract_algebra"])
        sample_data = [
            {"question": "Q1", "choices": ["A", "B", "C", "D"], "answer": 0},
            {"question": "Q2", "choices": ["A", "B", "C", "D"], "answer": 1},
        ]
        dev_data = [{"question": "Dev", "choices": ["X", "Y"], "answer": 0}]

        with patch.object(mmlu, "load_dataset", new_callable=AsyncMock, return_value=sample_data):
            mmlu._few_shot_examples = {"abstract_algebra": dev_data}
            llm = mock_llm_factory(responses=["A", "B"])
            result = await mmlu.run(llm, shots=0)

        assert result.num_total == 2
        assert result.num_correct == 2
        assert result.accuracy == 1.0
