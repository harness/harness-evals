"""Tests for ARC benchmark."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from harness_evals.benchmarks.arc import ARC


@pytest.mark.unit
class TestARC:
    def test_invalid_subset(self):
        with pytest.raises(ValueError, match="subset must be"):
            ARC(subset="invalid")

    def test_default_challenge(self):
        arc = ARC()
        assert arc.subset == "ARC-Challenge"
        assert "challenge" in arc.name

    def test_easy_subset(self):
        arc = ARC(subset="ARC-Easy")
        assert arc.subset == "ARC-Easy"

    async def test_format_prompt(self):
        arc = ARC()
        item = {
            "question": "What gas do plants absorb?",
            "choices": {"label": ["A", "B", "C", "D"], "text": ["Oxygen", "CO2", "Nitrogen", "Helium"]},
            "answerKey": "B",
        }
        prompt = arc.format_prompt(item, shots=0)
        assert "What gas do plants absorb?" in prompt
        assert "A. Oxygen" in prompt
        assert "B. CO2" in prompt

    async def test_score_response_correct(self):
        arc = ARC()
        item = {
            "choices": {"label": ["A", "B", "C", "D"], "text": ["O2", "CO2", "N2", "He"]},
            "answerKey": "B",
        }
        value, _ = arc.score_response(item, "The answer is B")
        assert value == 1.0

    async def test_score_response_incorrect(self):
        arc = ARC()
        item = {
            "choices": {"label": ["A", "B", "C", "D"], "text": ["O2", "CO2", "N2", "He"]},
            "answerKey": "B",
        }
        value, _ = arc.score_response(item, "A")
        assert value == 0.0

    async def test_run_with_mock(self, mock_llm_factory):
        arc = ARC()
        sample_data = [
            {
                "question": "Q1",
                "choices": {"label": ["A", "B", "C", "D"], "text": ["a", "b", "c", "d"]},
                "answerKey": "A",
                "id": "test1",
            },
        ]

        with patch.object(arc, "load_dataset", new_callable=AsyncMock, return_value=sample_data):
            llm = mock_llm_factory(responses=["A"])
            result = await arc.run(llm, shots=0)

        assert result.accuracy == 1.0
