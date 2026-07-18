"""Tests for BoolQ benchmark."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from harness_evals.benchmarks.boolq import BoolQ, _extract_bool


@pytest.mark.unit
class TestBoolQ:
    def test_extract_bool_true(self):
        assert _extract_bool("True") is True
        assert _extract_bool("yes") is True
        assert _extract_bool("Yes, that is correct") is True

    def test_extract_bool_false(self):
        assert _extract_bool("False") is False
        assert _extract_bool("no") is False
        assert _extract_bool("No, that is incorrect") is False

    def test_extract_bool_ambiguous(self):
        # When both present, last one wins
        result = _extract_bool("No, actually yes")
        assert result is True

    def test_extract_bool_unparseable(self):
        assert _extract_bool("I'm not sure") is None

    async def test_format_prompt(self):
        boolq = BoolQ()
        item = {
            "passage": "The sky is blue due to Rayleigh scattering.",
            "question": "is the sky blue",
        }
        prompt = boolq.format_prompt(item, shots=0)
        assert "The sky is blue" in prompt
        assert "is the sky blue" in prompt
        assert "true or false" in prompt

    async def test_score_response_correct(self):
        boolq = BoolQ()
        item = {"passage": "...", "question": "...", "answer": True}
        value, _ = boolq.score_response(item, "True")
        assert value == 1.0

    async def test_score_response_incorrect(self):
        boolq = BoolQ()
        item = {"passage": "...", "question": "...", "answer": True}
        value, _ = boolq.score_response(item, "False")
        assert value == 0.0

    async def test_run_with_mock(self, mock_llm_factory):
        boolq = BoolQ()
        sample_data = [
            {"passage": "Water boils at 100C.", "question": "does water boil at 100 degrees", "answer": True},
            {"passage": "The moon is not a star.", "question": "is the moon a star", "answer": False},
        ]

        with patch.object(boolq, "load_dataset", new_callable=AsyncMock, return_value=sample_data):
            llm = mock_llm_factory(responses=["True", "False"])
            result = await boolq.run(llm, shots=0)

        assert result.accuracy == 1.0
        assert result.num_correct == 2
