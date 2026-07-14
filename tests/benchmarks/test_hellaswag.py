"""Tests for HellaSwag benchmark."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from harness_evals.benchmarks.hellaswag import HellaSwag


@pytest.mark.unit
class TestHellaSwag:
    async def test_format_prompt(self):
        hs = HellaSwag()
        item = {
            "ctx": "A man is sitting on a roof.",
            "activity_label": "Roof repair",
            "endings": ["He falls off.", "He fixes a shingle.", "He eats lunch.", "He flies away."],
            "label": "1",
        }
        prompt = hs.format_prompt(item, shots=0)
        assert "A man is sitting on a roof" in prompt
        assert "A. He falls off." in prompt
        assert "B. He fixes a shingle." in prompt

    async def test_score_correct_numeric_label(self):
        hs = HellaSwag()
        item = {"endings": ["a", "b", "c", "d"], "label": "1"}
        value, _ = hs.score_response(item, "B")
        assert value == 1.0

    async def test_score_correct_int_label(self):
        hs = HellaSwag()
        item = {"endings": ["a", "b", "c", "d"], "label": 2}
        value, _ = hs.score_response(item, "C")
        assert value == 1.0

    async def test_score_incorrect(self):
        hs = HellaSwag()
        item = {"endings": ["a", "b", "c", "d"], "label": "0"}
        value, _ = hs.score_response(item, "The answer is C")
        assert value == 0.0

    async def test_run_with_mock(self, mock_llm_factory):
        hs = HellaSwag()
        sample_data = [
            {"ctx": "Context", "activity_label": "Test", "endings": ["a", "b", "c", "d"], "label": "0"},
        ]

        with patch.object(hs, "load_dataset", new_callable=AsyncMock, return_value=sample_data):
            llm = mock_llm_factory(responses=["A"])
            result = await hs.run(llm, shots=0)

        assert result.accuracy == 1.0
