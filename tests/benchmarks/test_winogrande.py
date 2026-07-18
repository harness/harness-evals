"""Tests for WinoGrande benchmark."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from harness_evals.benchmarks.winogrande import WinoGrande, _extract_option


@pytest.mark.unit
class TestWinoGrande:
    def test_extract_option(self):
        assert _extract_option("1") == "1"
        assert _extract_option("2") == "2"
        assert _extract_option("The answer is 1") == "1"
        assert _extract_option("Option 2 is correct") == "2"

    def test_extract_option_no_match(self):
        assert _extract_option("I don't know") is None

    async def test_format_prompt(self):
        wg = WinoGrande()
        item = {
            "sentence": "The trophy doesn't fit in _ because it is too big.",
            "option1": "the trophy",
            "option2": "the suitcase",
            "answer": "1",
        }
        prompt = wg.format_prompt(item, shots=0)
        assert "The trophy" in prompt
        assert "1. the trophy" in prompt
        assert "2. the suitcase" in prompt

    async def test_score_correct(self):
        wg = WinoGrande()
        item = {"sentence": "...", "option1": "a", "option2": "b", "answer": "1"}
        value, _ = wg.score_response(item, "1")
        assert value == 1.0

    async def test_score_incorrect(self):
        wg = WinoGrande()
        item = {"sentence": "...", "option1": "a", "option2": "b", "answer": "1"}
        value, _ = wg.score_response(item, "2")
        assert value == 0.0

    async def test_run_with_mock(self, mock_llm_factory):
        wg = WinoGrande()
        sample_data = [
            {"sentence": "S1", "option1": "a", "option2": "b", "answer": "1"},
            {"sentence": "S2", "option1": "c", "option2": "d", "answer": "2"},
        ]

        with patch.object(wg, "load_dataset", new_callable=AsyncMock, return_value=sample_data):
            llm = mock_llm_factory(responses=["1", "2"])
            result = await wg.run(llm, shots=0)

        assert result.accuracy == 1.0
