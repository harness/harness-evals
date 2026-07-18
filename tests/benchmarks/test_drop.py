"""Tests for DROP benchmark."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from harness_evals.benchmarks.drop import DROP, _clean_response, _get_all_answers


@pytest.mark.unit
class TestDROP:
    def test_get_all_answers_spans(self):
        item = {"answers_spans": {"spans": ["Paris", "France"]}}
        assert _get_all_answers(item) == ["Paris", "France"]

    def test_get_all_answers_number(self):
        item = {"answers_spans": {"spans": [], "number": "42"}}
        assert _get_all_answers(item) == ["42"]

    def test_get_all_answers_list(self):
        item = {"answers_spans": ["yes", "no"]}
        assert _get_all_answers(item) == ["yes", "no"]

    def test_clean_response(self):
        assert _clean_response("The answer is Paris.") == "Paris"
        assert _clean_response("  42  ") == "42"

    async def test_format_prompt(self):
        drop = DROP()
        drop._few_shot_examples = []
        item = {
            "passage": "The Battle of Gettysburg was fought in 1863.",
            "question": "When was the Battle of Gettysburg?",
        }
        prompt = drop.format_prompt(item, shots=0)
        assert "Battle of Gettysburg" in prompt

    async def test_score_exact_match(self):
        drop = DROP()
        item = {"answers_spans": {"spans": ["1863"], "number": "", "date": {}}}
        value, _ = drop.score_response(item, "1863")
        assert value == 1.0

    async def test_score_numeric_match(self):
        drop = DROP()
        item = {"answers_spans": {"spans": ["42"], "number": "", "date": {}}}
        value, _ = drop.score_response(item, "The answer is 42")
        assert value == 1.0

    async def test_score_f1_partial(self):
        drop = DROP()
        item = {"answers_spans": {"spans": ["the red car"], "number": "", "date": {}}}
        value, _ = drop.score_response(item, "red car")
        assert value > 0.5

    async def test_score_incorrect(self):
        drop = DROP()
        item = {"answers_spans": {"spans": ["Paris"], "number": "", "date": {}}}
        value, _ = drop.score_response(item, "London")
        assert value == 0.0

    async def test_run_with_mock(self, mock_llm_factory):
        drop = DROP()
        sample_data = [
            {
                "passage": "There were 5 goals.",
                "question": "How many goals?",
                "answers_spans": {"spans": ["5"], "number": "", "date": {}},
                "section_id": "s1",
            },
        ]

        with patch.object(drop, "load_dataset", new_callable=AsyncMock, return_value=sample_data):
            drop._few_shot_examples = []
            llm = mock_llm_factory(responses=["5"])
            result = await drop.run(llm, shots=0)

        assert result.accuracy == 1.0
        assert "f1" in result.metrics
        assert "exact_match" in result.metrics
        assert result.metrics["f1"] == 1.0
        assert result.metrics["exact_match"] == 1.0

    async def test_metrics_account_for_errors(self, mock_llm_factory):
        """Errors before score_response() should count as zeros in aggregates."""
        drop = DROP()
        sample_data = [
            {
                "passage": "P1",
                "question": "Q1",
                "answers_spans": {"spans": ["5"], "number": "", "date": {}},
                "section_id": "s1",
            },
            {
                "passage": "P2",
                "question": "Q2",
                "answers_spans": {"spans": ["10"], "number": "", "date": {}},
                "section_id": "s2",
            },
        ]

        # LLM returns correct for first, raises for second
        llm = mock_llm_factory(responses=["5", RuntimeError("model error")])
        with patch.object(drop, "load_dataset", new_callable=AsyncMock, return_value=sample_data):
            drop._few_shot_examples = []
            result = await drop.run(llm, shots=0)

        # 2 total items: one scored EM=1.0, one errored (should count as 0)
        assert result.num_total == 2
        assert result.metrics["exact_match"] == 0.5
        assert result.metrics["f1"] == 0.5
