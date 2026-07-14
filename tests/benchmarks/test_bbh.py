"""Tests for BBH benchmark."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from harness_evals.benchmarks.bbh import BBH, _extract_bbh_answer


@pytest.mark.unit
class TestBBH:
    def test_invalid_tasks(self):
        with pytest.raises(ValueError, match="Unknown BBH tasks"):
            BBH(tasks=["fake_task"])

    def test_default_all_tasks(self):
        bbh = BBH()
        assert len(bbh.tasks) == 23

    def test_extract_bbh_answer(self):
        assert _extract_bbh_answer("The answer is True") == "True"
        assert _extract_bbh_answer("The answer is (A)") == "(A)"
        assert _extract_bbh_answer("Let me think...\nThe answer is False.") == "False"

    def test_extract_bbh_answer_fallback(self):
        result = _extract_bbh_answer("Just some text\nFinal line")
        assert result == "Final line"

    async def test_format_prompt(self):
        bbh = BBH(tasks=["boolean_expressions"])
        bbh._few_shot_examples = {"boolean_expressions": []}
        item = {
            "input": "not ( True ) and ( True ) is",
            "target": "False",
            "task": "boolean_expressions",
        }
        prompt = bbh.format_prompt(item, shots=0)
        assert "not ( True )" in prompt
        assert "Boolean Expressions" in prompt

    async def test_score_correct(self):
        bbh = BBH()
        item = {"target": "False"}
        value, _ = bbh.score_response(item, "The answer is False")
        assert value == 1.0

    async def test_score_correct_substring(self):
        bbh = BBH()
        item = {"target": "True"}
        value, _ = bbh.score_response(item, "The answer is True.")
        assert value == 1.0

    async def test_score_incorrect(self):
        bbh = BBH()
        item = {"target": "False"}
        value, _ = bbh.score_response(item, "The answer is True")
        assert value == 0.0

    async def test_run_with_mock(self, mock_llm_factory):
        bbh = BBH(tasks=["boolean_expressions"])
        sample_data = [
            {"input": "not True is", "target": "False", "task": "boolean_expressions"},
            {"input": "True and True is", "target": "True", "task": "boolean_expressions"},
        ]

        with patch.object(bbh, "load_dataset", new_callable=AsyncMock, return_value=sample_data):
            bbh._few_shot_examples = {"boolean_expressions": []}
            llm = mock_llm_factory(responses=["The answer is False", "The answer is True"])
            result = await bbh.run(llm, shots=0)

        assert result.accuracy == 1.0
        assert result.num_correct == 2
