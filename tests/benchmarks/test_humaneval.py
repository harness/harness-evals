"""Tests for HumanEval benchmark."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from harness_evals.benchmarks.humaneval import HumanEval, _extract_code


@pytest.mark.unit
class TestHumanEval:
    def test_extract_code_plain(self):
        response = "    return a + b"
        result = _extract_code(response, "add")
        assert "return a + b" in result

    def test_extract_code_markdown_block(self):
        response = "```python\n    return a + b\n```"
        result = _extract_code(response, "add")
        assert "return a + b" in result

    def test_extract_code_with_signature(self):
        response = "def add(a, b):\n    return a + b"
        result = _extract_code(response, "add")
        assert "return a + b" in result

    async def test_format_prompt(self):
        he = HumanEval()
        item = {
            "prompt": 'def add(a: int, b: int) -> int:\n    """Add two numbers."""\n',
            "test": "",
            "entry_point": "add",
        }
        prompt = he.format_prompt(item)
        assert "def add(a: int, b: int)" in prompt
        assert "Complete" in prompt

    async def test_score_response_correct(self):
        he = HumanEval(timeout=5.0)
        item = {
            "prompt": 'def add(a: int, b: int) -> int:\n    """Add two numbers."""\n',
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(0, 0) == 0\n",
            "entry_point": "add",
        }
        value, reason = he.score_response(item, "    return a + b")
        assert value == 1.0

    async def test_score_response_incorrect(self):
        he = HumanEval(timeout=5.0)
        item = {
            "prompt": 'def add(a: int, b: int) -> int:\n    """Add two numbers."""\n',
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
            "entry_point": "add",
        }
        value, reason = he.score_response(item, "    return a - b")
        assert value == 0.0

    async def test_run_with_mock(self, mock_llm_factory):
        he = HumanEval(timeout=5.0)
        sample_data = [
            {
                "prompt": 'def add(a: int, b: int) -> int:\n    """Add two numbers."""\n',
                "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
                "entry_point": "add",
                "task_id": "HumanEval/0",
                "canonical_solution": "    return a + b",
            },
        ]

        with patch.object(he, "load_dataset", new_callable=AsyncMock, return_value=sample_data):
            llm = mock_llm_factory(responses=["    return a + b"])
            result = await he.run(llm)

        assert result.num_total == 1
        assert result.num_correct == 1
