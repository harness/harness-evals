"""Tests for LLM abstraction with mocked providers."""

import pytest

from harness_evals.llm.base import BaseLLM


class MockLLM(BaseLLM):
    """Mock LLM for testing. Returns configurable responses."""

    def __init__(self, text_response: str = "mock response", json_response: dict | None = None):
        self._text_response = text_response
        self._json_response = json_response or {}
        self.calls: list[str] = []

    async def generate(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        return self._text_response

    async def generate_json(self, prompt: str, schema: dict, **kwargs) -> dict:
        self.calls.append(prompt)
        return self._json_response


@pytest.mark.unit
class TestBaseLLM:
    @pytest.mark.asyncio
    async def test_mock_generate(self):
        llm = MockLLM(text_response="hello")
        result = await llm.generate("test prompt")
        assert result == "hello"
        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_mock_generate_json(self):
        llm = MockLLM(json_response={"score": 0.9, "reasoning": "good"})
        result = await llm.generate_json("test", {"type": "object"})
        assert result["score"] == 0.9

    def test_sync_wrapper(self):
        llm = MockLLM(text_response="sync result")
        result = llm.generate_sync("test")
        assert result == "sync result"

    def test_sync_json_wrapper(self):
        llm = MockLLM(json_response={"key": "value"})
        result = llm.generate_json_sync("test", {})
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_multiple_calls_tracked(self):
        llm = MockLLM()
        await llm.generate("prompt 1")
        await llm.generate("prompt 2")
        assert len(llm.calls) == 2
        assert "prompt 1" in llm.calls[0]
