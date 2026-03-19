"""Tests for LLM abstraction with mocked providers."""

import pytest

from harness_evals._async_compat import _run_async
from harness_evals.llm._schema import make_strict_schema
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
    async def test_mock_generate(self):
        llm = MockLLM(text_response="hello")
        result = await llm.generate("test prompt")
        assert result == "hello"
        assert len(llm.calls) == 1

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

    async def test_sync_wrapper_inside_async_context(self):
        """generate_sync() must not crash when an event loop is already running."""
        llm = MockLLM(text_response="from async")
        result = llm.generate_sync("test")
        assert result == "from async"

    async def test_sync_json_wrapper_inside_async_context(self):
        """generate_json_sync() must not crash when an event loop is already running."""
        llm = MockLLM(json_response={"ok": True})
        result = llm.generate_json_sync("test", {})
        assert result == {"ok": True}

    async def test_multiple_calls_tracked(self):
        llm = MockLLM()
        await llm.generate("prompt 1")
        await llm.generate("prompt 2")
        assert len(llm.calls) == 2
        assert "prompt 1" in llm.calls[0]


@pytest.mark.unit
class TestRunAsync:
    def test_run_async_no_loop(self):
        """_run_async works from a plain sync context."""

        async def coro():
            return 42

        assert _run_async(coro()) == 42

    async def test_run_async_inside_loop(self):
        """_run_async works when an event loop is already running."""

        async def coro():
            return "hello from nested"

        result = _run_async(coro())
        assert result == "hello from nested"

    async def test_run_async_propagates_exceptions(self):
        async def failing():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            _run_async(failing())


@pytest.mark.unit
class TestMakeStrictSchema:
    def test_adds_additional_properties_false(self):
        schema = {
            "type": "object",
            "required": ["score"],
            "properties": {"score": {"type": "number"}},
        }
        result = make_strict_schema(schema)
        assert result["additionalProperties"] is False

    def test_populates_required_from_properties(self):
        schema = {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "score": {"type": "number"},
            },
        }
        result = make_strict_schema(schema)
        assert set(result["required"]) == {"reasoning", "score"}

    def test_preserves_existing_required(self):
        schema = {
            "type": "object",
            "required": ["score"],
            "properties": {
                "score": {"type": "number"},
                "reasoning": {"type": "string"},
            },
        }
        result = make_strict_schema(schema)
        assert result["required"] == ["score"]

    def test_strips_unsupported_keywords(self):
        schema = {
            "type": "object",
            "required": ["score"],
            "properties": {
                "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
        }
        result = make_strict_schema(schema)
        assert "minimum" not in result["properties"]["score"]
        assert "maximum" not in result["properties"]["score"]
        assert result["properties"]["score"]["type"] == "number"

    def test_handles_nested_objects_in_arrays(self):
        schema = {
            "type": "object",
            "required": ["verdicts"],
            "properties": {
                "verdicts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {"type": "string"},
                            "verdict": {"type": "string"},
                        },
                    },
                }
            },
        }
        result = make_strict_schema(schema)
        items = result["properties"]["verdicts"]["items"]
        assert items["additionalProperties"] is False
        assert set(items["required"]) == {"claim", "verdict"}

    def test_does_not_mutate_original(self):
        schema = {
            "type": "object",
            "properties": {"score": {"type": "number", "minimum": 0.0}},
        }
        make_strict_schema(schema)
        assert "minimum" in schema["properties"]["score"]
        assert "additionalProperties" not in schema
