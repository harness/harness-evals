"""Tests for LLM abstraction with mocked providers."""

import sys
from unittest.mock import MagicMock

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
class TestOpenAILLMParams:
    """Verify OpenAILLM forwards optional sampling params to the SDK."""

    @pytest.fixture(autouse=True)
    def _patch_openai(self, monkeypatch):
        import types

        self.mock_create = MagicMock()

        async def fake_create(**kwargs):
            self.mock_create(**kwargs)
            choice = MagicMock()
            choice.message.content = "ok"
            resp = MagicMock()
            resp.choices = [choice]
            return resp

        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_client.chat.completions.create = fake_create
        mock_openai.AsyncOpenAI.return_value = mock_client
        monkeypatch.setitem(sys.modules, "openai", mock_openai)

    def _make(self, **kwargs):
        from harness_evals.llm.openai import OpenAILLM

        return OpenAILLM(model="gpt-4o", api_key="sk-test", **kwargs)

    async def test_no_optional_params_by_default(self):
        llm = self._make()
        await llm.generate("hi")
        call_kwargs = self.mock_create.call_args[1]
        assert "top_p" not in call_kwargs
        assert "frequency_penalty" not in call_kwargs
        assert "presence_penalty" not in call_kwargs

    async def test_top_p_forwarded(self):
        llm = self._make(top_p=0.9)
        await llm.generate("hi")
        assert self.mock_create.call_args[1]["top_p"] == 0.9

    async def test_frequency_penalty_forwarded(self):
        llm = self._make(frequency_penalty=0.5)
        await llm.generate("hi")
        assert self.mock_create.call_args[1]["frequency_penalty"] == 0.5

    async def test_presence_penalty_forwarded(self):
        llm = self._make(presence_penalty=0.3)
        await llm.generate("hi")
        assert self.mock_create.call_args[1]["presence_penalty"] == 0.3

    async def test_all_optional_params(self):
        llm = self._make(top_p=0.95, frequency_penalty=0.4, presence_penalty=0.2)
        await llm.generate("hi")
        kw = self.mock_create.call_args[1]
        assert kw["top_p"] == 0.95
        assert kw["frequency_penalty"] == 0.4
        assert kw["presence_penalty"] == 0.2

    def test_positional_arg_backward_compat(self):
        """Existing callers using positional args still work."""
        from harness_evals.llm.openai import OpenAILLM

        llm = OpenAILLM("gpt-4o", "sk-test", 0.5, 2048)
        assert llm.model == "gpt-4o"
        assert llm.temperature == 0.5
        assert llm.max_tokens == 2048
        assert llm.top_p is None


@pytest.mark.unit
class TestAnthropicLLMParams:
    """Verify AnthropicLLM forwards optional sampling params to the SDK."""

    @pytest.fixture(autouse=True)
    def _patch_anthropic(self, monkeypatch):
        self.mock_create = MagicMock()

        async def fake_create(**kwargs):
            self.mock_create(**kwargs)
            block = MagicMock()
            block.text = "ok"
            resp = MagicMock()
            resp.content = [block]
            return resp

        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_client.messages.create = fake_create
        mock_anthropic.AsyncAnthropic.return_value = mock_client
        monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)

    def _make(self, **kwargs):
        from harness_evals.llm.anthropic import AnthropicLLM

        return AnthropicLLM(model="claude-sonnet-4-20250514", api_key="sk-ant-test", **kwargs)

    async def test_no_optional_params_by_default(self):
        llm = self._make()
        await llm.generate("hi")
        call_kwargs = self.mock_create.call_args[1]
        assert "top_p" not in call_kwargs
        assert "top_k" not in call_kwargs

    async def test_top_p_forwarded(self):
        llm = self._make(top_p=0.9)
        await llm.generate("hi")
        assert self.mock_create.call_args[1]["top_p"] == 0.9

    async def test_top_k_forwarded(self):
        llm = self._make(top_k=40)
        await llm.generate("hi")
        assert self.mock_create.call_args[1]["top_k"] == 40

    async def test_both_optional_params(self):
        llm = self._make(top_p=0.95, top_k=50)
        await llm.generate("hi")
        kw = self.mock_create.call_args[1]
        assert kw["top_p"] == 0.95
        assert kw["top_k"] == 50

    def test_positional_arg_backward_compat(self):
        from harness_evals.llm.anthropic import AnthropicLLM

        llm = AnthropicLLM("claude-sonnet-4-20250514", "sk-ant-test", 0.5, 2048)
        assert llm.model == "claude-sonnet-4-20250514"
        assert llm.temperature == 0.5
        assert llm.max_tokens == 2048
        assert llm.top_p is None
        assert llm.top_k is None


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
