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

        self.mock_create = MagicMock()

        async def fake_create(**kwargs):
            self.mock_create(**kwargs)
            choice = MagicMock()
            choice.message.content = '{"score": 1}' if "response_format" in kwargs else "ok"
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

    async def test_system_prompt_uses_chat_system_role(self):
        llm = self._make()
        await llm.generate("What is the return policy?", system_prompt="You are a policy assistant.")
        assert self.mock_create.call_args[1]["messages"] == [
            {"role": "system", "content": "You are a policy assistant."},
            {"role": "user", "content": "What is the return policy?"},
        ]

    async def test_generate_json_system_prompt_uses_chat_system_role(self):
        llm = self._make()
        await llm.generate_json(
            "Score this answer.",
            {"type": "object", "properties": {"score": {"type": "number"}}},
            system_prompt="You are a judge.",
        )
        assert self.mock_create.call_args[1]["messages"] == [
            {"role": "system", "content": "You are a judge."},
            {"role": "user", "content": "Score this answer."},
        ]

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
            block.text = '{"score": 1}' if "output_config" in kwargs else "ok"
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

    async def test_system_prompt_uses_native_system_param(self):
        llm = self._make()
        await llm.generate("What is the return policy?", system_prompt="You are a policy assistant.")
        kw = self.mock_create.call_args[1]
        assert kw["messages"] == [{"role": "user", "content": "What is the return policy?"}]
        assert kw["system"] == "You are a policy assistant."

    async def test_generate_json_system_prompt_uses_native_system_param(self):
        llm = self._make()
        await llm.generate_json(
            "Score this answer.",
            {"type": "object", "properties": {"score": {"type": "number"}}},
            system_prompt="You are a judge.",
        )
        kw = self.mock_create.call_args[1]
        assert kw["messages"] == [{"role": "user", "content": "Score this answer."}]
        assert kw["system"] == "You are a judge."

    def test_positional_arg_backward_compat(self):
        from harness_evals.llm.anthropic import AnthropicLLM

        llm = AnthropicLLM("claude-sonnet-4-20250514", "sk-ant-test", 0.5, 2048)
        assert llm.model == "claude-sonnet-4-20250514"
        assert llm.temperature == 0.5
        assert llm.max_tokens == 2048
        assert llm.top_p is None
        assert llm.top_k is None


@pytest.mark.unit
class TestBedrockAnthropicLLM:
    """BedrockAnthropicLLM routes to AsyncAnthropicBedrock with bearer auth + region,
    reusing AnthropicLLM's request logic."""

    @pytest.fixture(autouse=True)
    def _patch_anthropic(self, monkeypatch):
        self.mock_create = MagicMock()
        self.client_kwargs: dict = {}

        async def fake_create(**kwargs):
            self.mock_create(**kwargs)
            block = MagicMock()
            block.text = '{"ok": true}'
            resp = MagicMock()
            resp.content = [block]
            return resp

        def fake_bedrock_ctor(**kwargs):
            self.client_kwargs = kwargs
            client = MagicMock()
            client.messages.create = fake_create
            return client

        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropicBedrock.side_effect = fake_bedrock_ctor
        monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)
        # Bedrock must not depend on a direct-Anthropic key or ambient AWS env.
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)

    def _make(self, **kwargs):
        from harness_evals.llm.bedrock import BedrockAnthropicLLM

        kwargs.setdefault("api_key", "bedrock-bearer-token")
        kwargs.setdefault("aws_region", "us-east-2")
        return BedrockAnthropicLLM(model="global.anthropic.claude-sonnet-4-5-20250929-v1:0", **kwargs)

    def test_builds_bedrock_client_with_bearer_and_region(self):
        self._make()
        assert self.client_kwargs.get("api_key") == "bedrock-bearer-token"
        assert self.client_kwargs.get("aws_region") == "us-east-2"

    def test_requires_bearer_key(self):
        # Bearer-only: no api_key and no AWS_BEARER_TOKEN_BEDROCK must raise, rather than let the
        # SDK fall through to an AWS IAM/SigV4 path (which needs boto3) and fail confusingly.
        from harness_evals.llm.bedrock import BedrockAnthropicLLM

        with pytest.raises(ValueError, match="No Bedrock API key"):
            BedrockAnthropicLLM(model="m", api_key=None)

    def test_does_not_require_anthropic_api_key(self):
        # A bearer is sufficient; ANTHROPIC_API_KEY is neither needed nor used.
        self._make()  # passes a bearer via _make's default api_key
        assert self.client_kwargs.get("api_key") == "bedrock-bearer-token"

    async def test_generate_reuses_parent_path(self):
        llm = self._make()
        out = await llm.generate("hi")
        assert out == '{"ok": true}'
        assert self.mock_create.call_args[1]["model"] == "global.anthropic.claude-sonnet-4-5-20250929-v1:0"

    async def test_generate_json_uses_output_config(self):
        llm = self._make()
        result = await llm.generate_json("hi", {"type": "object", "properties": {"ok": {"type": "boolean"}}})
        assert result == {"ok": True}
        assert "output_config" in self.mock_create.call_args[1]

    async def test_sampling_params_forwarded(self):
        llm = self._make(top_p=0.9, top_k=40)
        await llm.generate("hi")
        kw = self.mock_create.call_args[1]
        assert kw["top_p"] == 0.9
        assert kw["top_k"] == 40

    def test_env_fallback_for_bearer_and_region(self, monkeypatch):
        # With no constructor api_key/aws_region, the SDK client gets the values from env.
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "env-bearer")
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        from harness_evals.llm.bedrock import BedrockAnthropicLLM

        BedrockAnthropicLLM(model="m")
        assert self.client_kwargs.get("api_key") == "env-bearer"
        assert self.client_kwargs.get("aws_region") == "eu-west-1"

    def test_constructor_args_override_env(self, monkeypatch):
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "env-bearer")
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        from harness_evals.llm.bedrock import BedrockAnthropicLLM

        BedrockAnthropicLLM(model="m", api_key="ctor-bearer", aws_region="us-east-2")
        assert self.client_kwargs.get("api_key") == "ctor-bearer"
        assert self.client_kwargs.get("aws_region") == "us-east-2"


@pytest.mark.unit
class TestBedrockOpenAILLM:
    """BedrockOpenAILLM points the OpenAI SDK at the Bedrock OpenAI-compatible endpoint
    and extracts JSON robustly (Bedrock doesn't enforce json_schema)."""

    @pytest.fixture(autouse=True)
    def _patch_openai(self, monkeypatch):
        self.mock_create = MagicMock()
        self.client_kwargs: dict = {}
        # content is set per-test via self.content
        self.content = "pong"

        async def fake_create(**kwargs):
            self.mock_create(**kwargs)
            msg = MagicMock()
            msg.content = self.content
            choice = MagicMock()
            choice.message = msg
            resp = MagicMock()
            resp.choices = [choice]
            resp.usage = None
            return resp

        def fake_ctor(**kwargs):
            self.client_kwargs = kwargs
            client = MagicMock()
            client.chat.completions.create = fake_create
            return client

        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI.side_effect = fake_ctor
        monkeypatch.setitem(sys.modules, "openai", mock_openai)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)

    def _make(self, **kwargs):
        from harness_evals.llm.bedrock import BedrockOpenAILLM

        kwargs.setdefault("api_key", "bedrock-bearer-token")
        kwargs.setdefault("aws_region", "us-east-1")
        return BedrockOpenAILLM(model="arn:aws:bedrock:us-east-1:1:application-inference-profile/x", **kwargs)

    def test_points_at_bedrock_openai_endpoint_with_bearer(self):
        self._make(aws_region="us-west-2")
        assert self.client_kwargs.get("api_key") == "bedrock-bearer-token"
        assert self.client_kwargs.get("base_url") == "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1"

    async def test_generate_json_extracts_reasoning_wrapped_json(self):
        # gpt-oss on Bedrock emits reasoning + non-strict JSON; extractor must recover it.
        self.content = '<reasoning>The answer is correct.</reasoning>**{ "score": 1.0, "reason": "correct" }'
        llm = self._make()
        result = await llm.generate_json("score it", {"type": "object", "properties": {"score": {"type": "number"}}})
        assert result == {"score": 1.0, "reason": "correct"}
        # No strict json_schema response_format is relied upon.
        assert "response_format" not in self.mock_create.call_args[1]

    async def test_generate_json_plain_json(self):
        self.content = '{"score": 0.5, "reason": "partial"}'
        llm = self._make()
        assert await llm.generate_json("x", {"type": "object"}) == {"score": 0.5, "reason": "partial"}

    def test_requires_bearer_key(self):
        # No api_key and no AWS_BEARER_TOKEN_BEDROCK: must raise rather than let the OpenAI SDK
        # silently fall back to OPENAI_API_KEY (which would send a direct-OpenAI key to Bedrock).
        from harness_evals.llm.bedrock import BedrockOpenAILLM

        with pytest.raises(ValueError, match="No Bedrock API key"):
            BedrockOpenAILLM(model="arn:aws:bedrock:us-east-1:1:application-inference-profile/x", api_key=None)

    def test_env_fallback_for_bearer_and_region(self, monkeypatch):
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "env-bearer")
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        from harness_evals.llm.bedrock import BedrockOpenAILLM

        BedrockOpenAILLM(model="m")
        assert self.client_kwargs.get("api_key") == "env-bearer"
        assert self.client_kwargs.get("base_url") == "https://bedrock-runtime.eu-west-1.amazonaws.com/openai/v1"

    def test_constructor_args_override_env(self, monkeypatch):
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "env-bearer")
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        from harness_evals.llm.bedrock import BedrockOpenAILLM

        BedrockOpenAILLM(model="m", api_key="ctor-bearer", aws_region="us-east-2")
        assert self.client_kwargs.get("api_key") == "ctor-bearer"
        assert self.client_kwargs.get("base_url") == "https://bedrock-runtime.us-east-2.amazonaws.com/openai/v1"


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
