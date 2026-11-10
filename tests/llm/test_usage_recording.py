"""Token-usage recording: collector semantics and provider-level capture (no network)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness_evals.llm import collect_token_usage, record_token_usage
from harness_evals.llm.usage import TokenUsage

pytestmark = pytest.mark.unit


def test_record_outside_context_is_noop() -> None:
    record_token_usage(input_tokens=10, output_tokens=5, cost_usd=0.01, model="gpt-4")


def test_collect_accumulates() -> None:
    with collect_token_usage() as usage:
        record_token_usage(input_tokens=10, output_tokens=1)
        record_token_usage(input_tokens=5)
    assert isinstance(usage, TokenUsage)
    assert usage.input_tokens == 15
    assert usage.output_tokens == 1


def test_nesting_isolation() -> None:
    with collect_token_usage() as outer:
        with collect_token_usage() as inner:
            record_token_usage(input_tokens=7)
        assert inner.input_tokens == 7
        assert outer.input_tokens is None
        record_token_usage(input_tokens=2)
    assert outer.input_tokens == 2
    assert inner.input_tokens == 7


async def test_concurrent_task_isolation() -> None:
    async def worker(tokens: int, delay: float) -> TokenUsage:
        with collect_token_usage() as usage:
            await asyncio.sleep(delay)
            record_token_usage(input_tokens=tokens, output_tokens=tokens * 2)
            await asyncio.sleep(0.01)
        return usage

    a, b = await asyncio.gather(worker(3, 0.02), worker(100, 0.0))
    assert (a.input_tokens, a.output_tokens) == (3, 6)
    assert (b.input_tokens, b.output_tokens) == (100, 200)


def test_partial_usage_survives_exception() -> None:
    with collect_token_usage() as usage:
        try:
            record_token_usage(input_tokens=3)
            raise ValueError("boom")
        except ValueError:
            pass
    assert usage.input_tokens == 3

    with pytest.raises(ValueError), collect_token_usage() as raised:
        record_token_usage(input_tokens=3)
        raise ValueError("boom")
    assert raised.input_tokens == 3


async def test_openai_generate_records_usage() -> None:
    import openai

    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=7),
    )
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    with patch.object(openai, "AsyncOpenAI", return_value=client):
        from harness_evals.llm.openai import OpenAILLM

        llm = OpenAILLM(model="gpt-4o", api_key="test-key")
    with collect_token_usage() as usage:
        assert await llm.generate("hello") == "hi"
    assert usage.input_tokens == 12
    assert usage.output_tokens == 7
    assert "gpt-4o" in usage.by_model


async def test_anthropic_generate_records_usage() -> None:
    import anthropic

    response = SimpleNamespace(
        content=[SimpleNamespace(text="hi")],
        usage=SimpleNamespace(input_tokens=21, output_tokens=9),
    )
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)
    with patch.object(anthropic, "AsyncAnthropic", return_value=client):
        from harness_evals.llm.anthropic import AnthropicLLM

        llm = AnthropicLLM(model="claude-sonnet-5-5", api_key="test-key")
    with collect_token_usage() as usage:
        assert await llm.generate("hello") == "hi"
    assert usage.input_tokens == 21
    assert usage.output_tokens == 9
