"""Anthropic LLM provider."""

from __future__ import annotations

import json
import os

from harness_evals.llm.base import BaseLLM


class AnthropicLLM(BaseLLM):
    """Anthropic-backed LLM. Requires ``pip install harness-evals[llm]``.

    API key resolution: constructor ``api_key`` > ``ANTHROPIC_API_KEY`` env var.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        try:
            import anthropic  # noqa: F811
        except ImportError as e:
            raise ImportError("Install anthropic: pip install harness-evals[llm]") from e

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError("No API key: pass api_key= or set ANTHROPIC_API_KEY")
        self._client = anthropic.AsyncAnthropic(api_key=resolved_key)

    async def generate(self, prompt: str, **kwargs: object) -> str:
        response = await self._client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.content[0].text if response.content else ""

    async def generate_json(self, prompt: str, schema: dict, **kwargs: object) -> dict:
        json_prompt = f"{prompt}\n\nRespond ONLY with valid JSON matching this schema:\n{json.dumps(schema)}"
        response = await self._client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": json_prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        text = response.content[0].text if response.content else "{}"
        return json.loads(text)
