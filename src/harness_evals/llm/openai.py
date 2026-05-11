"""OpenAI LLM provider."""

from __future__ import annotations

import json
import os
from typing import Any

from harness_evals.llm._schema import make_strict_schema
from harness_evals.llm.base import BaseLLM


class OpenAILLM(BaseLLM):
    """OpenAI-backed LLM. Requires ``pip install harness-evals[llm]``.

    API key resolution: constructor ``api_key`` > ``OPENAI_API_KEY`` env var.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        *,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
    ) -> None:
        try:
            import openai  # noqa: F811
        except ImportError as e:
            raise ImportError("Install openai: pip install harness-evals[llm]") from e

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("No API key: pass api_key= or set OPENAI_API_KEY")
        self._client = openai.AsyncOpenAI(api_key=resolved_key)

    def _optional_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
        return params

    async def generate(self, prompt: str, **kwargs: object) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
            **self._optional_params(),
        )
        return response.choices[0].message.content or ""

    async def generate_json(self, prompt: str, schema: dict, **kwargs: object) -> dict:
        strict_schema = make_strict_schema(schema)
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
            **self._optional_params(),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "eval_response",
                    "strict": True,
                    "schema": strict_schema,
                },
            },
        )
        text = response.choices[0].message.content or "{}"
        return json.loads(text)
