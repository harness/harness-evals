"""Anthropic LLM provider."""

from __future__ import annotations

import json
import os
from typing import Any

from harness_evals.llm._schema import ANTHROPIC_UNSUPPORTED, make_strict_schema
from harness_evals.llm.base import BaseLLM
from harness_evals.llm.cost import estimate_llm_cost
from harness_evals.llm.usage import record_token_usage


def _record_anthropic_usage(response: Any, *, model: str) -> None:
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None) if usage is not None else None
    output_tokens = getattr(usage, "output_tokens", None) if usage is not None else None
    cost_usd = estimate_llm_cost(response, model=model)
    record_token_usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        model=model,
    )


class AnthropicLLM(BaseLLM):
    """Anthropic-backed LLM. Requires ``pip install harness-evals[llm]``.

    API key resolution: constructor ``api_key`` > ``ANTHROPIC_API_KEY`` env var.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
        *,
        base_url: str | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> None:
        try:
            import anthropic  # noqa: F811
        except ImportError as e:
            raise ImportError("Install anthropic: pip install harness-evals[llm]") from e

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError("No API key: pass api_key= or set ANTHROPIC_API_KEY")
        client_kwargs: dict[str, Any] = {"api_key": resolved_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = anthropic.AsyncAnthropic(**client_kwargs)

    def _optional_params(self) -> dict[str, Any]:
        # Send only params that are explicitly set; omit the rest so the model applies its
        # own defaults. Newer Claude models (e.g. Claude 4 extended-thinking on Bedrock) reject
        # sampling knobs like ``temperature`` outright, so unconditionally sending them 400s.
        params: dict[str, Any] = {}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.top_k is not None:
            params["top_k"] = self.top_k
        return params

    async def generate(self, prompt: str, **kwargs: object) -> str:
        optional_params = self._optional_params()
        system_prompt = kwargs.get("system_prompt")
        if system_prompt:
            optional_params["system"] = str(system_prompt)
        response = await self._client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            **optional_params,
        )
        _record_anthropic_usage(response, model=self.model)
        return response.content[0].text if response.content else ""

    async def generate_json(self, prompt: str, schema: dict, **kwargs: object) -> dict:
        strict_schema = make_strict_schema(schema, strip_keywords=ANTHROPIC_UNSUPPORTED)
        optional_params = self._optional_params()
        system_prompt = kwargs.get("system_prompt")
        if system_prompt:
            optional_params["system"] = str(system_prompt)
        response = await self._client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            **optional_params,
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": strict_schema,
                }
            },
        )
        _record_anthropic_usage(response, model=self.model)
        text = response.content[0].text if response.content else "{}"
        return json.loads(text)
