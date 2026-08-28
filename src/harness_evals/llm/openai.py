"""OpenAI LLM provider."""

from __future__ import annotations

import json
import os
from typing import Any

from harness_evals.llm._schema import make_strict_schema
from harness_evals.llm.base import BaseLLM
from harness_evals.llm.cost import estimate_llm_cost
from harness_evals.llm.usage import record_token_usage


def _record_openai_usage(
    response: Any,
    *,
    model: str,
    pricing_model: str | None = None,
) -> None:
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
    output_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
    cost_usd = estimate_llm_cost(response, model=pricing_model or model)
    record_token_usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        model=model,
    )


class OpenAILLM(BaseLLM):
    """OpenAI-backed LLM. Requires ``pip install harness-evals[llm]``.

    API key resolution: constructor ``api_key`` > ``OPENAI_API_KEY`` env var.

    For the Harness LLM gateway (PAT via ``x-api-key``), use
    :class:`~harness_evals.llm.harness_gateway.HarnessGatewayOpenAILLM` instead.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
        *,
        base_url: str | None = None,
        organization: str | None = None,
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
        client_kwargs: dict[str, Any] = {"api_key": resolved_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if organization:
            client_kwargs["organization"] = organization
        self._client = openai.AsyncOpenAI(**client_kwargs)

    def _pricing_model(self) -> str:
        """Model name passed to cost estimation (may differ from routing alias)."""
        return self.model

    def _optional_params(self) -> dict[str, Any]:
        # Send only params that are explicitly set; omit the rest so the model applies its own
        # defaults. Some newer models reject/deprecate sampling knobs like ``temperature``.
        params: dict[str, Any] = {}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
        return params

    def _messages(self, prompt: str, system_prompt: object | None = None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def generate(self, prompt: str, **kwargs: object) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=self._messages(prompt, kwargs.get("system_prompt")),
            max_completion_tokens=self.max_tokens,
            **self._optional_params(),
        )
        _record_openai_usage(
            response,
            model=self.model,
            pricing_model=self._pricing_model(),
        )
        return response.choices[0].message.content or ""

    async def generate_json(self, prompt: str, schema: dict, **kwargs: object) -> dict:
        strict_schema = make_strict_schema(schema)
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=self._messages(prompt, kwargs.get("system_prompt")),
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
        _record_openai_usage(
            response,
            model=self.model,
            pricing_model=self._pricing_model(),
        )
        text = response.choices[0].message.content or "{}"
        return json.loads(text)
