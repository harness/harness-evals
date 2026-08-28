"""Harness LLM gateway provider (OpenAI-compatible ``/llm-gw/v1``).

Routes chat completions and embeddings through the Harness-managed LLM gateway
using PAT/SAT/scoped tokens via ``x-api-key``. The gateway validates
``Authorization`` as a service JWT and does not fall through to ``x-api-key``
when that header is present, so the OpenAI SDK's default Bearer header is stripped.

This is distinct from :class:`~harness_evals.llm.harness_ai.HarnessAILLM`, which
calls the Harness AI Service ``/chat`` endpoint with HS256 JWT auth.

Environment variables:
    HARNESS_TOKEN: PAT with ``llmGatewayAccess`` (fallback: ``LLM_GATEWAY_API_KEY``)
    LLM_GATEWAY_BASE_URL: Gateway URL (must include ``/v1``; fallback from ``HARNESS_BASE_URL``)
    LLM_GATEWAY_X_SOURCE: Budget identity header (default: ``harness-evals``)
    LLM_GATEWAY_X_SOURCE_CLASS: Budget class header (default: ``LocalDev``)
"""

from __future__ import annotations

import os
from typing import Any

from harness_evals.llm.cost import normalize_model_for_pricing
from harness_evals.llm.openai import OpenAILLM
from harness_evals.llm.openai_embedding import OpenAIEmbedding


def normalize_harness_gateway_model_for_pricing(model: str) -> str:
    """Strip Harness LLM gateway routing aliases before LiteLLM pricing lookup."""
    normalized = model.strip()
    if normalized.startswith("online/"):
        normalized = normalized[len("online/") :]
    return normalize_model_for_pricing(normalized)


def normalize_gateway_routing_model(provider: str, model: str) -> str:
    """Prefix bare vendor model IDs for Harness LLM gateway LiteLLM routing.

    The gateway ``/v1/chat/completions`` surface accepts OpenAI-shaped requests and
    routes to Anthropic (and other vendors) via ``online/<vendor>/...`` model aliases.
    """
    normalized = model.strip()
    if normalized.startswith("online/"):
        return normalized
    if provider == "anthropic":
        return f"online/anthropic/{normalized}"
    return normalized


def resolve_gateway_base_url(gateway_path: str = "/llm-gw/v1") -> str | None:
    """Build gateway base URL from ``HARNESS_BASE_URL`` when not pre-resolved."""
    harness_base = os.environ.get("HARNESS_BASE_URL", "").rstrip("/")
    for suffix in ("/ng", "/gateway"):
        if harness_base.endswith(suffix):
            harness_base = harness_base[: -len(suffix)]
            break
    return f"{harness_base}{gateway_path}" if harness_base else None


def _make_x_api_key_http_client(
    api_key: str,
    *,
    extra_headers: dict[str, str] | None = None,
) -> Any:
    """httpx client that sends the token via ``x-api-key`` (not ``Authorization: Bearer``)."""
    import httpx

    budget_headers = dict(extra_headers or {})

    async def _rewrite_auth(request: httpx.Request) -> None:
        request.headers.pop("authorization", None)
        request.headers["x-api-key"] = api_key
        for key, value in budget_headers.items():
            request.headers[key] = value

    return httpx.AsyncClient(
        event_hooks={"request": [_rewrite_auth]},
        timeout=httpx.Timeout(600.0, connect=5.0),
    )


def _gateway_budget_headers(
    x_source: str | None,
    x_source_class: str | None,
) -> dict[str, str]:
    """Headers required by the Harness LLM gateway for PAT budget identity."""
    source = (x_source or os.environ.get("LLM_GATEWAY_X_SOURCE") or "harness-evals").strip()
    source_class = (x_source_class or os.environ.get("LLM_GATEWAY_X_SOURCE_CLASS") or "LocalDev").strip()
    return {"x-source": source, "x-source-class": source_class}


def _resolve_gateway_api_key(api_key: str | None) -> str:
    resolved = (api_key or gateway_api_key_from_env() or "").strip()
    if not resolved:
        raise ValueError("No gateway API key: pass api_key= or set HARNESS_TOKEN or LLM_GATEWAY_API_KEY")
    return resolved


def gateway_api_key_from_env() -> str | None:
    """Return the configured gateway PAT using the shared precedence order.

    ``HARNESS_TOKEN`` wins over ``LLM_GATEWAY_API_KEY``. Whitespace-only values
    are treated as unset so callers can fail closed.
    """
    token = (os.environ.get("HARNESS_TOKEN") or "").strip()
    if token:
        return token
    key = (os.environ.get("LLM_GATEWAY_API_KEY") or "").strip()
    return key or None


def _resolve_gateway_base_url(base_url: str | None, *, gateway_path: str = "/llm-gw/v1") -> str:
    resolved = (
        base_url or os.environ.get("LLM_GATEWAY_BASE_URL") or resolve_gateway_base_url(gateway_path) or ""
    ).strip()
    if not resolved:
        raise ValueError("No gateway base URL: pass base_url= or set LLM_GATEWAY_BASE_URL or HARNESS_BASE_URL")
    return resolved.rstrip("/")


def _build_gateway_client_kwargs(
    api_key: str,
    base_url: str,
    *,
    x_source: str | None = None,
    x_source_class: str | None = None,
) -> dict[str, Any]:
    return {
        "api_key": "unused",
        "base_url": base_url,
        "http_client": _make_x_api_key_http_client(
            api_key,
            extra_headers=_gateway_budget_headers(x_source, x_source_class),
        ),
    }


class HarnessGatewayOpenAILLM(OpenAILLM):
    """OpenAI-compatible Harness LLM gateway. Requires ``pip install harness-evals[llm]``.

    Reuses :class:`OpenAILLM` for chat completion requests and token-usage recording;
    overrides only client construction (PAT via ``x-api-key`` + budget headers).
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
        *,
        base_url: str | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        x_source: str | None = None,
        x_source_class: str | None = None,
        gateway_path: str = "/llm-gw/v1",
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

        resolved_key = _resolve_gateway_api_key(api_key)
        resolved_base_url = _resolve_gateway_base_url(base_url, gateway_path=gateway_path)
        client_kwargs = _build_gateway_client_kwargs(
            resolved_key,
            resolved_base_url,
            x_source=x_source,
            x_source_class=x_source_class,
        )
        self._client = openai.AsyncOpenAI(**client_kwargs)

    def _pricing_model(self) -> str:
        return normalize_harness_gateway_model_for_pricing(self.model)


class HarnessGatewayOpenAIEmbedding(OpenAIEmbedding):
    """Embeddings routed through the Harness LLM gateway (PAT via ``x-api-key``)."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        x_source: str | None = None,
        x_source_class: str | None = None,
        gateway_path: str = "/llm-gw/v1",
    ) -> None:
        try:
            import openai  # noqa: F811
        except ImportError as e:
            raise ImportError("Install openai: pip install harness-evals[llm]") from e

        self.model = model
        resolved_key = _resolve_gateway_api_key(api_key)
        resolved_base_url = _resolve_gateway_base_url(base_url, gateway_path=gateway_path)
        client_kwargs = _build_gateway_client_kwargs(
            resolved_key,
            resolved_base_url,
            x_source=x_source,
            x_source_class=x_source_class,
        )
        self._client = openai.AsyncOpenAI(**client_kwargs)
