"""Dynamic LLM completion cost estimation for judge spend reporting.

Pricing is resolved at runtime — never hard-coded per model. Resolution order:

1. Cost fields embedded in the provider response (when the gateway or provider returns them).
2. ``litellm.completion_cost()`` when ``litellm`` is installed (same helper the Harness
   LLM gateway uses internally).
3. Token counts from the response usage object × LiteLLM ``model_cost`` rates when (2) fails
   but the model is in LiteLLM's pricing table (common for gateway-routed OpenAI responses).

Returns ``None`` when cost cannot be determined (unknown model, litellm missing, zero tokens).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def normalize_model_for_pricing(model: str) -> str:
    """Normalize model strings for LiteLLM's model-pricing database."""
    normalized = model.strip()
    if "@" in normalized:
        normalized = normalized.split("@", 1)[0]
    return normalized


def _coerce_cost(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        cost = float(value)
    except (TypeError, ValueError):
        return None
    return cost if cost >= 0 else None


def _extract_cost_from_response(response: Any) -> float | None:
    """Best-effort read of cost already computed by the upstream provider/gateway."""
    usage = getattr(response, "usage", None)
    if usage is not None:
        for attr in ("total_cost", "cost", "cost_usd"):
            cost = _coerce_cost(getattr(usage, attr, None))
            if cost is not None:
                return cost

    for attr in ("cost", "cost_usd", "total_cost"):
        cost = _coerce_cost(getattr(response, attr, None))
        if cost is not None:
            return cost

    if isinstance(response, dict):
        for key in ("cost", "cost_usd", "total_cost"):
            cost = _coerce_cost(response.get(key))
            if cost is not None:
                return cost
        usage_dict = response.get("usage")
        if isinstance(usage_dict, dict):
            for key in ("total_cost", "cost", "cost_usd"):
                cost = _coerce_cost(usage_dict.get(key))
                if cost is not None:
                    return cost
    return None


def _usage_token_counts(response: Any) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return None, None
    if isinstance(usage, dict):
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    else:
        input_tokens = getattr(usage, "prompt_tokens", None)
        if input_tokens is None:
            input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        if output_tokens is None:
            output_tokens = getattr(usage, "output_tokens", None)
    try:
        in_t = int(input_tokens) if input_tokens is not None else None
    except (TypeError, ValueError):
        in_t = None
    try:
        out_t = int(output_tokens) if output_tokens is not None else None
    except (TypeError, ValueError):
        out_t = None
    return in_t, out_t


def _model_cost_candidates(model: str) -> list[str]:
    normalized = normalize_model_for_pricing(model)
    candidates = [normalized]
    if "/" in normalized:
        candidates.append(normalized.split("/", 1)[1])
    if not normalized.startswith("gpt-"):
        tail = normalized.rsplit("/", 1)[-1]
        if tail not in candidates:
            candidates.append(tail)
    return candidates


def _litellm_token_based_cost(
    model: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    if not model or not input_tokens:
        return None
    try:
        import litellm
    except ImportError:
        return None

    out_tokens = output_tokens or 0
    for candidate in _model_cost_candidates(model):
        info = litellm.model_cost.get(candidate)
        if not info:
            continue
        in_rate = info.get("input_cost_per_token")
        out_rate = info.get("output_cost_per_token")
        if in_rate is None and out_rate is None:
            continue
        cost = float(in_rate or 0.0) * input_tokens + float(out_rate or 0.0) * out_tokens
        return cost if cost > 0 else None
    return None


def _litellm_completion_cost(response: Any, model: str | None) -> float | None:
    try:
        import litellm
    except ImportError:
        return None

    kwargs: dict[str, Any] = {"completion_response": response}
    if model:
        kwargs["model"] = normalize_model_for_pricing(model)
    try:
        return float(litellm.completion_cost(**kwargs) or 0.0)
    except Exception:
        logger.debug(
            "LiteLLM completion_cost failed for model=%r",
            model,
            exc_info=True,
        )
        return None


def estimate_llm_cost(response: Any, *, model: str | None = None) -> float | None:
    """Return USD cost for one completion, or ``None`` when it cannot be determined."""
    embedded = _extract_cost_from_response(response)
    if embedded is not None and embedded > 0:
        return embedded

    resolved_model = model or getattr(response, "model", None)
    resolved_model = normalize_model_for_pricing(resolved_model) if isinstance(resolved_model, str) else None

    litellm_cost = _litellm_completion_cost(response, resolved_model)
    if litellm_cost is not None and litellm_cost > 0:
        return litellm_cost
    if litellm_cost == 0.0 and embedded == 0.0:
        return 0.0

    if resolved_model:
        input_tokens, output_tokens = _usage_token_counts(response)
        token_cost = _litellm_token_based_cost(resolved_model, input_tokens, output_tokens)
        if token_cost is not None and token_cost > 0:
            return token_cost
    return None
