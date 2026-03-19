"""Shared JSON Schema adaptation for structured output providers."""

from __future__ import annotations

import copy
from typing import Any

OPENAI_UNSUPPORTED = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "pattern",
        "multipleOf",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
    }
)

ANTHROPIC_UNSUPPORTED = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "pattern",
        "multipleOf",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
    }
)


def make_strict_schema(
    schema: dict[str, Any],
    *,
    strip_keywords: frozenset[str] = OPENAI_UNSUPPORTED,
) -> dict[str, Any]:
    """Adapt a JSON Schema dict for structured output APIs.

    Both OpenAI and Anthropic require ``additionalProperties: false`` on every
    object and all properties listed in ``required``.  They also reject
    validation-only keywords, though the exact sets differ slightly.

    Pass ``strip_keywords=ANTHROPIC_UNSUPPORTED`` for Anthropic schemas.

    Returns a deep copy; the original schema is never mutated.
    """
    schema = copy.deepcopy(schema)

    def _fix(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        for key in list(node):
            if key in strip_keywords:
                del node[key]
        if node.get("type") == "object" and "properties" in node:
            node.setdefault("required", list(node["properties"].keys()))
            node["additionalProperties"] = False
            for prop in node["properties"].values():
                _fix(prop)
        if node.get("type") == "array" and "items" in node:
            _fix(node["items"])
        return node

    return _fix(schema)
