"""Shared JSON Schema adaptation for structured output providers.

Grounded against provider documentation:
- OpenAI: https://developers.openai.com/docs/guides/structured-outputs
- Anthropic: https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs

Both providers support: anyOf, allOf, $ref, $defs/definitions, enum, const.
Neither provider supports: oneOf (OpenAI converts unions to anyOf via SDK;
Anthropic only documents anyOf and allOf).

Anthropic-specific: no external $ref, no recursive schemas, no allOf+$ref,
no complex enum types (objects in enums), minItems only 0 or 1.
"""

from __future__ import annotations

import copy
from typing import Any

# Ref: https://developers.openai.com/docs/guides/structured-outputs#supported-schemas
OPENAI_UNSUPPORTED = frozenset(
    {
        "default",
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

# Ref: https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs#json-schema-limitations
ANTHROPIC_UNSUPPORTED = frozenset(
    {
        "default",
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

    Recurses into ``anyOf``/``allOf`` branches and ``$defs``/``definitions``
    so that nested object schemas are also made strict. ``$ref`` nodes are
    left as-is — they are pointers resolved by the API, and the referenced
    definitions are fixed at their definition site.

    Note: ``oneOf`` is not supported by either provider. If present it is
    still recursed into (to strip keywords), but the API may reject the
    schema.

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
        for combiner in ("anyOf", "allOf", "oneOf"):
            if combiner in node:
                for branch in node[combiner]:
                    _fix(branch)
        if "$defs" in node:
            for defn in node["$defs"].values():
                _fix(defn)
        if "definitions" in node:
            for defn in node["definitions"].values():
                _fix(defn)
        return node

    return _fix(schema)
