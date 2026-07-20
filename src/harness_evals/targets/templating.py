"""Request templating for HTTP targets.

Builds a per-request JSON body from a ``body_template`` (and per-request header
values from ``headers``) by resolving ``{{...}}`` placeholders against the
current golden. Placeholders address golden fields by dotted path:

    {{input}}            -> golden.input (whole value, native type preserved)
    {{input.question}}   -> golden.input["question"]
    {{input.items.0}}    -> golden.input["items"][0]
    {{metadata.user_id}} -> (golden.metadata or {})["user_id"]
    {{env.VAR}}          -> os.environ["VAR"] (for secrets injected at runtime)

A string that is *exactly* one placeholder resolves to the referenced value with
its native type — so ``{{input}}`` where the input is a dict yields a dict, not a
stringified dict. A placeholder embedded in surrounding text (``"Hi {{input.name}}"``)
is string-interpolated. A placeholder that does not resolve raises ``ValueError``
rather than silently sending a null/omitted field.

This is distinct from ``${VAR}`` interpolation, which resolves *environment
variables at config-load time*; ``{{...}}`` resolves *golden fields per request*.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from harness_evals.core.golden import Golden

# Matches a single ``{{ path }}`` placeholder; the captured group is the path.
_PLACEHOLDER = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")
# The whole string is exactly one placeholder (ignoring surrounding whitespace).
_WHOLE_PLACEHOLDER = re.compile(r"^\s*\{\{\s*([^{}]+?)\s*\}\}\s*$")

_MISSING = object()


def render_request_body(body_template: dict | None, golden: Golden) -> dict:
    """Build a request body from ``body_template``, resolving ``{{...}}`` against ``golden``.

    When ``body_template`` is ``None`` the default body ``{"input": golden.input}``
    is returned. Otherwise the template is rendered recursively (dict values, list
    items, and strings). Non-string leaves pass through unchanged.
    """
    if body_template is None:
        return {"input": golden.input}
    context = _context(golden)
    rendered = _render(body_template, context)
    # A body_template is always a dict, so rendering yields a dict.
    return rendered  # type: ignore[return-value]


def render_headers(headers: dict[str, str], golden: Golden) -> dict[str, str]:
    """Resolve ``{{...}}`` placeholders in header *values* against ``golden``.

    Headers are always strings over the wire, so values are string-interpolated
    even when a value is exactly one placeholder (unlike the body, a header can't
    carry a dict/list). Values with no placeholder pass through unchanged. Header
    names are never templated. An unresolved placeholder raises ``ValueError``.
    """
    if not headers:
        return {}
    context = _context(golden)
    rendered: dict[str, str] = {}
    for name, value in headers.items():
        rendered[name] = _interpolate(value, context) if isinstance(value, str) and "{{" in value else value
    return rendered


def _context(golden: Golden) -> dict[str, Any]:
    return {"input": golden.input, "metadata": golden.metadata or {}}


def _render(node: Any, context: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        return {key: _render(value, context) for key, value in node.items()}
    if isinstance(node, list):
        return [_render(item, context) for item in node]
    if isinstance(node, str):
        return _render_string(node, context)
    return node


def _render_string(value: str, context: dict[str, Any]) -> Any:
    whole = _WHOLE_PLACEHOLDER.match(value)
    if whole is not None:
        # Whole-string placeholder: substitute the native value (dict/list/int/…).
        return _resolve(whole.group(1), context)

    if "{{" not in value:
        return value

    return _interpolate(value, context)


def _interpolate(value: str, context: dict[str, Any]) -> str:
    """Replace every ``{{...}}`` placeholder in ``value`` with its stringified value."""

    def _sub(match: re.Match[str]) -> str:
        return _stringify(_resolve(match.group(1), context))

    return _PLACEHOLDER.sub(_sub, value)


def _resolve(expr: str, context: dict[str, Any]) -> Any:
    """Resolve a dotted placeholder path against the context, raising if missing."""
    parts = expr.split(".")
    root = parts[0]

    if root == "env":
        if len(parts) < 2:
            raise ValueError(
                f"template placeholder {{{{{expr}}}}} — 'env' requires a variable name (e.g. {{{{env.MY_VAR}}}})"
            )
        var_name = parts[1]
        value = os.environ.get(var_name)
        if value is None:
            raise ValueError(
                f"template placeholder {{{{{expr}}}}} references environment variable {var_name!r} which is not set"
            )
        return value

    if root not in context:
        raise ValueError(
            f"template placeholder {{{{{expr}}}}} references unknown root {root!r} "
            f"(available: {', '.join(sorted(context))}, env)"
        )
    current: Any = context[root]
    for part in parts[1:]:
        current = _step(current, part)
        if current is _MISSING:
            raise ValueError(
                f"template placeholder {{{{{expr}}}}} did not resolve against the golden (no value at {part!r})"
            )
    return current


def _step(obj: Any, part: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(part, _MISSING)
    if isinstance(obj, list):
        try:
            return obj[int(part)]
        except (ValueError, IndexError):
            return _MISSING
    return _MISSING


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None or isinstance(value, (dict, list, bool)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)
