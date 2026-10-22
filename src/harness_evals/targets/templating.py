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
from dataclasses import dataclass
from typing import Any

from harness_evals.core.golden import Golden

# Matches a single ``{{ path }}`` placeholder; the captured group is the path.
_PLACEHOLDER = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")
# The whole string is exactly one placeholder (ignoring surrounding whitespace).
_WHOLE_PLACEHOLDER = re.compile(r"^\s*\{\{\s*([^{}]+?)\s*\}\}\s*$")

_MISSING = object()


@dataclass(frozen=True)
class PlaceholderUse:
    """One ``{{...}}`` placeholder that contributed to a rendered string.

    ``kind`` is ``env`` when the value was read from the process environment and
    ``golden`` when it was read from the golden. Diagnostics use this to redact
    environment-sourced values without guessing which literals are secrets.
    """

    expression: str
    kind: str
    env_var: str | None = None


def render_request_body(body_template: dict | None, golden: Golden) -> dict:
    """Build a request body from ``body_template``, resolving ``{{...}}`` against ``golden``.

    When ``body_template`` is ``None`` the default body ``{"input": golden.input}``
    is returned. Otherwise the template is rendered recursively (dict values, list
    items, and strings). Non-string leaves pass through unchanged.
    """
    rendered, _secrets = render_request_body_with_provenance(body_template, golden)
    return rendered


def render_request_body_with_provenance(body_template: dict | None, golden: Golden) -> tuple[dict, list[str]]:
    """Render a body and return environment values that were substituted into it.

    The secret list contains the resolved ``{{env.*}}`` strings in encounter
    order, including duplicates when the same variable is used more than once.
    Callers redact every occurrence of those strings, so a secret copied into
    another body field is covered by provenance rather than by field name.
    """
    if body_template is None:
        return {"input": golden.input}, []
    secrets: list[str] = []
    rendered = _render(body_template, _context(golden), secrets)
    # A body_template is always a dict, so rendering yields a dict.
    return rendered, secrets  # type: ignore[return-value]


def render_headers(headers: dict[str, str], golden: Golden) -> dict[str, str]:
    """Resolve ``{{...}}`` placeholders in header *values* against ``golden``.

    Headers are always strings over the wire, so values are string-interpolated
    even when a value is exactly one placeholder (unlike the body, a header can't
    carry a dict/list). Values with no placeholder pass through unchanged. Header
    names are never templated. An unresolved placeholder raises ``ValueError``.
    """
    rendered, _uses, _secrets = render_headers_with_provenance(headers, golden)
    return rendered


def render_headers_with_provenance(
    headers: dict[str, str], golden: Golden
) -> tuple[dict[str, str], dict[str, tuple[PlaceholderUse, ...]], list[str]]:
    """Render header values and record which placeholders produced them.

    Header names are never templated. A header with no placeholder has no entry
    in the use map (callers treat that as a literal). Environment values land
    in the returned secret list so a later body field with the same bytes can
    be redacted.
    """
    if not headers:
        return {}, {}, []
    context = _context(golden)
    secrets: list[str] = []
    rendered: dict[str, str] = {}
    uses: dict[str, tuple[PlaceholderUse, ...]] = {}
    for name, value in headers.items():
        if isinstance(value, str) and "{{" in value:
            sink: list[PlaceholderUse] = []
            rendered[name] = _interpolate(value, context, secrets, sink)
            uses[name] = tuple(sink)
        else:
            rendered[name] = value
    return rendered, uses, secrets


def _context(golden: Golden) -> dict[str, Any]:
    return {"input": golden.input, "metadata": golden.metadata or {}}


def _render(node: Any, context: dict[str, Any], secrets: list[str] | None = None) -> Any:
    if isinstance(node, dict):
        return {key: _render(value, context, secrets) for key, value in node.items()}
    if isinstance(node, list):
        return [_render(item, context, secrets) for item in node]
    if isinstance(node, str):
        return _render_string(node, context, secrets)
    return node


def _render_string(value: str, context: dict[str, Any], secrets: list[str] | None = None) -> Any:
    whole = _WHOLE_PLACEHOLDER.match(value)
    if whole is not None:
        # Whole-string placeholder: substitute the native value (dict/list/int/…).
        return _resolve(whole.group(1), context, secrets, None)

    if "{{" not in value:
        return value

    return _interpolate(value, context, secrets, None)


def _interpolate(
    value: str,
    context: dict[str, Any],
    secrets: list[str] | None = None,
    uses: list[PlaceholderUse] | None = None,
) -> str:
    """Replace every ``{{...}}`` placeholder in ``value`` with its stringified value."""

    def _sub(match: re.Match[str]) -> str:
        return _stringify(_resolve(match.group(1), context, secrets, uses))

    return _PLACEHOLDER.sub(_sub, value)


def _resolve(
    expr: str,
    context: dict[str, Any],
    secrets: list[str] | None = None,
    uses: list[PlaceholderUse] | None = None,
) -> Any:
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
        # Record the resolved value, not the variable name, so a copy of the
        # same bytes in another field can be redacted by provenance.
        if secrets is not None and value:
            secrets.append(value)
        if uses is not None:
            uses.append(PlaceholderUse(expression=f"env.{var_name}", kind="env", env_var=var_name))
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
    if uses is not None:
        uses.append(PlaceholderUse(expression=expr, kind="golden"))
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
