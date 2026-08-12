"""Parse JSON values from LLM text (markdown fences, surrounding prose)."""

from __future__ import annotations

import json
import re
from typing import Any

_JSON_FENCE_RE = re.compile(r"```(?:json)?[ \t]*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_REASONING_RE = re.compile(r"<(reasoning|thinking|think)>.*?</\1>", re.DOTALL | re.IGNORECASE)


class _JsonParseFailedType:
    """Sentinel returned when string output cannot be parsed as JSON."""

    def __repr__(self) -> str:
        return "JSON_PARSE_FAILED"


JSON_PARSE_FAILED = _JsonParseFailedType()


def _strip_reasoning(text: str) -> str:
    return _REASONING_RE.sub("", text).strip()


def _json_candidate_text(text: str) -> str:
    """Use fenced block content when present; strip outer reasoning tags only as fallback.

    Prefer a leading bare object/array (optional trailing prose or example fences)
    before walking markdown fences. Among fences, prefer the first parseable
    object/array outside a reasoning/thinking span; if none exist, fall back to
    the first parseable scalar fence.
    """
    stripped = text.strip()
    try:
        json.loads(stripped)
    except json.JSONDecodeError:
        pass
    else:
        return stripped

    reasoning_spans = [match.span() for match in _REASONING_RE.finditer(stripped)]
    first_scalar: str | None = None

    # Prefer a bare object/array answer (outside reasoning spans) before walking
    # fences, so a trailing example/schema ```json``` block never outranks it —
    # e.g. "Answer:\n{...}\n\nSchema:\n```json\n{...}```".
    first_delim = min((i for i in (stripped.find("{"), stripped.find("[")) if i != -1), default=-1)
    if (
        first_delim != -1
        and not any(start <= first_delim < end for start, end in reasoning_spans)
        and _try_parse_json_text(stripped)[1] is None
    ):
        return stripped

    for raw_fence in _JSON_FENCE_RE.finditer(stripped):
        if any(start <= raw_fence.start() < end for start, end in reasoning_spans):
            continue
        candidate = raw_fence.group(1).strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, (dict, list)):
            return candidate
        if first_scalar is None:
            first_scalar = candidate

    stripped = _strip_reasoning(stripped)
    for fence in _JSON_FENCE_RE.finditer(stripped):
        candidate = fence.group(1).strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, (dict, list)):
            return candidate
        if first_scalar is None:
            first_scalar = candidate
    if first_scalar is not None:
        return first_scalar
    return stripped


def _decode_error_detail(exc: json.JSONDecodeError) -> str:
    if exc.lineno and exc.colno:
        return f"{exc.msg} (line {exc.lineno}, column {exc.colno})"
    if exc.pos is not None:
        return f"{exc.msg} (char {exc.pos})"
    return exc.msg


def _delimiter_is_anchored(text: str, start: int) -> bool:
    """True when a bare ``{`` / ``[`` looks like the payload, not an in-prose example.

    Accept only when the text starts at the delimiter, or the preceding prose ends
    with ``:`` or a newline (after stripping spaces/tabs). This keeps
    ``Sure — here you go:\\n{...}`` working while rejecting schema echoes inside refusals.
    """
    if start <= 0:
        return True
    prefix = text[:start].rstrip(" \t")
    return not prefix or prefix.endswith(":") or prefix.endswith("\n")


def _try_parse_anchored_value(text: str, open_ch: str) -> tuple[Any | None, str | None]:
    """Parse an anchored container via ``raw_decode`` so trailing braces do not extend the slice.

    Continues past anchored delimiters that fail to decode (e.g. a pseudo-schema
    echo) so a later valid answer can still be found.
    """
    start = text.find(open_ch)
    while start != -1 and not _delimiter_is_anchored(text, start):
        start = text.find(open_ch, start + 1)

    decoder = json.JSONDecoder()
    last_detail: str | None = None
    while start != -1:
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError as exc:
            last_detail = _decode_error_detail(exc)
        else:
            # Mid-prose hits on inner values (e.g. "feature_map": []) look anchored by
            # the key colon; reject when the remainder continues or closes the
            # surrounding structure.
            if start == 0 or not text[end:].lstrip().startswith((",", ":", "}", "]")):
                return value, None
        start = text.find(open_ch, start + 1)
        while start != -1 and not _delimiter_is_anchored(text, start):
            start = text.find(open_ch, start + 1)
    return None, last_detail


def _try_parse_json_text(text: str) -> tuple[Any | None, str | None]:
    """Run the same parse strategies as :func:`parse_json_value` on *text*."""
    last_error: str | None = None

    def note_error(detail: str) -> None:
        nonlocal last_error
        last_error = detail

    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        note_error(_decode_error_detail(exc))
    else:
        return value, None

    # Leading container that failed to parse as a whole — salvage only the
    # matching outer value (trailing prose), never an inner example later.
    if text.startswith("{"):
        value, error = _try_parse_anchored_value(text, "{")
        if error is None and value is not None:
            return value, None
        return None, error or last_error or "could not decode JSON"
    if text.startswith("["):
        value, error = _try_parse_anchored_value(text, "[")
        if error is None and value is not None:
            return value, None
        return None, error or last_error or "could not decode JSON"

    value, error = _try_parse_anchored_value(text, "{")
    if error is not None:
        note_error(error)
    elif value is not None:
        return value, None

    value, error = _try_parse_anchored_value(text, "[")
    if error is not None:
        note_error(error)
    elif value is not None:
        return value, None

    if "{" not in text and "[" not in text:
        return None, "no JSON object or array found in output"
    return None, last_error or "could not decode JSON"


def json_parse_failure_reason(output: str | dict | list | None) -> str:
    """Short explanation when :func:`parse_json_value` returns :data:`JSON_PARSE_FAILED`."""
    if output is None:
        return "output is empty or missing"
    if isinstance(output, (dict, list)):
        return "output is already structured data"
    if not isinstance(output, str):
        return f"output has unsupported type {type(output).__name__}"

    text = output.strip()
    if not text:
        return "output is empty"

    candidate = _json_candidate_text(text)
    _, error = _try_parse_json_text(candidate)
    if error is None:
        return "output parsed successfully; failure was not a parse error"
    return error or "could not decode JSON"


def parse_json_value(output: str | dict | list | None) -> Any:
    """Best-effort parse of model output into a JSON value.

    Returns :data:`JSON_PARSE_FAILED` when parsing fails. A successful parse of the
    JSON literal ``null`` returns Python ``None`` — callers must distinguish failure
    with ``value is JSON_PARSE_FAILED``.
    """
    if output is None:
        return JSON_PARSE_FAILED
    if isinstance(output, (dict, list)):
        return output
    if not isinstance(output, str):
        return JSON_PARSE_FAILED

    text = output.strip()
    if not text:
        return JSON_PARSE_FAILED

    candidate = _json_candidate_text(text)
    value, error = _try_parse_json_text(candidate)
    if error is not None:
        return JSON_PARSE_FAILED
    return value
