"""Parse JSON values from LLM text (markdown fences, surrounding prose)."""

from __future__ import annotations

import json
import re
from typing import Any

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL | re.IGNORECASE)


def _json_candidate_text(text: str) -> str:
    """Strip outer whitespace and use fenced block content when present."""
    stripped = text.strip()
    fence = _JSON_FENCE_RE.search(stripped)
    if fence:
        return fence.group(1).strip()
    return stripped


def _decode_error_detail(exc: json.JSONDecodeError) -> str:
    if exc.lineno and exc.colno:
        return f"{exc.msg} (line {exc.lineno}, column {exc.colno})"
    if exc.pos is not None:
        return f"{exc.msg} (char {exc.pos})"
    return exc.msg


def json_parse_failure_reason(output: str | dict | list | None) -> str:
    """Short explanation when :func:`parse_json_value` returns ``None``."""
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
    last_error: str | None = None

    def note_error(exc: json.JSONDecodeError) -> None:
        nonlocal last_error
        last_error = _decode_error_detail(exc)

    try:
        json.loads(candidate)
    except json.JSONDecodeError as exc:
        note_error(exc)
    else:
        return "unexpected parse failure"

    if (candidate.startswith("{") or candidate.startswith("[")) and last_error:
        return last_error

    first_obj = text.find("{")
    last_obj = text.rfind("}")
    if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
        try:
            json.loads(text[first_obj : last_obj + 1])
        except json.JSONDecodeError as exc:
            note_error(exc)
        else:
            return "unexpected parse failure"

    first_arr = text.find("[")
    last_arr = text.rfind("]")
    if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
        try:
            json.loads(text[first_arr : last_arr + 1])
        except json.JSONDecodeError as exc:
            note_error(exc)

    if "{" not in text and "[" not in text:
        return "no JSON object or array found in output"
    return last_error or "could not decode JSON"


def parse_json_value(output: str | dict | list | None) -> Any | None:
    """Best-effort parse of model output into a JSON value (object or array)."""
    if output is None:
        return None
    if isinstance(output, (dict, list)):
        return output
    if not isinstance(output, str):
        return None

    text = output.strip()
    if not text:
        return None

    text = _json_candidate_text(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    if text.startswith("{") or text.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    first_obj = text.find("{")
    last_obj = text.rfind("}")
    if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
        try:
            return json.loads(text[first_obj : last_obj + 1])
        except json.JSONDecodeError:
            pass

    first_arr = text.find("[")
    last_arr = text.rfind("]")
    if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
        try:
            return json.loads(text[first_arr : last_arr + 1])
        except json.JSONDecodeError:
            pass

    return None
