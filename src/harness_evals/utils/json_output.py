"""Parse JSON values from LLM text (markdown fences, surrounding prose)."""

from __future__ import annotations

import json
import re
from typing import Any

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL | re.IGNORECASE)


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

    fence = _JSON_FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()

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
