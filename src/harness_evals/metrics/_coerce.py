"""Safe numeric coercion for values parsed from LLM judge responses.

The ``generate_json`` contract guarantees a key is present (via schema
``required``) but not that its value is numeric — a judge may emit
``{"score": "high"}`` or ``null``. A bare ``float()``/``int()`` would raise
``ValueError``/``TypeError`` and abort the whole evaluation. These helpers
degrade to the metric's safe default instead.
"""

from __future__ import annotations

from typing import Any


def safe_float(value: Any, default: float) -> float:
    """Coerce *value* to float, returning *default* if it is not numeric."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int) -> int:
    """Coerce *value* to int, returning *default* if it is not numeric."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
