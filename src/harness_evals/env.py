"""Environment variable interpolation helpers."""

from __future__ import annotations

import os
import re
from typing import Any

_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def resolve_env_value(value: str) -> str:
    """Replace ``${VAR}`` and ``${VAR:-default}`` references with resolved values."""

    def _replace(match: re.Match[str]) -> str:
        var = match.group(1)
        default = match.group(2)
        val = os.environ.get(var)
        if val is None:
            if default is not None:
                return default
            raise ValueError(f"Environment variable ${{{var}}} is not set")
        return val

    return _ENV_VAR_RE.sub(_replace, value)


def resolve_env_in_value(value: Any) -> Any:
    """Resolve ``${VAR}`` references recursively in strings, dicts, and lists."""

    if isinstance(value, str) and "${" in value:
        return resolve_env_value(value)
    if isinstance(value, dict):
        return {key: resolve_env_in_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_env_in_value(item) for item in value]
    return value
