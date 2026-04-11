from __future__ import annotations

from typing import Any

import jsonpath_ng.ext


def get_by_path(obj: Any, path: str) -> Any:
    """Get value at dot-separated path. Supports array indices including negative.
    
    This is the fast-path implementation for simple dot paths.
    """
    for part in path.split("."):
        if isinstance(obj, dict) and part in obj:
            obj = obj[part]
        elif isinstance(obj, list):
            try:
                idx = int(part)
                obj = obj[idx]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return obj


def extract_path(obj: Any, path: str) -> Any:
    """Extract value using either fast dot-path or JSONPath.
    
    Falls back to JSONPath if the expression contains JSONPath-specific syntax
    like wildcards (*), filters (?()), or starts with '$'.
    """
    if not path:
        return obj
        
    # If it looks like JSONPath, use jsonpath-ng
    if path.startswith("$") or "[" in path or "*" in path or "?" in path:
        try:
            expr = jsonpath_ng.ext.parse(path)
            matches = expr.find(obj)
            if not matches:
                return None
            if len(matches) == 1:
                return matches[0].value
            return [m.value for m in matches]
        except Exception:
            return None

    # Fast path for simple dot notation
    return get_by_path(obj, path)
