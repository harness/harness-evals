from __future__ import annotations

import json
import re
from typing import Any

from deepdiff import DeepDiff
import jsonschema

from harness_evals.utils.path import extract_path


class OperatorError(Exception):
    pass


def _resolve_paths(eval_case_dict: dict[str, Any], config: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    """Resolve JSONPaths from config into values."""
    resolved = {}
    for k in keys:
        path = config.get(k)
        if not path:
            continue
        val = extract_path(eval_case_dict, path)
        resolved[k] = val
    return resolved


def _deep_diff_operator(eval_case_dict: dict[str, Any], config: dict[str, Any]) -> float:
    res = _resolve_paths(eval_case_dict, config, ["output_field", "expected_field"])
    actual = res.get("output_field")
    expected = res.get("expected_field")

    if actual is None or expected is None:
        raise OperatorError("Missing required fields for deep_diff")

    diff = DeepDiff(
        expected,
        actual,
        ignore_order=config.get("ignore_order", True),
        exclude_paths=config.get("ignore_keys", []),
        get_deep_distance=True,
        threshold_to_diff_deeper=0,
    )
    dist = diff.get("deep_distance", 0.0)
    return max(0.0, min(1.0, 1.0 - dist))


def _json_schema_operator(eval_case_dict: dict[str, Any], config: dict[str, Any]) -> float:
    res = _resolve_paths(eval_case_dict, config, ["field"])
    val = res.get("field")
    if val is None:
        raise OperatorError("Missing field for json_schema")
        
    schema = config.get("schema")
    if not schema:
        raise OperatorError("Missing 'schema' in config")

    try:
        jsonschema.validate(instance=val, schema=schema)
        return 1.0
    except jsonschema.exceptions.ValidationError:
        return 0.0


def _field_exists_operator(eval_case_dict: dict[str, Any], config: dict[str, Any]) -> float:
    val = extract_path(eval_case_dict, config.get("field", ""))
    return 1.0 if val is not None else 0.0


def _equals_operator(eval_case_dict: dict[str, Any], config: dict[str, Any]) -> float:
    res = _resolve_paths(eval_case_dict, config, ["field", "expected_field"])
    actual = res.get("field")
    
    if "value" in config:
        expected = config["value"]
    else:
        expected = res.get("expected_field")

    return 1.0 if actual == expected else 0.0


def _contains_operator(eval_case_dict: dict[str, Any], config: dict[str, Any]) -> float:
    res = _resolve_paths(eval_case_dict, config, ["field"])
    actual = res.get("field")
    val = config.get("value")

    if actual is None or val is None:
        return 0.0

    try:
        return 1.0 if val in actual else 0.0
    except TypeError:
        return 0.0


def _contains_all_operator(eval_case_dict: dict[str, Any], config: dict[str, Any]) -> float:
    res = _resolve_paths(eval_case_dict, config, ["field"])
    actual = res.get("field")
    values = config.get("values", [])

    if actual is None or not values:
        return 0.0

    try:
        return 1.0 if all(v in actual for v in values) else 0.0
    except TypeError:
        return 0.0


def _count_match_operator(eval_case_dict: dict[str, Any], config: dict[str, Any]) -> float:
    res = _resolve_paths(eval_case_dict, config, ["field", "expected_field"])
    actual = res.get("field")
    expected = res.get("expected_field")

    try:
        len_actual = len(actual) if actual is not None else 0
        len_expected = len(expected) if expected is not None else 0
        if len_expected == 0:
            return 1.0 if len_actual == 0 else 0.0
            
        ratio = min(len_actual, len_expected) / max(len_actual, len_expected)
        return float(ratio)
    except TypeError:
        raise OperatorError("Field is not a collection with length")


def _list_field_match_operator(eval_case_dict: dict[str, Any], config: dict[str, Any]) -> float:
    res = _resolve_paths(eval_case_dict, config, ["field", "expected_field"])
    actual = res.get("field")
    expected = res.get("expected_field")

    if not isinstance(actual, list) or not isinstance(expected, list):
        raise OperatorError("Fields must be lists")

    if not expected:
        return 1.0 if not actual else 0.0

    matches = sum(1 for a, e in zip(actual, expected) if a == e)
    return matches / len(expected)


def _unique_ratio_operator(eval_case_dict: dict[str, Any], config: dict[str, Any]) -> float:
    res = _resolve_paths(eval_case_dict, config, ["field"])
    actual = res.get("field")

    if not isinstance(actual, list):
        raise OperatorError("Field must be a list")

    if not actual:
        return 0.0

    try:
        unique_count = len(set(actual))
        ratio = unique_count / len(actual)
        min_ratio = config.get("min_ratio", 0.0)
        return 1.0 if ratio >= min_ratio else (ratio / min_ratio)  # Partial score if below min
    except TypeError:
        raise OperatorError("List items are unhashable")


def _regex_match_operator(eval_case_dict: dict[str, Any], config: dict[str, Any]) -> float:
    res = _resolve_paths(eval_case_dict, config, ["field"])
    actual = res.get("field")
    pattern = config.get("pattern")

    if not isinstance(actual, str) or not pattern:
        return 0.0

    try:
        return 1.0 if re.search(pattern, actual) else 0.0
    except re.error as e:
        raise OperatorError(f"Invalid regex pattern: {e}")


def _range_operator(eval_case_dict: dict[str, Any], config: dict[str, Any]) -> float:
    res = _resolve_paths(eval_case_dict, config, ["field"])
    actual = res.get("field")

    if not isinstance(actual, (int, float)):
        return 0.0

    min_val = config.get("min", float("-inf"))
    max_val = config.get("max", float("inf"))

    return 1.0 if min_val <= actual <= max_val else 0.0


def _length_operator(eval_case_dict: dict[str, Any], config: dict[str, Any]) -> float:
    res = _resolve_paths(eval_case_dict, config, ["field"])
    actual = res.get("field")

    try:
        actual_len = len(actual)
        min_len = config.get("min_length", 0)
        max_len = config.get("max_length", float("inf"))
        return 1.0 if min_len <= actual_len <= max_len else 0.0
    except TypeError:
        return 0.0


# Registry of built-in operators
OPERATORS = {
    "deep_diff": _deep_diff_operator,
    "json_schema": _json_schema_operator,
    "field_exists": _field_exists_operator,
    "equals": _equals_operator,
    "contains": _contains_operator,
    "contains_all": _contains_all_operator,
    "count_match": _count_match_operator,
    "list_field_match": _list_field_match_operator,
    "unique_ratio": _unique_ratio_operator,
    "regex_match": _regex_match_operator,
    "range": _range_operator,
    "length": _length_operator,
}
