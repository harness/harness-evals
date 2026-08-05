"""Helpers for Skill-tool SSE trajectory checks in conversation evals."""

from __future__ import annotations

from typing import Any


def normalize_skill_name(name: str) -> str:
    """Canonical form for comparing prod underscore vs SKILL.md hyphen names."""
    return name.strip().replace("_", "-").lower()


def skill_names_equivalent(left: str, right: str) -> bool:
    return normalize_skill_name(left) == normalize_skill_name(right)


def _skill_request_match(skill_name: str) -> list[dict[str, Any]]:
    return [
        {"path": "$.name", "equals": "Skill"},
        {"path": "$.arguments.skill", "skill_equals": skill_name},
    ]


def skill_sse_checks(skill_names: list[str]) -> list[dict[str, Any]]:
    """SSE checks asserting Skill was invoked with an expected skill name.

    Tool *requests* are matched on ``arguments.skill`` (hyphen/underscore equivalent).
    Tool *results* only carry the tool name in the stream, so results assert ``Skill``
    was returned without re-checking the skill argument.
    """
    if not skill_names:
        return []

    if len(skill_names) == 1:
        request_check: dict[str, Any] = {
            "event": "assistant_tool_request",
            "path": "$.v[*]",
            "match": _skill_request_match(skill_names[0]),
        }
    else:
        request_check = {
            "event": "assistant_tool_request",
            "path": "$.v[*]",
            "match_any": [_skill_request_match(name) for name in skill_names],
        }

    result_check = {
        "event": "assistant_tool_result",
        "path": "$.v[*]",
        "match": [{"path": "$.name", "equals": "Skill"}],
    }
    return [request_check, result_check]


def upgrade_skill_checks_in_sse_checks(
    sse_checks: list[dict[str, Any]],
    skill_names: list[str],
) -> list[dict[str, Any]]:
    """Replace generic Skill-only checks with skill-specific request assertions."""
    if not skill_names or _has_skill_specific_checks(sse_checks):
        return sse_checks

    upgraded = skill_sse_checks(skill_names)
    if not upgraded:
        return sse_checks

    filtered: list[dict[str, Any]] = []
    skip_next_result = False
    for check in sse_checks:
        if skip_next_result:
            skip_next_result = False
            if _is_skill_only_result(check):
                continue
        if _is_skill_only_request(check):
            skip_next_result = True
            continue
        if _is_skill_only_result(check):
            continue
        filtered.append(check)

    # Insert skill checks before assistant_message if present, else append.
    insert_at = len(filtered)
    for index, check in enumerate(filtered):
        if check.get("event") == "assistant_message":
            insert_at = index
            break
    return filtered[:insert_at] + upgraded + filtered[insert_at:]


def _is_skill_only_request(check: dict[str, Any]) -> bool:
    if check.get("event") != "assistant_tool_request":
        return False
    match = check.get("match")
    return isinstance(match, list) and match == [{"path": "$.name", "equals": "Skill"}]


def _is_skill_only_result(check: dict[str, Any]) -> bool:
    if check.get("event") != "assistant_tool_result":
        return False
    match = check.get("match")
    return isinstance(match, list) and match == [{"path": "$.name", "equals": "Skill"}]


def _has_skill_specific_checks(sse_checks: list[dict[str, Any]]) -> bool:
    for check in sse_checks:
        if check.get("event") != "assistant_tool_request":
            continue
        nested_lists: list[list[dict[str, Any]]] = []
        if isinstance(check.get("match"), list):
            nested_lists.append(check["match"])
        if isinstance(check.get("match_any"), list):
            nested_lists.extend(item for item in check["match_any"] if isinstance(item, list))
        for nested in nested_lists:
            for clause in nested:
                if isinstance(clause, dict) and "skill_equals" in clause:
                    return True
    return False
