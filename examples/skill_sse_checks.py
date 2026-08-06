"""Helpers for Skill-tool SSE trajectory checks in conversation evals."""

from __future__ import annotations


def normalize_skill_name(name: str) -> str:
    """Canonical form for comparing prod underscore vs SKILL.md hyphen names."""
    return name.strip().replace("_", "-").lower()


def skill_names_equivalent(left: str, right: str) -> bool:
    return normalize_skill_name(left) == normalize_skill_name(right)
