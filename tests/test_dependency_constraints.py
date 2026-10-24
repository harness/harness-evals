"""Regression tests for security-sensitive package constraints."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CERTIFI_FLOOR = (2026, 10, 19)


def _version_tuple(version: str) -> tuple[int, int, int]:
    major, minor, patch = version.split(".")[:3]
    return int(major), int(minor), int(patch)


def test_certifi_constraint_and_lock_meet_security_floor() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text()
    lock = (ROOT / "poetry.lock").read_text()

    assert 'certifi = ">=2026.10.19"' in pyproject

    match = re.search(
        r'\[\[package\]\]\nname = "certifi"\nversion = "([^"]+)"',
        lock,
    )
    assert match is not None
    assert _version_tuple(match.group(1)) >= CERTIFI_FLOOR
