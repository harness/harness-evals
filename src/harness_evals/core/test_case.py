from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TestCase:
    """A single evaluation test case.

    For reliability metrics that require multiple runs of the same task,
    populate the `runs` field with K repeated TestCase instances.
    Nested runs on sub-cases are ignored — ReliabilityMetric reads
    test_case.runs but never recurses into each run's .runs field.
    """

    input: str
    actual_output: str | dict | list
    expected_output: str | dict | list | None = None
    context: list[str] | None = None
    metadata: dict[str, Any] | None = field(default=None)
    tags: dict[str, str] | None = field(default=None)
    runs: list[TestCase] | None = field(default=None)
