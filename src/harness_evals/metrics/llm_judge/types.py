"""Shared types for LLM-judge metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RubricLevel:
    """A score band with an expected outcome description.

    Used by ``GEvalMetric`` to provide score-band rubrics instead of a single
    free-form criteria string. A rubric is a list of ``RubricLevel`` objects
    covering a contiguous integer scoring range (typically 0-10 or 1-5).

    Example::

        rubric = [
            RubricLevel(0, 2, "Fix targets the wrong vulnerability type."),
            RubricLevel(3, 5, "Fix addresses symptoms but not root cause."),
            RubricLevel(6, 7, "Root cause addressed with minor gaps."),
            RubricLevel(8, 10, "Clean, complete root-cause fix."),
        ]
    """

    min_score: int
    max_score: int
    description: str

    def __post_init__(self) -> None:
        if self.min_score > self.max_score:
            raise ValueError(f"RubricLevel: min_score ({self.min_score}) must be <= max_score ({self.max_score})")
        if self.min_score < 0:
            raise ValueError(f"RubricLevel: min_score must be non-negative, got {self.min_score}")
        if not self.description or not self.description.strip():
            raise ValueError("RubricLevel: description must be a non-empty string")


__all__ = ["RubricLevel"]
