"""Vendor-neutral typed question/answer shapes for decision providers.

Mirrors the shape of TypeSafe's System One primitives (Choice, Score, Noul)
without depending on the ``typesafe_sdk`` package, so ``BaseDecisionProvider``
implementations other than TypeSafe can satisfy the same contract.
"""

from __future__ import annotations

from dataclasses import dataclass

JSONContent = str | dict | list


@dataclass
class ChoiceQuestion:
    """Pick exactly one of N named categories."""

    instructions: JSONContent
    criteria: dict[str, JSONContent | None]


@dataclass
class ScoreQuestion:
    """Rate against an ordered rubric of 2-10 levels, index 0 = low end."""

    instructions: JSONContent
    criteria: list[JSONContent]


@dataclass
class NoulQuestion:
    """Calibrated yes/no question."""

    instructions: JSONContent
    criteria: dict[str, str] | None = None


Question = ChoiceQuestion | ScoreQuestion | NoulQuestion


@dataclass
class ChoiceAnswer:
    choice: str
    confidence: float
    probabilities: dict[str, float]


@dataclass
class ScoreAnswer:
    score: float
    confidence: float
    legend: dict[int, JSONContent]
    probabilities: dict[int, float]


@dataclass
class NoulAnswer:
    noul: float


Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer


@dataclass
class DecisionResponse:
    """Result of one batched call to a decision provider."""

    model: str
    answers: dict[str, Answer]
    input_tokens: int | None = None
    output_tokens: int | None = None
