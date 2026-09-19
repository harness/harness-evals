"""Shared fake decision provider for ChoiceMetric/ScoreMetric/NoulMetric tests."""

from __future__ import annotations

from harness_evals.decision.base import BaseDecisionProvider
from harness_evals.decision.types import Answer, DecisionResponse, Question


class FakeDecisionProvider(BaseDecisionProvider):
    """Returns a canned answer for every question name, or raises a canned error."""

    def __init__(self, answer: Answer | None = None, model: str = "fake-model", error: Exception | None = None):
        self.answer = answer
        self.model = model
        self.error = error
        self.calls: list[tuple[object, dict[str, Question]]] = []

    async def a_ask(self, state, questions: dict[str, Question]) -> DecisionResponse:
        self.calls.append((state, questions))
        if self.error is not None:
            raise self.error
        return DecisionResponse(
            model=self.model,
            answers={name: self.answer for name in questions},
            input_tokens=11,
            output_tokens=3,
        )
