"""TypeSafe-backed decision provider."""

from __future__ import annotations

import os
from typing import Any

from harness_evals.decision.base import BaseDecisionProvider
from harness_evals.decision.types import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)
from harness_evals.llm.usage import record_token_usage


def _to_sdk_question(question: Question) -> Any:
    import typesafe_sdk

    if isinstance(question, ChoiceQuestion):
        return typesafe_sdk.Choice(instructions=question.instructions, criteria=question.criteria)
    if isinstance(question, ScoreQuestion):
        return typesafe_sdk.Score(instructions=question.instructions, criteria=question.criteria)
    if isinstance(question, NoulQuestion):
        return typesafe_sdk.Noul(instructions=question.instructions, criteria=question.criteria)
    raise TypeError(f"Unsupported question type: {type(question).__name__}")


def _from_sdk_answer(answer: Any) -> Answer:
    import typesafe_sdk

    if isinstance(answer, typesafe_sdk.ChoiceAnswer):
        return ChoiceAnswer(
            choice=answer.choice, confidence=answer.confidence, probabilities=dict(answer.probabilities)
        )
    if isinstance(answer, typesafe_sdk.ScoreAnswer):
        return ScoreAnswer(
            score=answer.score,
            confidence=answer.confidence,
            legend=dict(answer.legend),
            probabilities=dict(answer.probabilities),
        )
    if isinstance(answer, typesafe_sdk.NoulAnswer):
        return NoulAnswer(noul=answer.noul)
    raise TypeError(f"Unsupported answer type: {type(answer).__name__}")


class TypeSafeDecisionProvider(BaseDecisionProvider):
    """TypeSafe System-One-backed decision provider. Requires ``pip install harness-evals[decision]``.

    API key resolution: constructor ``api_key`` > ``TYPESAFE_API_KEY`` env var
    (handled by ``typesafe_sdk`` itself).
    """

    def __init__(
        self,
        model: str = "jev-latest",
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        try:
            import typesafe_sdk
        except ImportError as e:
            raise ImportError("Install typesafe-sdk: pip install harness-evals[decision]") from e

        resolved_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not resolved_key:
            raise ValueError("No API key: pass api_key= or set TYPESAFE_API_KEY")

        self.model = model
        self._client = typesafe_sdk.AsyncTypeSafeClient(api_key=resolved_key, model=model, timeout=timeout)

    async def a_ask(self, state: str | dict | list, questions: dict[str, Question]) -> DecisionResponse:
        sdk_questions = {name: _to_sdk_question(q) for name, q in questions.items()}
        response = await self._client.system_one(state=state, questions=sdk_questions)

        answers = {name: _from_sdk_answer(answer) for name, answer in response.answers.items()}
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        record_token_usage(input_tokens=input_tokens, output_tokens=output_tokens, model=response.model)

        return DecisionResponse(
            model=response.model,
            answers=answers,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
