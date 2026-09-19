"""Tests for the TypeSafe decision provider.

Stubs ``typesafe_sdk`` in ``sys.modules`` (same pattern as the openai/anthropic
provider tests) so these run without the real package installed and without
any live network call.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from harness_evals.decision.types import ChoiceQuestion, NoulQuestion, ScoreQuestion


class _SdkNoul:
    def __init__(self, instructions=None, criteria=None):
        self.instructions = instructions
        self.criteria = criteria


class _SdkChoice:
    def __init__(self, instructions=None, criteria=None):
        self.instructions = instructions
        self.criteria = criteria


class _SdkScore:
    def __init__(self, instructions=None, criteria=None):
        self.instructions = instructions
        self.criteria = criteria


class _SdkNoulAnswer:
    def __init__(self, noul):
        self.noul = noul


class _SdkChoiceAnswer:
    def __init__(self, choice, confidence, probabilities):
        self.choice = choice
        self.confidence = confidence
        self.probabilities = probabilities


class _SdkScoreAnswer:
    def __init__(self, score, confidence, legend, probabilities):
        self.score = score
        self.confidence = confidence
        self.legend = legend
        self.probabilities = probabilities


def _make_sdk_module(system_one_impl, api_error=None):
    mock_module = MagicMock()
    mock_module.Noul = _SdkNoul
    mock_module.Choice = _SdkChoice
    mock_module.Score = _SdkScore
    mock_module.NoulAnswer = _SdkNoulAnswer
    mock_module.ChoiceAnswer = _SdkChoiceAnswer
    mock_module.ScoreAnswer = _SdkScoreAnswer
    if api_error is not None:
        mock_module.TypeSafeAPIError = api_error

    client = MagicMock()
    client.system_one = system_one_impl
    mock_module.AsyncTypeSafeClient.return_value = client
    return mock_module, client


@pytest.fixture
def api_key_env(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test-key")


@pytest.mark.unit
class TestQuestionTranslation:
    async def test_noul_question_round_trips(self, monkeypatch, api_key_env):
        captured = {}

        async def fake_system_one(*, state, questions):
            captured["state"] = state
            captured["questions"] = questions
            return SimpleNamespace(
                model="jev-latest",
                usage=SimpleNamespace(input_tokens=10, output_tokens=2),
                answers={"is_escalation": _SdkNoulAnswer(noul=0.92)},
            )

        mock_module, _ = _make_sdk_module(fake_system_one)
        monkeypatch.setitem(sys.modules, "typesafe_sdk", mock_module)

        from harness_evals.decision.typesafe import TypeSafeDecisionProvider

        provider = TypeSafeDecisionProvider()
        response = await provider.a_ask(
            state="Can I talk to a human?",
            questions={"is_escalation": NoulQuestion(instructions="Is this an escalation request?")},
        )

        assert isinstance(captured["questions"]["is_escalation"], _SdkNoul)
        assert captured["questions"]["is_escalation"].instructions == "Is this an escalation request?"
        assert response.answers["is_escalation"].noul == 0.92
        assert response.model == "jev-latest"
        assert response.input_tokens == 10
        assert response.output_tokens == 2

    async def test_choice_question_round_trips(self, monkeypatch, api_key_env):
        async def fake_system_one(*, state, questions):
            return SimpleNamespace(
                model="jev-latest",
                usage=SimpleNamespace(input_tokens=5, output_tokens=1),
                answers={
                    "department": _SdkChoiceAnswer(
                        choice="billing", confidence=0.8, probabilities={"billing": 0.8, "returns": 0.2}
                    )
                },
            )

        mock_module, _ = _make_sdk_module(fake_system_one)
        monkeypatch.setitem(sys.modules, "typesafe_sdk", mock_module)

        from harness_evals.decision.typesafe import TypeSafeDecisionProvider

        provider = TypeSafeDecisionProvider()
        response = await provider.a_ask(
            state="I was charged twice",
            questions={
                "department": ChoiceQuestion(instructions="Which team?", criteria={"billing": None, "returns": None})
            },
        )

        answer = response.answers["department"]
        assert answer.choice == "billing"
        assert answer.confidence == 0.8
        assert answer.probabilities == {"billing": 0.8, "returns": 0.2}

    async def test_score_question_round_trips(self, monkeypatch, api_key_env):
        async def fake_system_one(*, state, questions):
            return SimpleNamespace(
                model="jev-latest",
                usage=SimpleNamespace(input_tokens=7, output_tokens=3),
                answers={
                    "urgency": _SdkScoreAnswer(
                        score=1.3,
                        confidence=0.7,
                        legend={0: "low", 1: "medium", 2: "high"},
                        probabilities={0: 0.1, 1: 0.5, 2: 0.4},
                    )
                },
            )

        mock_module, _ = _make_sdk_module(fake_system_one)
        monkeypatch.setitem(sys.modules, "typesafe_sdk", mock_module)

        from harness_evals.decision.typesafe import TypeSafeDecisionProvider

        provider = TypeSafeDecisionProvider()
        response = await provider.a_ask(
            state="This is on fire",
            questions={"urgency": ScoreQuestion(instructions="How urgent?", criteria=["low", "medium", "high"])},
        )

        answer = response.answers["urgency"]
        assert answer.score == 1.3
        assert answer.legend == {0: "low", 1: "medium", 2: "high"}
        assert answer.probabilities == {0: 0.1, 1: 0.5, 2: 0.4}

    async def test_mixed_question_types_one_call(self, monkeypatch, api_key_env):
        call_count = {"n": 0}

        async def fake_system_one(*, state, questions):
            call_count["n"] += 1
            assert set(questions) == {"a", "b", "c"}
            return SimpleNamespace(
                model="jev-latest",
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                answers={
                    "a": _SdkNoulAnswer(noul=0.5),
                    "b": _SdkChoiceAnswer(choice="x", confidence=0.9, probabilities={"x": 0.9}),
                    "c": _SdkScoreAnswer(
                        score=1.0, confidence=0.6, legend={0: "lo", 1: "hi"}, probabilities={0: 0.4, 1: 0.6}
                    ),
                },
            )

        mock_module, _ = _make_sdk_module(fake_system_one)
        monkeypatch.setitem(sys.modules, "typesafe_sdk", mock_module)

        from harness_evals.decision.typesafe import TypeSafeDecisionProvider

        provider = TypeSafeDecisionProvider()
        response = await provider.a_ask(
            state="mixed",
            questions={
                "a": NoulQuestion(instructions="?"),
                "b": ChoiceQuestion(instructions="?", criteria={"x": None}),
                "c": ScoreQuestion(instructions="?", criteria=["lo", "hi"]),
            },
        )

        assert call_count["n"] == 1
        assert set(response.answers) == {"a", "b", "c"}


@pytest.mark.unit
class TestConstruction:
    def test_missing_package_raises_clear_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
        from harness_evals.decision.typesafe import TypeSafeDecisionProvider

        with pytest.raises(ImportError, match="harness-evals\\[decision\\]"):
            TypeSafeDecisionProvider()

    def test_missing_api_key_raises_clear_error(self, monkeypatch):
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        mock_module, _ = _make_sdk_module(None)
        monkeypatch.setitem(sys.modules, "typesafe_sdk", mock_module)

        from harness_evals.decision.typesafe import TypeSafeDecisionProvider

        with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
            TypeSafeDecisionProvider()

    def test_constructor_api_key_used_over_env(self, monkeypatch):
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        mock_module, _ = _make_sdk_module(None)
        monkeypatch.setitem(sys.modules, "typesafe_sdk", mock_module)

        from harness_evals.decision.typesafe import TypeSafeDecisionProvider

        TypeSafeDecisionProvider(api_key="explicit-key")
        assert mock_module.AsyncTypeSafeClient.call_args.kwargs["api_key"] == "explicit-key"


@pytest.mark.unit
class TestUsageAndErrors:
    async def test_usage_recorded_and_returned_independent_of_collector(self, monkeypatch, api_key_env):
        async def fake_system_one(*, state, questions):
            return SimpleNamespace(
                model="jev-latest",
                usage=SimpleNamespace(input_tokens=42, output_tokens=8),
                answers={"a": _SdkNoulAnswer(noul=0.1)},
            )

        mock_module, _ = _make_sdk_module(fake_system_one)
        monkeypatch.setitem(sys.modules, "typesafe_sdk", mock_module)

        from harness_evals.decision.typesafe import TypeSafeDecisionProvider
        from harness_evals.llm.usage import collect_token_usage

        provider = TypeSafeDecisionProvider()

        # Independent of an active collector: response carries usage directly.
        response = await provider.a_ask(state="x", questions={"a": NoulQuestion(instructions="?")})
        assert response.input_tokens == 42
        assert response.output_tokens == 8

        # Also recorded into an active collector, same as judge-metric usage.
        with collect_token_usage() as usage:
            await provider.a_ask(state="x", questions={"a": NoulQuestion(instructions="?")})
        assert usage.input_tokens == 42
        assert usage.output_tokens == 8

    async def test_provider_error_propagates_unchanged(self, monkeypatch, api_key_env):
        class _FakeApiError(Exception):
            pass

        async def failing_system_one(*, state, questions):
            raise _FakeApiError("upstream 529")

        mock_module, _ = _make_sdk_module(failing_system_one, api_error=_FakeApiError)
        monkeypatch.setitem(sys.modules, "typesafe_sdk", mock_module)

        from harness_evals.decision.typesafe import TypeSafeDecisionProvider

        provider = TypeSafeDecisionProvider()
        with pytest.raises(_FakeApiError, match="upstream 529"):
            await provider.a_ask(state="x", questions={"a": NoulQuestion(instructions="?")})
