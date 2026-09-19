from __future__ import annotations

from typing import Any

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.decision.base import BaseDecisionProvider
from harness_evals.decision.types import DecisionResponse, NoulAnswer, NoulQuestion, ScoreAnswer, ScoreQuestion
from harness_evals.metrics.composite.decision_composite import DecisionCompositeMetric


class FakeGroupedProvider(BaseDecisionProvider):
    """Returns canned answers keyed by question name; records every a_ask() call."""

    def __init__(self, answers: dict[str, Any], error_states: set[str] | None = None):
        self.answers = answers
        self.error_states = error_states or set()
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    async def a_ask(self, state, questions):
        self.calls.append((state, dict(questions)))
        if state in self.error_states:
            raise RuntimeError(f"provider failed for state {state!r}")
        return DecisionResponse(
            model="fake-model",
            answers={name: self.answers[name] for name in questions},
            input_tokens=10,
            output_tokens=2,
        )


@pytest.mark.unit
def test_shared_state_field_produces_one_call():
    provider = FakeGroupedProvider(
        answers={
            "helpful": NoulAnswer(noul=0.8),
            "concise": NoulAnswer(noul=0.4),
        }
    )
    sub_scores = [
        {
            "name": "helpful",
            "weight": 0.5,
            "state_field": "output",
            "question": NoulQuestion(instructions="Is it helpful?"),
        },
        {
            "name": "concise",
            "weight": 0.5,
            "state_field": "output",
            "question": NoulQuestion(instructions="Is it concise?"),
        },
    ]
    metric = DecisionCompositeMetric(provider=provider, sub_scores=sub_scores, threshold=0.5)

    ec = EvalCase(input="q", output="some answer")
    score = metric.measure(ec)

    assert len(provider.calls) == 1
    state, questions = provider.calls[0]
    assert state == "some answer"
    assert set(questions) == {"helpful", "concise"}
    assert score.metadata["sub_scores"]["helpful"]["value"] == pytest.approx(0.8)
    assert score.metadata["sub_scores"]["concise"]["value"] == pytest.approx(0.4)


@pytest.mark.unit
def test_different_state_fields_produce_two_calls():
    provider = FakeGroupedProvider(
        answers={
            "helpful": NoulAnswer(noul=0.9),
            "grounded": NoulAnswer(noul=0.6),
        }
    )
    sub_scores = [
        {
            "name": "helpful",
            "weight": 0.5,
            "state_field": "output",
            "question": NoulQuestion(instructions="Is it helpful?"),
        },
        {
            "name": "grounded",
            "weight": 0.5,
            "state_field": "input",
            "question": NoulQuestion(instructions="Is it grounded?"),
        },
    ]
    metric = DecisionCompositeMetric(provider=provider, sub_scores=sub_scores, threshold=0.5)

    ec = EvalCase(input="the question", output="the answer")
    metric.measure(ec)

    assert len(provider.calls) == 2
    states = {state for state, _ in provider.calls}
    assert states == {"the answer", "the question"}


@pytest.mark.unit
def test_weighted_sum_combines_correctly():
    provider = FakeGroupedProvider(
        answers={
            "python_depth": ScoreAnswer(score=3, confidence=0.9, legend={}, probabilities={}),
            "team_leadership": ScoreAnswer(score=1, confidence=0.9, legend={}, probabilities={}),
        }
    )
    sub_scores = [
        {
            "name": "python_depth",
            "weight": 0.7,
            "state_field": "output",
            "expected_field": "expected.python_depth",
            "question": ScoreQuestion(instructions="Rate python depth", criteria=["novice", "mid", "senior", "expert"]),
        },
        {
            "name": "team_leadership",
            "weight": 0.3,
            "state_field": "output",
            "expected_field": "expected.team_leadership",
            "question": ScoreQuestion(instructions="Rate leadership", criteria=["none", "some", "lead", "exec"]),
        },
    ]
    metric = DecisionCompositeMetric(provider=provider, sub_scores=sub_scores, threshold=0.5)

    ec = EvalCase(input="resume", output="resume text", expected={"python_depth": 3, "team_leadership": 2})
    score = metric.measure(ec)

    # python_depth: exact match -> 1.0; team_leadership: |1/3 - 2/3| = 1/3 -> value 2/3
    expected_value = (1.0 * 0.7) + ((2 / 3) * 0.3)
    assert score.value == pytest.approx(expected_value)


@pytest.mark.unit
def test_group_provider_failure_isolated_to_that_group():
    provider = FakeGroupedProvider(
        answers={
            "ok_check": NoulAnswer(noul=0.7),
        },
        error_states={"bad state"},
    )
    sub_scores = [
        {
            "name": "failing_check",
            "weight": 0.5,
            "state_field": "input",
            "question": NoulQuestion(instructions="Check?"),
        },
        {
            "name": "ok_check",
            "weight": 0.5,
            "state_field": "output",
            "question": NoulQuestion(instructions="Check?"),
        },
    ]
    metric = DecisionCompositeMetric(provider=provider, sub_scores=sub_scores, threshold=0.5)

    ec = EvalCase(input="bad state", output="good state")
    score = metric.measure(ec)

    assert score.metadata["sub_scores"]["failing_check"]["status"] == "error"
    assert score.metadata["sub_scores"]["ok_check"]["status"] == "ok"
    assert score.metadata["sub_scores"]["ok_check"]["value"] == pytest.approx(0.7)
    # failing_check's weight still counts toward active_weight_sum but contributes 0
    assert score.value == pytest.approx(0.35)


@pytest.mark.unit
def test_skip_when_missing_avoids_provider_call():
    provider = FakeGroupedProvider(answers={})
    sub_scores = [
        {
            "name": "optional_check",
            "weight": 1.0,
            "state_field": "missing_field",
            "skip_when_missing": True,
            "question": NoulQuestion(instructions="Check?"),
        },
    ]
    metric = DecisionCompositeMetric(provider=provider, sub_scores=sub_scores, threshold=0.5)

    ec = EvalCase(input="q", output="a")
    score = metric.measure(ec)

    assert len(provider.calls) == 0
    assert score.metadata["sub_scores"]["optional_check"]["status"] == "skipped"
