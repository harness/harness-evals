"""Configured input alternatives and decision validation before provider calls."""

from __future__ import annotations

from itertools import product
from typing import Any

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, MetricInput, MetricInputs, OnlineScope
from harness_evals.core.score import Score
from harness_evals.core.types import Message
from harness_evals.decision.base import BaseDecisionProvider
from harness_evals.decision.types import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)
from harness_evals.metrics import (
    ChoiceMetric,
    CompositeMetric,
    DecisionCompositeMetric,
    HallucinationMetric,
    NoulMetric,
    ScoreMetric,
    StructuralSimilarityMetric,
)
from harness_evals.metrics.composite import operators
from tests.conftest import MockLLM

OUTPUT = MetricInput.OUTPUT
EXPECTED = MetricInput.EXPECTED
CONTEXT = MetricInput.CONTEXT
MESSAGES = MetricInput.MESSAGES
METADATA = MetricInput.METADATA
TAGS = MetricInput.TAGS


class RecordingProvider(BaseDecisionProvider):
    """Local-only provider whose call history exposes premature evaluation."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Question]]] = []

    async def a_ask(self, state: Any, questions: dict[str, Question]) -> DecisionResponse:
        self.calls.append((state, dict(questions)))
        answers = {}
        for name, question in questions.items():
            if isinstance(question, ChoiceQuestion):
                answers[name] = ChoiceAnswer(choice="a", confidence=0.8, probabilities={"a": 0.8})
            elif isinstance(question, ScoreQuestion):
                answers[name] = ScoreAnswer(score=1.3, confidence=0.8, legend={}, probabilities={})
            else:
                answers[name] = NoulAnswer(noul=0.8)
        return DecisionResponse(model="recording", answers=answers)


_FLAGS = (
    "include_messages_as_reference",
    "include_assistant_tool_inputs_as_reference",
    "include_assistant_tool_results_as_reference",
    "include_scenario_metadata_as_reference",
    "include_sse_events_as_reference",
)


@pytest.mark.unit
@pytest.mark.parametrize("enabled", tuple(product((False, True), repeat=len(_FLAGS))))
def test_hallucination_declares_every_enabled_reference_as_an_alternative(enabled: tuple[bool, ...]) -> None:
    metric = HallucinationMetric(llm=MockLLM(), **dict(zip(_FLAGS, enabled, strict=True)))
    references = {CONTEXT, EXPECTED}
    if any(enabled[:3]):
        references.add(MESSAGES)
    if any(enabled[3:]):
        references.add(METADATA)

    assert metric.eval_inputs == MetricInputs({OUTPUT}, any_of=({field} for field in references))
    assert metric.eval_inputs.fields == {OUTPUT, *references}
    assert bool(metric.applicability.online_scopes) == any(enabled[:3])
    assert OnlineScope.GROUP not in metric.applicability.online_scopes


@pytest.mark.unit
@pytest.mark.parametrize("reference", ["context", "expected", "messages", "metadata"])
def test_hallucination_can_use_any_reference_without_the_other_sources(reference: str) -> None:
    llm = MockLLM(default={"score": 1.0})
    metric = HallucinationMetric(
        llm=llm, include_messages_as_reference=True, include_scenario_metadata_as_reference=True
    )
    values = {
        "context": ["context fact"],
        "expected": "expected fact",
        "messages": [Message(role="user", content="message fact")],
        "metadata": {"scenario": "scenario fact"},
    }
    case = EvalCase(input="q", output="fact", **{reference: values[reference]})

    assert metric.measure(case).value == 1.0
    assert len(llm.prompts) == 1


@pytest.mark.unit
def test_hallucination_keeps_all_references_in_projection_and_prompt() -> None:
    llm = MockLLM(default={"score": 1.0})
    metric = HallucinationMetric(
        llm=llm, include_messages_as_reference=True, include_scenario_metadata_as_reference=True
    )
    case = EvalCase(
        input="unused",
        output="answer",
        context=["context fact"],
        expected="expected fact",
        messages=[Message(role="user", content="message fact")],
        metadata={"scenario": "scenario fact"},
    )
    fields = {field.value: getattr(case, field.value) for field in metric.eval_inputs.fields}
    projected = EvalCase(**{"input": "", "output": "", **fields})

    assert metric.measure(projected).value == metric.measure(case).value
    assert llm.prompts[0] == llm.prompts[1]
    assert all(text in llm.prompts[0] for text in ("context fact", "expected fact", "message fact", "scenario fact"))


@pytest.mark.unit
def test_structural_operands_form_a_cartesian_product_of_paths_and_fallbacks() -> None:
    metric = StructuralSimilarityMetric(output_field="metadata.actual", expected_field="tags.expected")

    assert metric.eval_inputs.required == frozenset()
    assert set(metric.eval_inputs.any_of) == {
        frozenset({METADATA, TAGS}),
        frozenset({METADATA, EXPECTED}),
        frozenset({OUTPUT, TAGS}),
        frozenset({OUTPUT, EXPECTED}),
    }
    assert metric.eval_inputs.fields == {OUTPUT, EXPECTED, METADATA, TAGS}
    assert not metric.applicability.online_scopes


@pytest.mark.unit
@pytest.mark.parametrize("actual_present, expected_present", tuple(product((False, True), repeat=2)))
def test_structural_paths_and_string_fallbacks_score_identically(actual_present: bool, expected_present: bool) -> None:
    metric = StructuralSimilarityMetric(
        format="json", output_field="metadata.actual", expected_field="metadata.expected"
    )
    metadata = {}
    if actual_present:
        metadata["actual"] = {"a": 1}
    if expected_present:
        metadata["expected"] = {"a": 1}
    case = EvalCase(input="q", output='{"a": 1}', expected='{"a": 1}', metadata=metadata)

    assert metric.measure(case).value == 1.0


@pytest.mark.unit
@pytest.mark.parametrize("level", ["raw", "structural", "schema_validated"])
def test_structural_levels_still_require_both_operands(level: str) -> None:
    metric = StructuralSimilarityMetric(level=level)
    assert metric.eval_inputs == MetricInputs({OUTPUT, EXPECTED})
    score = metric.measure(EvalCase(input="q", output={"a": 1}))
    assert score.value == 0.0
    assert "missing" in score.reason


@pytest.mark.unit
@pytest.mark.parametrize("path", ["$..actual", "$.*", "unknown.actual"])
def test_structural_unknown_path_fails_closed_even_with_a_fallback(path: str) -> None:
    assert StructuralSimilarityMetric(output_field=path).eval_inputs == MetricInputs.unknown()


def _composite(check: dict[str, Any], **sub_options: Any) -> CompositeMetric:
    return CompositeMetric(sub_scores=[{"name": "check", "weight": 1.0, "check": check, **sub_options}])


@pytest.mark.unit
@pytest.mark.parametrize(
    "check, value",
    [
        ({"type": "field_exists", "field": "metadata.absent"}, 0.0),
        ({"type": "equals", "field": "metadata.absent", "value": None}, 1.0),
        ({"type": "equals", "field": "metadata.absent", "value": "x"}, 0.0),
        ({"type": "count_match", "field": "metadata.absent", "expected_field": "expected.items"}, 1.0),
        ({"type": "contains", "field": "metadata.absent", "value": "x"}, 0.0),
        ({"type": "contains_all", "field": "metadata.absent", "values": ["x"]}, 0.0),
        ({"type": "regex_match", "field": "metadata.absent", "pattern": "x"}, 0.0),
        ({"type": "range", "field": "metadata.absent", "min": 0}, 0.0),
        ({"type": "length", "field": "metadata.absent", "min_length": 1}, 0.0),
    ],
)
def test_composite_operators_that_score_absence_only_require_parsed_output(check: dict[str, Any], value: float) -> None:
    metric = _composite(check)
    assert metric.eval_inputs == MetricInputs({OUTPUT}, {EXPECTED, METADATA})
    assert metric.applicability.online_scopes == {OnlineScope.TRACE, OnlineScope.SPAN}
    score = metric.measure(EvalCase(input="q", output="{}"))
    assert score.value == value
    assert score.metadata["sub_scores"]["check"]["status"] == "ok"
    assert metric.measure(EvalCase(input="q", output="not json")).value == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("operator", ["json_schema", "unique_ratio", "list_field_match", "deep_diff"])
def test_composite_operators_requiring_operands_declare_them(operator: str) -> None:
    check = {
        "type": operator,
        "field": "metadata.actual",
        "output_field": "metadata.actual",
        "expected_field": "tags.expected",
    }
    metric = _composite(check)
    required = {OUTPUT, METADATA}
    if operator in ("list_field_match", "deep_diff"):
        required.add(TAGS)
    assert metric.eval_inputs == MetricInputs(required, {EXPECTED})


@pytest.mark.unit
def test_composite_literal_ignores_an_irrelevant_unknown_expected_path(monkeypatch: pytest.MonkeyPatch) -> None:
    original = operators.extract_path
    paths: list[str] = []

    def recording_extract(data: dict[str, Any], path: str) -> Any:
        paths.append(path)
        return original(data, path)

    monkeypatch.setattr(operators, "extract_path", recording_extract)
    metric = _composite({"type": "equals", "field": "tags.value", "value": "a", "expected_field": "$..anything"})
    assert metric.eval_inputs == MetricInputs({OUTPUT}, {EXPECTED, TAGS})
    assert metric.measure(EvalCase(input="q", output="{}", tags={"value": "a"})).value == 1.0
    assert paths == ["tags.value"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "check",
    [
        {"type": "unknown", "field": "output.x"},
        {"type": "field_exists", "field": "$..anything"},
        {"type": "field_exists"},
    ],
)
def test_composite_unknown_operator_or_read_path_is_conservative(check: dict[str, Any]) -> None:
    assert _composite(check).eval_inputs == MetricInputs.unknown()


@pytest.mark.unit
def test_composite_skippable_operand_is_optional_but_materialized() -> None:
    metric = _composite({"type": "json_schema", "field": "metadata.value"}, skip_when_missing=True)
    assert metric.eval_inputs == MetricInputs({OUTPUT}, {EXPECTED, METADATA})
    score = metric.measure(EvalCase(input="q", output="{}"))
    assert score.metadata["sub_scores"]["check"]["status"] == "skipped"


_PRIMITIVES = [
    (ChoiceMetric, ChoiceQuestion(instructions="?", criteria={"a": None, "b": None})),
    (ScoreMetric, ScoreQuestion(instructions="?", criteria=["low", "medium", "high"])),
    (NoulMetric, NoulQuestion(instructions="?")),
]
_INVALID_EXPECTED = [
    *[(ChoiceMetric, _PRIMITIVES[0][1], value) for value in ("unknown", True, 1, [], {})],
    *[(ScoreMetric, _PRIMITIVES[1][1], value) for value in ("high", True, -1, 3, float("nan"), float("inf"))],
    *[(NoulMetric, _PRIMITIVES[2][1], value) for value in ("yes", 2, 1.0, [], {})],
]


def _primitive(
    metric_class: type[BaseMetric], question: Question, provider: RecordingProvider, **kwargs: Any
) -> BaseMetric:
    options: dict[str, Any] = {"instructions": question.instructions}
    if isinstance(question, ChoiceQuestion | ScoreQuestion):
        options["criteria"] = question.criteria
    return metric_class(provider=provider, **options, **kwargs)


@pytest.mark.unit
@pytest.mark.parametrize("metric_class, question, expected", _INVALID_EXPECTED)
@pytest.mark.parametrize("mode", [None, "correctness"])
async def test_standalone_invalid_expected_never_calls_provider(
    metric_class: type[BaseMetric], question: Question, expected: Any, mode: str | None
) -> None:
    provider = RecordingProvider()
    metric = _primitive(metric_class, question, provider, mode=mode)
    with pytest.raises(ValueError):
        await metric.a_measure(EvalCase(input="q", output="answer", expected=expected))
    assert provider.calls == []


@pytest.mark.unit
@pytest.mark.parametrize("metric_class, question, expected", _INVALID_EXPECTED)
@pytest.mark.parametrize("mode", [None, "correctness"])
async def test_composite_invalid_expected_never_calls_provider(
    metric_class: type[BaseMetric], question: Question, expected: Any, mode: str | None
) -> None:
    provider = RecordingProvider()
    metric = DecisionCompositeMetric(
        provider=provider,
        sub_scores=[
            {
                "name": "check",
                "weight": 1.0,
                "state_field": "output",
                "question": question,
                "mode": mode,
                "expected_field": "expected",
            }
        ],
    )
    score = await metric.a_measure(EvalCase(input="q", output="answer", expected=expected))
    assert score.metadata["sub_scores"]["check"]["status"] == "error"
    assert provider.calls == []


@pytest.mark.unit
@pytest.mark.parametrize("metric_class, question", _PRIMITIVES)
async def test_missing_correctness_expected_and_invalid_mode_never_call_provider(
    metric_class: type[BaseMetric], question: Question
) -> None:
    for mode in ("correctness", "invalid"):
        provider = RecordingProvider()
        metric = _primitive(metric_class, question, provider, mode=mode)
        with pytest.raises(ValueError):
            await metric.a_measure(EvalCase(input="q", output="answer"))
        assert provider.calls == []
        composite = DecisionCompositeMetric(
            provider=provider,
            sub_scores=[
                {
                    "name": "check",
                    "state_field": "output",
                    "question": question,
                    "mode": mode,
                    "expected_field": "expected",
                }
            ],
        )
        score = await composite.a_measure(EvalCase(input="q", output="answer"))
        assert score.metadata["sub_scores"]["check"]["status"] == "error"
        assert provider.calls == []


@pytest.mark.unit
@pytest.mark.parametrize("metric_class, question", _PRIMITIVES)
def test_correctness_composite_without_expected_field_is_rejected_at_construction(
    metric_class: type[BaseMetric], question: Question
) -> None:
    provider = RecordingProvider()
    with pytest.raises(ValueError, match="requires an expected_field"):
        DecisionCompositeMetric(
            provider=provider,
            sub_scores=[
                {
                    "name": "check",
                    "state_field": "output",
                    "question": question,
                    "mode": "correctness",
                }
            ],
        )
    assert provider.calls == []


@pytest.mark.unit
@pytest.mark.parametrize("metric_class, question", _PRIMITIVES)
@pytest.mark.parametrize("expected, mode", [(None, None), ({"ignored": True}, "confidence"), ("a", "confidence")])
async def test_none_infers_confidence_and_explicit_confidence_ignores_expected(
    metric_class: type[BaseMetric], question: Question, expected: Any, mode: str | None
) -> None:
    provider = RecordingProvider()
    metric = _primitive(metric_class, question, provider, mode=mode)
    score = await metric.a_measure(EvalCase(input="q", output="answer", expected=expected))
    assert score.value == pytest.approx(0.8)
    assert score.metadata["mode"] == "confidence"
    composite = DecisionCompositeMetric(
        provider=provider,
        sub_scores=[
            {
                "name": "check",
                "weight": 1.0,
                "state_field": "output",
                "expected_field": "expected",
                "question": question,
                "mode": mode,
            }
        ],
    )
    composite_score = await composite.a_measure(EvalCase(input="q", output="answer", expected=expected))
    assert composite_score.value == pytest.approx(0.8)
    assert composite_score.metadata["sub_scores"]["check"]["metadata"]["mode"] == "confidence"
    assert len(provider.calls) == 2


@pytest.mark.unit
@pytest.mark.parametrize("expected", [True, False, 0, 1, "true", "FALSE", " True "])
async def test_supported_noul_labels_have_identical_standalone_and_composite_scores(expected: Any) -> None:
    provider = RecordingProvider()
    standalone = NoulMetric(provider=provider, instructions="?")
    composite = DecisionCompositeMetric(
        provider=provider,
        sub_scores=[
            {
                "name": "check",
                "weight": 1.0,
                "state_field": "output",
                "expected_field": "expected",
                "question": NoulQuestion(instructions="?"),
            }
        ],
    )
    case = EvalCase(input="q", output="answer", expected=expected)
    assert (await standalone.a_measure(case)).value == (await composite.a_measure(case)).value
    assert len(provider.calls) == 2


@pytest.mark.unit
@pytest.mark.parametrize("expected", [0, 1, 1.5, 2])
async def test_fractional_rubric_behavior_is_preserved_in_both_metrics(expected: float) -> None:
    provider = RecordingProvider()
    standalone = ScoreMetric(provider=provider, instructions="?", criteria=["low", "medium", "high"])
    composite = DecisionCompositeMetric(
        provider=provider,
        sub_scores=[
            {
                "name": "check",
                "weight": 1.0,
                "state_field": "output",
                "expected_field": "expected",
                "question": ScoreQuestion(instructions="?", criteria=["low", "medium", "high"]),
            }
        ],
    )
    case = EvalCase(input="q", output="answer", expected=expected)
    expected_score = 1.0 - abs(1.3 / 2 - expected / 2)
    assert (await standalone.a_measure(case)).value == pytest.approx(expected_score)
    assert (await composite.a_measure(case)).value == pytest.approx(expected_score)


@pytest.mark.unit
async def test_composite_filters_invalid_subchecks_before_any_group_call() -> None:
    provider = RecordingProvider()
    metric = DecisionCompositeMetric(
        provider=provider,
        sub_scores=[
            {"name": "valid", "weight": 1.0, "state_field": "output", "question": NoulQuestion(instructions="?")},
            {
                "name": "invalid_shared",
                "weight": 1.0,
                "state_field": "output",
                "expected_field": "expected",
                "question": NoulQuestion(instructions="?"),
            },
            {
                "name": "invalid_other",
                "weight": 1.0,
                "state_field": "input",
                "expected_field": "expected",
                "question": ScoreQuestion(instructions="?", criteria=["low", "high"]),
            },
        ],
    )
    score = await metric.a_measure(EvalCase(input="q", output="answer", expected="invalid"))
    assert len(provider.calls) == 1
    assert set(provider.calls[0][1]) == {"valid"}
    assert score.metadata["sub_scores"]["invalid_shared"]["status"] == "error"
    assert score.metadata["sub_scores"]["invalid_other"]["status"] == "error"
    assert score.value == pytest.approx(0.8 / 3)


@pytest.mark.unit
@pytest.mark.parametrize("metric_class", [HallucinationMetric, StructuralSimilarityMetric, CompositeMetric])
def test_configured_declarations_preserve_unknown_and_explicit_custom_subclasses(
    metric_class: type[BaseMetric],
) -> None:
    options = {
        HallucinationMetric: {"llm": MockLLM(), "include_messages_as_reference": True},
        StructuralSimilarityMetric: {"output_field": "tags.actual"},
        CompositeMetric: {"sub_scores": [{"name": "check", "check": {"type": "field_exists", "field": "tags.x"}}]},
    }[metric_class]

    class Custom(metric_class):
        def measure(self, eval_case: EvalCase) -> Score:
            return Score(name=self.name, value=1.0, threshold=self.threshold)

    declared = MetricInputs({METADATA}, any_of=({CONTEXT}, {EXPECTED}))

    class Explicit(Custom):
        eval_inputs: MetricInputs = declared

    assert Custom(**options).eval_inputs == MetricInputs.unknown()
    assert Explicit(**options).eval_inputs is declared
