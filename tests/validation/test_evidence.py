"""Admission tests deliberately disable network and scoring entry points."""

from __future__ import annotations

import json
import socket
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import httpx
import pytest

from harness_evals.adapters.group import (
    GroupDiagnostics,
    GroupIssue,
    GroupIssueCount,
    GroupUsageCoverage,
)
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import EvaluationMode, MetricInputs, OnlineScope
from harness_evals.core.types import Message, ToolCall
from harness_evals.decision.base import BaseDecisionProvider
from harness_evals.llm.base import BaseLLM
from harness_evals.metrics.composite.composite import CompositeMetric
from harness_evals.metrics.composite.decision_composite import DecisionCompositeMetric
from harness_evals.metrics.composite.operators import OPERATORS
from harness_evals.metrics.operational.turn_latency import TurnLatencyMetric
from harness_evals.metrics.operational.turn_token_cost import TurnTokenCostMetric
from harness_evals.metrics.safety.hallucination import HallucinationMetric
from harness_evals.metrics.structural.structural_similarity import StructuralSimilarityMetric
from harness_evals.validation.evidence import validate_metric
from harness_evals.validation.types import (
    ConfiguredMetricContract,
    EvidenceRequirement,
    ExecutionKind,
    ReasonCode,
    ValidationContext,
    ValidationReason,
)

pytestmark = pytest.mark.unit


def _explode(*args, **kwargs):
    raise AssertionError("Admission must not perform I/O or scoring")


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    monkeypatch.setattr(socket, "socket", _explode)
    monkeypatch.setattr(socket, "create_connection", _explode)
    monkeypatch.setattr(httpx, "Client", _explode)
    monkeypatch.setattr(httpx, "AsyncClient", _explode)
    for cls, names in (
        (BaseLLM, ("generate", "generate_json")),
        (BaseDecisionProvider, ("ask", "a_ask")),
        (HallucinationMetric, ("measure", "a_measure")),
        (CompositeMetric, ("measure", "a_measure")),
        (DecisionCompositeMetric, ("measure", "a_measure")),
        (StructuralSimilarityMetric, ("measure", "a_measure")),
        (TurnLatencyMetric, ("measure",)),
        (TurnTokenCostMetric, ("measure",)),
    ):
        for name in names:
            monkeypatch.setattr(cls, name, _explode)
    for name in OPERATORS:
        monkeypatch.setitem(OPERATORS, name, _explode)


def field(path="output", predicate="present", **parameters):
    return EvidenceRequirement(path=path, predicate=predicate, parameters=parameters)


def contract(rule=None, **kwargs):
    return ConfiguredMetricContract(
        kind=kwargs.pop("kind", "test"), inputs=MetricInputs(), requirements=rule or field(), **kwargs
    )


def codes(result):
    return {reason.code for reason in result.reasons}


def case(**kwargs):
    return EvalCase(input=kwargs.pop("input", "query"), output=kwargs.pop("output", "answer"), **kwargs)


def group_diagnostics(**kwargs):
    values = dict(
        member_count=2,
        span_count=2,
        duplicate_span_count=0,
        message_count=0,
        tool_execution_count=0,
        complete_member_count=2,
        incomplete_member_count=0,
        unknown_member_count=0,
        issues=(),
        input_tokens=GroupUsageCoverage(),
        output_tokens=GroupUsageCoverage(),
        cost_usd=GroupUsageCoverage(),
    )
    values.update(kwargs)
    return GroupDiagnostics(**values)


def group_contract(rule=None, **kwargs):
    return contract(
        rule,
        modes=frozenset({EvaluationMode.ONLINE}),
        online_scopes=frozenset({OnlineScope.GROUP}),
        group_compatible=True,
        **kwargs,
    )


def group_context(diagnostics=None):
    return ValidationContext(mode=EvaluationMode.ONLINE, scope=OnlineScope.GROUP, group_diagnostics=diagnostics)


def test_context_only_and_resource_assertions():
    configured = contract(resources=("llm",))
    result = validate_metric(configured)
    assert result.eligible and not result.evidence_checked and not result.ready
    assert codes(result) == {ReasonCode.PROVIDER_NOT_CHECKED}
    assert result.reasons[0].severity == "warning"
    assert validate_metric(configured, case()).ready
    absent = validate_metric(configured, case(), context=ValidationContext(available_resources=frozenset()))
    assert not absent.eligible and not absent.evidence_checked
    assert codes(absent) == {ReasonCode.PROVIDER_UNAVAILABLE}
    assert validate_metric(configured, case(), context=ValidationContext(available_resources=frozenset({"llm"}))).ready


class ExplodingCase(EvalCase):
    def __getattribute__(self, name):
        raise AssertionError("An ineligible case was inspected")


@pytest.mark.parametrize(
    "context,code",
    [
        (ValidationContext(mode="invalid"), ReasonCode.UNSUPPORTED_MODE),
        (ValidationContext(scope=OnlineScope.TRACE), ReasonCode.UNSUPPORTED_SCOPE),
        (ValidationContext(mode=EvaluationMode.ONLINE), ReasonCode.UNSUPPORTED_MODE),
        (ValidationContext(scope="secret-scope"), ReasonCode.UNSUPPORTED_SCOPE),
        (ValidationContext(execution="invalid"), ReasonCode.UNSUPPORTED_EXECUTION),
        (ValidationContext(execution=ExecutionKind.DATASET), ReasonCode.UNSUPPORTED_EXECUTION),
        (ValidationContext(available_resources={"llm"}), ReasonCode.INVALID_CONFIGURATION),
    ],
)
def test_ineligible_context_never_reads_case(context, code):
    result = validate_metric(contract(), object.__new__(ExplodingCase), context=context)
    assert not result.ready and not result.evidence_checked
    assert code in codes(result)


def test_unreviewed_and_configuration_errors_precede_evidence():
    for configured in (
        contract(reviewed=False),
        contract(reasons=(ValidationReason(ReasonCode.INVALID_CONFIGURATION),)),
    ):
        result = validate_metric(configured, object.__new__(ExplodingCase))
        assert not result.eligible and not result.evidence_checked


def test_online_scope_is_required_and_exact():
    configured = contract(modes=frozenset({EvaluationMode.ONLINE}), online_scopes=frozenset({OnlineScope.TRACE}))
    for scope in (None, "invalid", OnlineScope.SPAN):
        assert not validate_metric(configured, case(), context=ValidationContext(mode="online", scope=scope)).ready
    assert validate_metric(configured, case(), context=ValidationContext(mode="online", scope="trace")).ready


@pytest.mark.parametrize(
    "predicate,value",
    [
        ("present", []),
        ("sequence", []),
        ("tool_calls", []),
        ("string_list", []),
        ("string", ""),
        ("nonnegative_number", 0),
        ("nonnegative_integer", 0),
        ("nonnegative_integer", 2.0),
        ("probability", 0),
        ("probability", 1),
        ("messages", [Message(role="assistant", tool_calls=[ToolCall("tool")])]),
        ("tool_calls", [ToolCall("tool", input={}, output={})]),
    ],
)
def test_valid_empty_zero_and_typed_evidence(predicate, value):
    assert validate_metric(contract(field("metadata.value", predicate)), case(metadata={"value": value})).ready


@pytest.mark.parametrize(
    "predicate,value",
    [
        ("nonnegative_number", True),
        ("nonnegative_number", float("nan")),
        ("nonnegative_number", float("inf")),
        ("nonnegative_number", -1),
        ("nonnegative_number", "0"),
        ("nonnegative_integer", 1.1),
        ("nonnegative_integer", False),
        ("probability", 1.01),
        ("nonempty", []),
        ("nonempty_text", "  "),
        ("string_list", [1]),
        ("tool_calls", ["tool"]),
        ("tool_calls", [ToolCall("tool", input="invalid")]),
        ("messages", [Message("assistant", content=42)]),
        ("messages", [Message("assistant", tool_calls=[ToolCall(42)])]),
        ("messages", [Message("assistant", token_count=1.5)]),
        ("messages", [Message("assistant", latency_ms=True)]),
        ("tool_calls", [ToolCall("")]),
    ],
)
def test_invalid_values_fail_safely(predicate, value):
    result = validate_metric(contract(field("metadata.value", predicate)), case(metadata={"value": value}))
    assert not result.ready and codes(result) == {ReasonCode.INVALID_EVIDENCE}


@pytest.mark.parametrize("predicate", ["present", "sequence", "tool_calls", "string_list"])
def test_none_is_unknown_not_known_empty(predicate):
    result = validate_metric(contract(field("metadata.value", predicate)), case(metadata={"value": None}))
    assert codes(result) == {ReasonCode.MISSING_EVIDENCE}


def test_minimum_missing_policies_and_bad_paths():
    assert codes(validate_metric(contract(field("context", "sequence", min_items=2)), case(context=[]))) == {
        ReasonCode.INSUFFICIENT_EVIDENCE
    }
    skipped = validate_metric(contract(field("context", missing_policy="skip")), case())
    assert skipped.ready and codes(skipped) == {ReasonCode.PARTIAL_EVIDENCE}
    assert validate_metric(contract(field("context", missing_policy="score_absence")), case()).ready
    for requirement in (
        field("$[bad"),
        field(None),
        field("output", "not-a-predicate"),
        field("output", min_items=True),
        field("output", missing_policy="bad"),
    ):
        assert codes(validate_metric(contract(requirement), case())) == {ReasonCode.INVALID_CONFIGURATION}
    assert validate_metric(contract(field("$.metadata.items[0].value")), case(metadata={"items": [{"value": 0}]})).ready


def test_alternatives_suppress_failed_branch_errors_but_all_combines():
    left, right = field("expected"), field("context", "string_list")
    either = contract(EvidenceRequirement(op="any", children=(left, right)))
    assert validate_metric(either, case(context=[])).ready
    assert not validate_metric(either, case()).ready
    both = contract(EvidenceRequirement(op="all", children=(left, right)))
    rejected = validate_metric(both, case())
    assert [reason.requirement_id for reason in rejected.reasons] == ["r.0", "r.1"]
    assert not validate_metric(contract(EvidenceRequirement(op="any")), case()).ready


def test_fallback_preserves_primary_and_string_only_default_even_empty():
    rule = EvidenceRequirement(
        op="fallback",
        path="metadata.actual",
        predicate="present",
        parameters={"default_path": "output", "fallback_string_only": True},
    )
    configured = contract(rule)
    assert validate_metric(configured, case(output="")).ready
    assert validate_metric(configured, case(output={}, metadata={"actual": []})).ready
    assert not validate_metric(configured, case(output={})).ready
    strict = replace(
        configured,
        requirements=EvidenceRequirement(
            op="fallback", path="metadata.actual", predicate="string", parameters={"default_path": "output"}
        ),
    )
    assert not validate_metric(strict, case(output="valid", metadata={"actual": []})).ready


@pytest.mark.parametrize(
    "metric,name,value",
    [
        ("turn_latency", "latency_ms", 0),
        ("turn_token_cost", "token_count", 0),
    ],
)
def test_assistant_telemetry_actual_nested_values_and_partials(metric, name, value):
    configured = contract(EvidenceRequirement(op="assistant_turns", parameters={"metric": metric}))
    assert not validate_metric(configured, case(messages=[Message("user", metadata={name: value})])).ready
    assert not validate_metric(configured, case(messages=[Message("assistant")], **{name: 10})).ready
    message = Message("assistant", **{name: value})
    assert validate_metric(configured, case(messages=[message])).ready
    partial = validate_metric(configured, case(messages=[message, Message("assistant")]))
    assert partial.ready and codes(partial) == {ReasonCode.PARTIAL_EVIDENCE}
    metadata_only = validate_metric(configured, case(messages=[Message("assistant", metadata={name: value})]))
    assert metadata_only.ready is (metric == "turn_latency")
    if metric == "turn_token_cost":
        assert codes(metadata_only) == {ReasonCode.INSUFFICIENT_EVIDENCE}
    for value in (True, -1, float("nan"), "0"):
        assert not validate_metric(configured, case(messages=[Message("assistant", **{name: value})])).ready


@pytest.mark.parametrize(
    "metric", ["turn_faithfulness", "turn_contextual_precision", "turn_contextual_recall", "turn_contextual_relevancy"]
)
@pytest.mark.parametrize("allow_skips", [False, True])
def test_turn_rag_nested_sources_and_partial_policy(metric, allow_skips):
    configured = contract(
        EvidenceRequirement(op="assistant_turns", parameters={"metric": metric, "allow_skips": allow_skips})
    )
    usable = [Message("user", "query"), Message("assistant", "answer", retrieval_context=["chunk"], expected="label")]
    assert validate_metric(configured, case(messages=usable)).ready
    partial = validate_metric(configured, case(messages=[*usable, Message("assistant")]))
    assert partial.ready and codes(partial) == {ReasonCode.PARTIAL_EVIDENCE}
    assert not validate_metric(configured, case(messages=[Message("assistant", "answer")], context=["top-level"])).ready
    assert not validate_metric(configured, case(messages=[Message("assistant", "answer", retrieval_context=[1])])).ready


def test_turn_rag_requires_actual_query_expected_and_content_where_used():
    for metric, message in (
        ("turn_faithfulness", Message("assistant", retrieval_context=["chunk"])),
        ("turn_contextual_precision", Message("assistant", "answer", retrieval_context=["chunk"])),
        ("turn_contextual_recall", Message("assistant", "answer", retrieval_context=["chunk"])),
    ):
        configured = contract(EvidenceRequirement(op="assistant_turns", parameters={"metric": metric}))
        assert not validate_metric(configured, case(messages=[message])).ready


@pytest.mark.parametrize(
    "parameters,evidence",
    [
        ({}, {"context": ["chunk"]}),
        ({}, {"expected": "reference"}),
        ({"include_messages_as_reference": True}, {"messages": [Message("user", "reference")]}),
        (
            {"include_assistant_tool_inputs_as_reference": True},
            {"messages": [Message("assistant", tool_calls=[ToolCall("tool", input={})])]},
        ),
        (
            {"include_assistant_tool_results_as_reference": True},
            {"messages": [Message("assistant", tool_calls=[ToolCall("tool", output={})])]},
        ),
        ({"include_assistant_tool_results_as_reference": True}, {"messages": [Message("tool", "reference")]}),
        ({"include_scenario_metadata_as_reference": True}, {"metadata": {"scenario": "reference"}}),
        ({"include_scenario_metadata_as_reference": True}, {"metadata": {"expected_outcome": "reference"}}),
        (
            {"include_sse_events_as_reference": True, "sse_reference_events": ["selected"]},
            {"metadata": {"sse_events": {"selected": [{}]}}},
        ),
    ],
)
def test_hallucination_enabled_actual_reference_alternatives(parameters, evidence):
    configured = contract(EvidenceRequirement(op="hallucination_references", parameters=parameters))
    assert validate_metric(configured, case(**evidence)).ready


@pytest.mark.parametrize(
    "evidence",
    [
        {"messages": [Message("assistant", "answer")]},
        {"context": []},
        {"expected": ""},
        {"metadata": {"arbitrary": "reference"}},
        {"tool_calls": [ToolCall("tool", output="reference")]},
        {"metadata": {"sse_events": {"unselected": ["reference"]}}},
    ],
)
def test_hallucination_does_not_invent_references(evidence):
    parameters = {
        "include_messages_as_reference": True,
        "include_assistant_tool_results_as_reference": True,
        "include_scenario_metadata_as_reference": True,
        "include_sse_events_as_reference": True,
        "sse_reference_events": ["selected"],
    }
    assert not validate_metric(
        contract(EvidenceRequirement(op="hallucination_references", parameters=parameters)), case(**evidence)
    ).ready


def decision(question="choice", **parameters):
    defaults = {
        "question_type": question,
        "criteria": {"yes": "description", "no": None}
        if question == "choice"
        else ["low", "high"]
        if question == "score"
        else None,
        "state_field": "output",
    }
    defaults.update(parameters)
    return contract(EvidenceRequirement(op="decision", parameters=defaults))


@pytest.mark.parametrize(
    "question,valid,invalid",
    [
        ("choice", "yes", "secret-invalid-label"),
        ("score", 0.5, 2),
        ("noul", False, "yes"),
    ],
)
def test_decision_reuses_expected_validators_before_providers(question, valid, invalid):
    assert validate_metric(decision(question), case(expected=valid)).ready
    bad = validate_metric(decision(question), case(expected=invalid))
    assert codes(bad) == {ReasonCode.INVALID_EVIDENCE}
    assert "secret-invalid-label" not in json.dumps(bad.to_dict())
    assert validate_metric(decision(question, mode="confidence"), case(expected=invalid)).ready
    assert not validate_metric(decision(question, mode="correctness"), case()).ready


def test_decision_configuration_explicit_expected_and_missing_state():
    assert validate_metric(decision(expected_field="metadata.label"), case(expected="invalid")).ready
    assert not validate_metric(
        decision(mode="correctness", expected_field="metadata.label"), case(expected="yes")
    ).ready
    assert validate_metric(decision(expected_field=None), case(expected="invalid")).ready
    skipped = validate_metric(decision(state_field="metadata.state", skip_when_missing=True), case())
    assert skipped.ready and codes(skipped) == {ReasonCode.PARTIAL_EVIDENCE}
    assert not validate_metric(decision(criteria=["yes"]), case()).ready
    assert not validate_metric(decision("score", criteria={"low": 0}), case()).ready
    assert not validate_metric(decision(mode="invalid"), case()).ready
    for question, criteria in (
        ("choice", {"yes": 1}),
        ("choice", {"": "description"}),
        ("score", ["one"]),
        ("score", [0, 1]),
        ("noul", {"positive": True}),
    ):
        result = validate_metric(decision(question, criteria=criteria), case())
        assert codes(result) == {ReasonCode.INVALID_CONFIGURATION}


def test_decision_children_require_at_least_one_non_skipped_usable_branch():
    children = tuple(
        decision(state_field=f"metadata.{name}", skip_when_missing=True, expected_field=None).requirements
        for name in ("first", "second")
    )
    configured = contract(EvidenceRequirement(op="all", children=children, parameters={"minimum_active": 1}))
    skipped = validate_metric(configured, case(metadata={}))
    assert not skipped.ready
    assert codes(skipped) == {ReasonCode.PARTIAL_EVIDENCE, ReasonCode.INSUFFICIENT_EVIDENCE}
    assert (
        next(reason for reason in skipped.reasons if reason.code == ReasonCode.INSUFFICIENT_EVIDENCE).severity
        == "error"
    )
    for state in (0, False, "", {}, []):
        partial = validate_metric(configured, case(metadata={"first": state}))
        assert partial.ready and codes(partial) == {ReasonCode.PARTIAL_EVIDENCE}
    assert validate_metric(configured, case(metadata={"first": 0, "second": False})).ready
    invalid_child = replace(
        configured,
        requirements=EvidenceRequirement(
            op="all",
            children=(children[0], decision(mode="correctness").requirements),
            parameters={"minimum_active": 1},
        ),
    )
    invalid = validate_metric(invalid_child, case(metadata={}))
    assert not invalid.ready and ReasonCode.INSUFFICIENT_EVIDENCE in codes(invalid)


@pytest.mark.parametrize("minimum", [True, -1, 1.5, "1", None])
def test_malformed_minimum_active_fails_before_case_access(minimum):
    configured = contract(EvidenceRequirement(op="all", children=(field(),), parameters={"minimum_active": minimum}))
    result = validate_metric(configured, object.__new__(ExplodingCase))
    assert not result.evidence_checked and codes(result) == {ReasonCode.INVALID_CONFIGURATION}


def structural(**parameters):
    return contract(EvidenceRequirement(op="structural", parameters={"format": "json", **parameters}))


@pytest.mark.parametrize("value", ["{}", "[]", "0", "false", "null", {}, [], 0, False])
def test_structural_parses_actual_operand_without_comparing_quality(value):
    assert validate_metric(structural(), case(output=value, expected={"different": True})).ready


@pytest.mark.parametrize("value", ["{", "", "plain text", '"legitimate JSON string"'])
def test_structural_rejects_actual_strings_remaining_after_parse(value):
    result = validate_metric(structural(), case(output=value, expected={}))
    assert not result.ready and codes(result) == {ReasonCode.INVALID_EVIDENCE}


def test_structural_selected_operands_exact_fallback_and_expected_parse_semantics():
    configured = structural(output_field="metadata.actual", expected_field="metadata.reference")
    assert validate_metric(configured, case(output="{}", expected="[]")).ready
    assert not validate_metric(configured, case(output={}, expected=[])).ready
    assert not validate_metric(configured, case(output="", expected="{}")).ready
    assert not validate_metric(configured, case(output="{}", expected="{}", metadata={"actual": "{"})).ready
    # A selected non-null value takes precedence even when falsey, and bad
    # default strings must not be parsed when they were not selected.
    assert validate_metric(
        configured, case(output="{", expected="Infinity", metadata={"actual": 0, "reference": False})
    ).ready
    assert validate_metric(configured, case(output="{}", metadata={"reference": {}})).ready
    assert validate_metric(structural(output_field="", expected_field=""), case(output={}, expected={})).ready
    assert validate_metric(structural(), case(output={}, expected="unparsed reference text")).ready
    assert validate_metric(structural(format="yaml"), case(output="key: [1]", expected={})).ready
    assert not validate_metric(structural(format="yaml"), case(output="[", expected={})).ready


def composite(*checks, output_format="json"):
    return contract(
        EvidenceRequirement(
            op="composite",
            parameters={
                "output_format": output_format,
                "sub_scores": [
                    {"name": f"check-{index}", "weight": 1, "check": check} for index, check in enumerate(checks)
                ],
            },
        )
    )


@pytest.mark.parametrize(
    "check",
    [
        {"type": "field_exists", "field": "output.missing"},
        {"type": "equals", "field": "output.value", "value": 1, "expected_field": "$[bad"},
        {"type": "count_match", "field": "output.missing", "expected_field": "expected.missing"},
        {"type": "range", "field": "output.missing"},
    ],
)
def test_composite_absence_observations_and_literal_precedence(check):
    assert validate_metric(composite(check), case(output={"value": 1})).ready


def test_composite_operands_parse_and_types_without_operator_execution():
    deep = composite({"type": "deep_diff", "output_field": "output.value", "expected_field": "expected.value"})
    assert validate_metric(deep, case(output='{"value": []}', expected='{"value": []}')).ready
    assert not validate_metric(deep, case(output='{"value": []}')).ready
    assert not validate_metric(deep, case(output="not-json", expected={})).ready
    assert validate_metric(
        composite({"type": "field_exists", "field": "output.value"}, output_format="yaml"), case(output="value: 1")
    ).ready
    for check in (
        {"type": "unknown"},
        {"type": "list_field_match", "field": "output"},
        {"type": "count_match", "field": "output.value"},
        {"type": "unique_ratio", "field": "output.items"},
    ):
        assert not validate_metric(composite(check), case(output={"value": 3, "items": [{}]})).ready


@pytest.mark.parametrize("engine", ["structural", "composite"])
@pytest.mark.parametrize("operand", ["output", "expected"])
@pytest.mark.parametrize(
    "output_format,value",
    [
        ("json", "NaN"),
        ("json", "Infinity"),
        ("json", "-Infinity"),
        ("json", '{"value": [NaN]}'),
        ("json", '{"value": [1e999]}'),
        ("json", {"value": [float("inf")]}),
        ("json", [float("nan")]),
        ("yaml", "value: [.nan]"),
        ("yaml", "value: [.inf]"),
        ("yaml", "value: [-.inf]"),
        ("yaml", {"value": [float("-inf")]}),
    ],
)
def test_structural_and_composite_reject_nested_nonfinite_values(engine, operand, output_format, value):
    configured = (
        structural(format=output_format)
        if engine == "structural"
        else composite({"type": "range", "field": "output.value", "min": 0}, output_format=output_format)
    )
    result = validate_metric(configured, case(**{"output": {"value": 1}, "expected": {}, operand: value}))
    assert not result.ready and codes(result) == {ReasonCode.INVALID_EVIDENCE}


def test_composite_rejects_nonfinite_unselected_native_values_before_operator_checks():
    configured = composite({"type": "field_exists", "field": "output.valid"})
    result = validate_metric(configured, case(output={"valid": 1, "other": [float("nan")]}))
    assert not result.ready and codes(result) == {ReasonCode.INVALID_EVIDENCE}
    # Failed assertions are still usable observations, not evidence errors.
    assert validate_metric(
        composite({"type": "range", "field": "output.value", "max": 0}), case(output={"value": 1})
    ).ready


def test_all_skipped_composite_has_no_usable_checks():
    rule = EvidenceRequirement(
        op="composite",
        parameters={
            "sub_scores": [
                {
                    "name": "hidden-name",
                    "weight": 1,
                    "skip_when_missing": True,
                    "check": {"type": "field_exists", "field": "output.absent"},
                }
            ]
        },
    )
    result = validate_metric(contract(rule), case(output={}))
    assert not result.ready and codes(result) == {ReasonCode.PARTIAL_EVIDENCE, ReasonCode.INSUFFICIENT_EVIDENCE}


def test_dataset_requires_real_cases_labels_and_confidence_values():
    configured = contract(field("confidence", "probability"), execution=ExecutionKind.DATASET)
    cases = [case(confidence=0), case(confidence=1)]
    context = ValidationContext(execution=ExecutionKind.DATASET, outcomes=(False, True))
    assert validate_metric(configured, cases, context=context).ready
    for outcomes in (None, (True,), [False, True], (0, 1)):
        assert not validate_metric(configured, cases, context=replace(context, outcomes=outcomes)).ready
    for confidence in (None, True, float("inf"), -1):
        assert not validate_metric(configured, [case(confidence=confidence), cases[1]], context=context).ready
    assert not validate_metric(configured, case(confidence=1), context=context).ready
    assert not validate_metric(
        replace(configured, kind="discrimination"), cases, context=replace(context, outcomes=(True, True))
    ).ready


def test_repeated_runs_exact_execution_minimum_and_nested_values():
    configured = contract(
        field("metadata.latency", "nonnegative_number"), execution=ExecutionKind.REPEATED_RUNS, min_runs=3
    )
    context = ValidationContext(execution=ExecutionKind.REPEATED_RUNS)
    parent = case(metadata={"latency": 0}, runs=[case(metadata={"latency": 0}) for _ in range(3)])
    assert validate_metric(configured, parent, context=context).ready
    assert not validate_metric(configured, replace(parent, runs=parent.runs[:2]), context=context).ready
    malformed = replace(parent, runs=[*parent.runs[:2], case(metadata={"latency": True})])
    assert not validate_metric(configured, malformed, context=context).ready
    assert not validate_metric(configured, parent).evidence_checked


def test_group_missing_diagnostics_and_explicit_review_precede_case_access():
    result = validate_metric(group_contract(), object.__new__(ExplodingCase), context=group_context())
    assert codes(result) == {ReasonCode.UNKNOWN_CAPTURE} and not result.evidence_checked
    excluded = replace(group_contract(field("messages", "sequence")), group_compatible=False)
    assert not validate_metric(
        excluded, object.__new__(ExplodingCase), context=group_context(group_diagnostics())
    ).ready


def test_group_known_empty_tools_complete_and_unknown_capture():
    rule = EvidenceRequirement(
        op="all", children=(field("tool_calls", "tool_calls"), field("expected_tools", "string_list"))
    )
    configured = group_contract(rule, group_requires_complete=True)
    observed = case(input="", output="", tool_calls=[], expected_tools=[])
    assert validate_metric(configured, observed, context=group_context(group_diagnostics())).ready
    assert not validate_metric(
        configured, replace(observed, tool_calls=None), context=group_context(group_diagnostics())
    ).ready
    for name in ("unknown_member_count", "incomplete_member_count"):
        diagnostics = group_diagnostics(complete_member_count=1, **{name: 1})
        assert codes(validate_metric(configured, observed, context=group_context(diagnostics))) == {
            ReasonCode.INCOMPLETE_CAPTURE
        }


@pytest.mark.parametrize(
    "issue",
    [
        GroupIssue.MISSING_TRACE_ID,
        GroupIssue.MISSING_SPAN_ID,
        GroupIssue.AMBIGUOUS_TOOL_LINK,
        GroupIssue.UNMATCHED_TOOL_RESULT,
        GroupIssue.AMBIGUOUS_MESSAGE_IDENTITY,
    ],
)
def test_group_strict_tool_identity_flags(issue):
    diagnostics = group_diagnostics(issues=(GroupIssueCount(issue, 1),))
    result = validate_metric(group_contract(group_requires_complete=True), case(), context=group_context(diagnostics))
    assert not result.ready and ReasonCode.AMBIGUOUS_TOOLS in codes(result)


@pytest.mark.parametrize("issue", [GroupIssue.CYCLIC_ORDER, GroupIssue.UNKNOWN_TIMESTAMP, GroupIssue.MISSING_PARENT])
def test_group_order_flags(issue):
    diagnostics = group_diagnostics(issues=(GroupIssueCount(issue, 1),))
    result = validate_metric(group_contract(group_requires_order=True), case(), context=group_context(diagnostics))
    assert not result.ready and ReasonCode.AMBIGUOUS_ORDER in codes(result)


def test_group_telemetry_partial_subtotals_do_not_require_complete_capture():
    configured = group_contract(EvidenceRequirement(op="assistant_turns", parameters={"metric": "turn_token_cost"}))
    diagnostics = group_diagnostics(
        message_count=2,
        complete_member_count=0,
        unknown_member_count=2,
        issues=(GroupIssueCount(GroupIssue.AMBIGUOUS_USAGE, 1),),
    )
    observed = case(input="", output="", messages=[Message("assistant", token_count=0), Message("assistant")])
    result = validate_metric(configured, observed, context=group_context(diagnostics))
    assert result.ready and codes(result) == {ReasonCode.PARTIAL_EVIDENCE, ReasonCode.AMBIGUOUS_USAGE}


@pytest.mark.parametrize(
    "updates",
    [
        {"member_count": True},
        {"complete_member_count": 3},
        {"unknown_member_count": -1},
        {"issues": (GroupIssueCount("unknown", 1),)},
        {"issues": (GroupIssueCount(GroupIssue.MISSING_PARENT, True),)},
        {"input_tokens": GroupUsageCoverage(observed=1, selected=2)},
        {"cost_usd": "not-coverage"},
    ],
)
def test_group_diagnostic_shape_and_assertion_consistency(updates):
    result = validate_metric(
        group_contract(), object.__new__(ExplodingCase), context=group_context(group_diagnostics(**updates))
    )
    assert not result.evidence_checked and ReasonCode.INVALID_EVIDENCE in codes(result)


def test_serialized_contracts_inputs_and_payload_safe_deterministic_concurrency():
    payload = "secret-source-id-label-exception"
    original = case(
        output={"value": 0},
        metadata={"private": payload},
        messages=[Message("assistant", tool_calls=[ToolCall("tool", input={"private": payload})])],
    )
    snapshot = original.to_dict()
    configured = contract(
        EvidenceRequirement(
            op="all",
            children=(
                field("messages", "messages"),
                field(f"metadata.{payload}"),
                EvidenceRequirement(op="decision", parameters={"question_type": "choice", "criteria": {"valid": None}}),
            ),
        )
    )
    serialized = json.loads(json.dumps(configured.to_dict(), allow_nan=False))
    restored = ConfiguredMetricContract.from_dict(serialized)
    typed = EvalCase.from_dict(json.loads(json.dumps(snapshot)))
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: validate_metric(restored, typed).to_dict(), range(12)))
    assert all(result == results[0] for result in results)
    assert original.to_dict() == snapshot == typed.to_dict()
    assert payload not in json.dumps(results)
    assert all(reason["requirement_id"].startswith("r.") for reason in results[0]["reasons"])


def test_validation_never_invokes_eval_case_utility_methods(monkeypatch):
    for name in ("to_dict", "output_as_str", "output_as_dict", "expected_as_str", "expected_as_dict"):
        monkeypatch.setattr(EvalCase, name, _explode)
    assert validate_metric(contract(field("tool_calls", "tool_calls")), case(tool_calls=[])).ready


@pytest.mark.parametrize(
    "root,value",
    [
        ("messages", [{"role": "assistant", "latency_ms": 0}]),
        ("messages", [Message("assistant", tool_calls=[{"name": "tool"}])]),
        ("messages", (Message("assistant"),)),
        ("tool_calls", [{"name": "tool"}]),
        ("expected_tool_calls", [{"name": "tool"}]),
        ("runs", [{"input": "query", "output": "answer"}]),
        ("runs", [case(runs=[{"input": "query", "output": "answer"}])]),
        ("runs", [case(messages=[{"role": "assistant"}])]),
        ("runs", [case(tool_calls=[{"name": "tool"}])]),
    ],
)
def test_requested_runtime_types_are_checked_before_snapshot(root, value):
    result = validate_metric(contract(field(root, "sequence")), case(**{root: value}))
    assert not result.ready and codes(result) == {ReasonCode.INVALID_EVIDENCE}


@pytest.mark.parametrize("root", ["tool_calls", "expected_tool_calls"])
def test_json_callers_can_explicitly_hydrate_runtime_tool_types(root):
    configured = contract(field(root, "tool_calls"))
    payload = {"input": "query", "output": "answer", root: [{"name": "tool"}]}
    assert not validate_metric(configured, EvalCase(**payload)).ready
    assert validate_metric(configured, EvalCase.from_dict(payload)).ready


def test_json_callers_can_explicitly_hydrate_messages_and_nested_tools():
    configured = contract(field("messages", "messages"))
    payload = {
        "input": "query",
        "output": "answer",
        "messages": [{"role": "assistant", "tool_calls": [{"name": "tool"}]}],
    }
    assert not validate_metric(configured, EvalCase(**payload)).ready
    assert validate_metric(configured, EvalCase.from_dict(payload)).ready


def test_only_projected_roots_are_read_or_snapshotted():
    class ProjectedCase(EvalCase):
        def __getattribute__(self, name):
            if name not in ("output", "__class__"):
                raise AssertionError("Unrequested case field was inspected")
            return super().__getattribute__(name)

    configured = contract(field("output", "string"))
    assert validate_metric(configured, ProjectedCase(input=object(), output="answer", metadata=object())).ready
    assert validate_metric(
        configured,
        case(
            metadata=object(),
            messages=[{"role": "assistant"}],
            tool_calls=[{"name": "tool"}],
            expected_tool_calls=[{"name": "tool"}],
            token_count=True,
            runs=[{}],
        ),
    ).ready
    requested_optional = replace(configured, inputs=MetricInputs(optional={"tool_calls"}))
    assert not validate_metric(requested_optional, case(tool_calls=[{"name": "tool"}])).ready


def test_partial_turns_retain_usable_turn_when_another_nested_value_is_invalid():
    configured = contract(EvidenceRequirement(op="assistant_turns", parameters={"metric": "turn_faithfulness"}))
    observed = case(
        messages=[
            Message("assistant", "answer", retrieval_context=["chunk"]),
            Message("assistant", "answer", retrieval_context=[1]),
        ]
    )
    result = validate_metric(configured, observed)
    assert result.ready and codes(result) == {ReasonCode.PARTIAL_EVIDENCE}


@pytest.mark.parametrize(
    "updates",
    [
        {"modes": None},
        {"modes": frozenset({"offline"})},
        {"resources": ([],)},
        {"execution": "case"},
        {"group_compatible": 1},
        {"online_scopes": None},
        {"inputs": None},
        {"inputs": {"required": []}},
        {"kind": []},
        {"requirements": None},
    ],
)
def test_malformed_contract_assertions_fail_before_case_access(updates):
    malformed = contract()
    # Exercise the admission boundary even if typed construction was bypassed.
    for name, value in updates.items():
        object.__setattr__(malformed, name, value)
    result = validate_metric(malformed, object.__new__(ExplodingCase))
    assert not result.evidence_checked and ReasonCode.INVALID_CONFIGURATION in codes(result)


@pytest.mark.parametrize(
    "kind,options,evidence",
    [
        ("hallucination", {"include_messages_as_reference": True}, case(messages=[Message("user", "reference")])),
        (
            "structural_similarity",
            {"output_field": "metadata.answer", "expected_field": "metadata.reference"},
            case(output="{}", expected="{}"),
        ),
        (
            "decision_choice",
            {"instructions": "choose", "criteria": {"yes": "affirmative", "no": None}},
            case(expected="yes"),
        ),
        (
            "composite",
            {
                "sub_scores": [
                    {"name": "exists", "weight": 1, "check": {"type": "field_exists", "field": "output.missing"}}
                ]
            },
            case(output={}),
        ),
    ],
)
def test_resolver_contracts_round_trip_into_evidence_engine(kind, options, evidence):
    from harness_evals.validation.contracts import resolve_metric_contract

    configured = resolve_metric_contract(kind, options)
    assert configured.reviewed, configured.to_dict()
    restored = ConfiguredMetricContract.from_dict(json.loads(json.dumps(configured.to_dict(), allow_nan=False)))
    assert restored.inputs == configured.inputs
    assert validate_metric(restored, EvalCase.from_dict(evidence.to_dict())).ready


@pytest.mark.parametrize(
    "attribute,value",
    [("op", []), ("path", []), ("predicate", []), ("children", [field()]), ("_parameters_json", "[]")],
)
def test_malformed_runtime_rule_containers_fail_before_case_access(attribute, value):
    requirement = field()
    object.__setattr__(requirement, attribute, value)
    result = validate_metric(contract(requirement), object.__new__(ExplodingCase))
    assert not result.evidence_checked and codes(result) == {ReasonCode.INVALID_CONFIGURATION}


def test_resolved_decision_composite_round_trip_rejects_all_skipped_children():
    from harness_evals.validation.contracts import resolve_metric_contract

    configured = resolve_metric_contract(
        "decision_composite",
        {
            "sub_scores": [
                {
                    "name": name,
                    "weight": 1,
                    "state_field": f"metadata.{name}",
                    "skip_when_missing": True,
                    "question": {"type": "choice", "instructions": "Choose", "criteria": {"yes": None}},
                }
                for name in ("first", "second")
            ]
        },
    )
    assert configured.reviewed
    restored = ConfiguredMetricContract.from_dict(json.loads(json.dumps(configured.to_dict(), allow_nan=False)))
    skipped = validate_metric(restored, case(metadata={}))
    assert not skipped.ready
    assert {ReasonCode.PARTIAL_EVIDENCE, ReasonCode.INSUFFICIENT_EVIDENCE} <= codes(skipped)
    partial = validate_metric(restored, case(metadata={"first": 0}))
    assert partial.ready and ReasonCode.PARTIAL_EVIDENCE in codes(partial)


def test_resolved_structural_round_trip_rejects_bad_actual_parse():
    from harness_evals.validation.contracts import resolve_metric_contract

    configured = resolve_metric_contract("structural_similarity", {"format": "json", "output_field": "metadata.actual"})
    assert configured.reviewed
    restored = ConfiguredMetricContract.from_dict(json.loads(json.dumps(configured.to_dict(), allow_nan=False)))
    result = validate_metric(restored, case(output="{", expected={}))
    assert not result.ready and codes(result) == {ReasonCode.INVALID_EVIDENCE}
    assert validate_metric(restored, case(output="{", metadata={"actual": {}}, expected={})).ready


def test_resolved_repeated_runs_validate_nested_evidence_without_treating_each_run_as_a_container():
    from harness_evals.validation.contracts import resolve_metric_contract

    configured = resolve_metric_contract("outcome_consistency")
    assert configured.reviewed
    context = ValidationContext(execution=ExecutionKind.REPEATED_RUNS)
    observed = case(input="", output="", runs=[case(output=""), case(output=[])])
    assert validate_metric(configured, observed, context=context).ready
    for invalid in (case(output=None), case(output="", tool_calls=[ToolCall(42)]), case(token_count=True)):
        result = validate_metric(configured, replace(observed, runs=[observed.runs[0], invalid]), context=context)
        assert not result.ready


@pytest.mark.parametrize("issue", list(GroupIssue))
def test_every_group_issue_has_explicit_strict_tool_handling(issue):
    blocked_identity = {
        GroupIssue.MISSING_TRACE_ID,
        GroupIssue.MISSING_SPAN_ID,
        GroupIssue.AMBIGUOUS_MESSAGE_IDENTITY,
        GroupIssue.AMBIGUOUS_TOOL_LINK,
        GroupIssue.UNMATCHED_TOOL_RESULT,
    }
    blocked_order = {
        GroupIssue.CYCLIC_ORDER,
        GroupIssue.MISSING_PARENT,
        GroupIssue.UNKNOWN_TIMESTAMP,
        GroupIssue.AMBIGUOUS_TOOL_ORDER,
    }
    blocked_capture = {GroupIssue.UNSUPPORTED_EVIDENCE}
    result = validate_metric(
        group_contract(group_requires_complete=True, group_requires_order=True),
        case(),
        context=group_context(group_diagnostics(issues=(GroupIssueCount(issue, 1),))),
    )
    assert result.ready is (issue not in blocked_identity | blocked_order | blocked_capture)
    if issue in blocked_identity:
        assert ReasonCode.AMBIGUOUS_TOOLS in codes(result)
    if issue in blocked_order:
        assert ReasonCode.AMBIGUOUS_ORDER in codes(result)


def test_resolved_group_tools_and_turn_telemetry_and_excluded_placeholders():
    from harness_evals.validation.contracts import resolve_metric_contract

    tools = resolve_metric_contract("tool_correctness")
    assert tools.reviewed and tools.group_compatible
    observed = case(input="", output="", tool_calls=[], expected_tools=[])
    assert validate_metric(tools, observed, context=group_context(group_diagnostics())).ready
    turns = resolve_metric_contract("turn_latency")
    observed.messages = [Message("assistant", latency_ms=0)]
    assert validate_metric(turns, observed, context=group_context(group_diagnostics(message_count=1))).ready
    excluded = resolve_metric_contract("exact_match")
    assert not validate_metric(excluded, observed, context=group_context(group_diagnostics())).evidence_checked


@pytest.mark.parametrize("field_name,counter", [("tool_calls", "tool_execution_count"), ("messages", "message_count")])
def test_group_case_cannot_claim_known_empty_when_diagnostics_observed_evidence(field_name, counter):
    configured = group_contract(field(field_name, "sequence"), group_requires_complete=True)
    observed = case(**{field_name: []})
    diagnostics = group_diagnostics(**{counter: 1})
    result = validate_metric(configured, observed, context=group_context(diagnostics))
    assert not result.ready and ReasonCode.INVALID_EVIDENCE in codes(result)


def test_active_weights_do_not_admit_only_zero_weight_usable_children():
    children = (
        field("context", missing_policy="skip"),
        field("output"),
    )
    configured = contract(
        EvidenceRequirement("all", children=children, parameters={"minimum_active": 1, "active_weights": [1, 0]})
    )
    restored = ConfiguredMetricContract.from_dict(json.loads(json.dumps(configured.to_dict())))
    assert not validate_metric(restored, case()).ready
    assert validate_metric(restored, case(context=["reference"])).ready
