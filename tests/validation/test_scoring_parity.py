"""Real evidence/admission/scorer parity without provider or network calls."""

from __future__ import annotations

import json
import socket
from copy import deepcopy
from typing import Any
from unittest.mock import patch

import pytest

from harness_evals.adapters.group import GroupAuthoredData, GroupTrace, traces_to_group_eval_case
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.types import Message, ToolCall
from harness_evals.metrics.agent.tool_argument_match import ToolArgumentMatchMetric
from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric
from harness_evals.metrics.composite.composite import CompositeMetric
from harness_evals.metrics.factory import build_metric
from harness_evals.metrics.mcp.trace_completeness import MCPTraceCompletenessMetric
from harness_evals.metrics.operational.turn_latency import TurnLatencyMetric
from harness_evals.metrics.operational.turn_token_cost import TurnTokenCostMetric
from harness_evals.metrics.structural.schema_validation import SchemaValidationMetric
from harness_evals.validation import (
    ConfiguredMetricContract,
    ReasonCode,
    ValidationContext,
    describe_configured_metric,
    resolve_metric_contract,
    validate_metric,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Admission/scorer parity must not access the network")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def roundtrip(kind: str, options: dict[str, Any] | None = None) -> ConfiguredMetricContract:
    configured = resolve_metric_contract(kind, options)
    return ConfiguredMetricContract.from_dict(json.loads(json.dumps(configured.to_dict(), allow_nan=False)))


@pytest.mark.parametrize("pair", ["exact", "subset", "subsequence"])
@pytest.mark.parametrize("arg_match", ["exact", "subset"])
@pytest.mark.parametrize("selective", [False, True])
@pytest.mark.parametrize("observed", [False, True])
def test_group_argument_comparison_requires_observed_arguments(
    pair: str, arg_match: str, selective: bool, observed: bool
) -> None:
    options = {"pair": pair, "arg_match": arg_match, "skip_when_missing": False}
    metric = ToolArgumentMatchMetric(**options)
    attrs: dict[str, Any] = {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": "lookup",
        "gen_ai.tool.call.id": "call",
    }
    if observed:
        attrs["gen_ai.tool.call.arguments"] = {}
    row = {"span_id": "execution", "start_timestamp": 1, "end_timestamp": 2, "attributes": attrs}
    original = deepcopy(row)
    assembled = traces_to_group_eval_case(
        [GroupTrace([row], "one", True), GroupTrace([], "two", True)],
        authored=GroupAuthoredData(expected_tool_calls=[ToolCall("lookup", {})], expected_tools=["lookup"]),
        metrics=[metric, ToolCorrectnessMetric()] if selective else None,
    )
    context = ValidationContext(mode="online", scope="group", group_diagnostics=assembled.diagnostics)
    with patch.object(ToolArgumentMatchMetric, "measure", side_effect=AssertionError("Gate called scorer")) as scorer:
        decision = validate_metric(roundtrip("tool_argument_match", options), assembled.case, context=context)
        scorer.assert_not_called()
    assert decision.ready is observed
    assert row == original
    assert assembled.case.tool_calls[0].input == ({} if observed else None)
    if observed:
        assert metric.measure(assembled.case).value == 1.0
    else:
        assert {reason.code for reason in decision.reasons} == {ReasonCode.INVALID_EVIDENCE}
    assert validate_metric(roundtrip("tool_correctness"), assembled.case, context=context).ready
    assert ToolCorrectnessMetric().measure(assembled.case).value == 1.0


@pytest.mark.parametrize("missing", ["actual", "authored"])
def test_argument_comparison_does_not_treat_unknown_arguments_as_empty(missing: str) -> None:
    case = EvalCase(
        input="",
        output="",
        tool_calls=[ToolCall("lookup", None if missing == "actual" else {})],
        expected_tool_calls=[ToolCall("lookup", None if missing == "authored" else {})],
    )
    decision = validate_metric(roundtrip("tool_argument_match"), case)
    assert not decision.ready
    assert {reason.code for reason in decision.reasons} == {ReasonCode.INVALID_EVIDENCE}


@pytest.mark.parametrize("expected", [[], [ToolCall("lookup", {})]])
def test_argument_comparison_retains_known_empty_execution_evidence(expected: list[ToolCall]) -> None:
    assembled = traces_to_group_eval_case(
        [GroupTrace([], "one", True), GroupTrace([], "two", True)],
        authored=GroupAuthoredData(expected_tool_calls=expected),
    )
    context = ValidationContext(mode="online", scope="group", group_diagnostics=assembled.diagnostics)
    assert assembled.case.tool_calls == []
    assert validate_metric(roundtrip("tool_argument_match"), assembled.case, context=context).ready
    assert ToolArgumentMatchMetric().measure(assembled.case).value == (0.0 if expected else 1.0)


@pytest.mark.parametrize(
    "schema,output",
    [
        ({"type": "object", "properties": {"$ref": {"type": "string"}}}, {"$ref": "literal property"}),
        ({"const": {"$ref": "https://invalid.example/literal"}}, {"$ref": "https://invalid.example/literal"}),
        ({"enum": [{"$ref": "file:///private/literal"}]}, {"$ref": "file:///private/literal"}),
        ({"$defs": {"$ref": {"type": "object"}}, "$ref": "#/$defs/$ref"}, {}),
        ({"$defs": {"node": {"$anchor": "local", "type": "object"}}, "$ref": "#local"}, {}),
        (
            {
                "$schema": "http://json-schema.org/draft-04/schema#",
                "definitions": {"node": {"id": "#node", "type": "object"}},
                "$ref": "#node",
            },
            {},
        ),
        ({"extension": {"type": "object"}, "$ref": "#/extension"}, {}),
        (
            {
                "$defs": {
                    "inner": {
                        "$id": "https://invalid.example/inner",
                        "$defs": {"target": {"type": "object"}},
                        "properties": {"foo": {"$ref": "#/$defs/target"}},
                    }
                },
                "$ref": "#/$defs/inner/properties/foo",
            },
            {},
        ),
    ],
)
def test_admitted_schema_property_names_and_literals_score_without_network(schema: dict, output: dict) -> None:
    case = EvalCase(input="", output=output)
    with patch.object(SchemaValidationMetric, "measure", side_effect=AssertionError("Gate called scorer")) as scorer:
        configured = roundtrip("schema_validation", {"schema": schema})
        assert configured.reviewed and validate_metric(configured, case).ready
        scorer.assert_not_called()
    assert SchemaValidationMetric(schema=schema).measure(case).value == 1.0


@pytest.mark.parametrize("expected", [None, {}, {"q": "value"}])
@pytest.mark.parametrize("actual", [None, {}, {"q": "value"}])
def test_mcp_completeness_requires_known_argument_objects(expected: dict | None, actual: dict | None) -> None:
    metric = MCPTraceCompletenessMetric([ToolCall("lookup", expected)])
    case = EvalCase(input="", output="", tool_calls=[ToolCall("lookup", actual)])
    with patch.object(
        MCPTraceCompletenessMetric, "measure", side_effect=AssertionError("Gate called scorer")
    ) as scorer:
        configured = roundtrip("mcp_trace_completeness", {"expected_trace": [{"name": "lookup", "input": expected}]})
        described = describe_configured_metric(metric)
        decision = validate_metric(configured, case)
        scorer.assert_not_called()
    assert decision.ready is (expected is not None and actual is not None)
    if expected is None:
        assert not described.reviewed
        assert configured.reasons[0].code == described.reasons[0].code == ReasonCode.INVALID_CONFIGURATION
    elif actual is None:
        assert {reason.code for reason in decision.reasons} == {ReasonCode.INVALID_EVIDENCE}
    if expected is not None:
        assert configured.to_dict() == described.to_dict()
    # Legacy direct scoring remains unchanged; admission is an opt-in boundary.
    assert metric.measure(case).value == float(expected == actual)


@pytest.mark.parametrize("expected", [[], [ToolCall("lookup", {})]])
@pytest.mark.parametrize("actual", [None, [], [ToolCall("lookup")]])
def test_mcp_completeness_known_empty_lists_do_not_require_vacuous_argument_checks(
    expected: list[ToolCall], actual: list[ToolCall] | None
) -> None:
    metric = MCPTraceCompletenessMetric(expected)
    decision = validate_metric(describe_configured_metric(metric), EvalCase(input="", output="", tool_calls=actual))
    assert decision.ready is (actual is not None and (not expected or not actual))
    if decision.ready:
        assert metric.measure(EvalCase(input="", output="", tool_calls=actual)).value == (0.0 if expected else 1.0)


@pytest.mark.parametrize("selective", [False, True])
@pytest.mark.parametrize("observed", [False, True])
def test_mcp_group_admission_distinguishes_unknown_arguments_from_observed_empty(
    selective: bool, observed: bool
) -> None:
    options = {"expected_trace": [{"name": "lookup", "input": {}}]}
    metric = MCPTraceCompletenessMetric([ToolCall("lookup", {})])
    attrs = {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "lookup", "gen_ai.tool.call.id": "call"}
    if observed:
        attrs["gen_ai.tool.call.arguments"] = {}
    assembled = traces_to_group_eval_case(
        [GroupTrace([{"span_id": "call", "attributes": attrs}], "one", True), GroupTrace([], "two", True)],
        metrics=[metric] if selective else None,
    )
    context = ValidationContext(mode="online", scope="group", group_diagnostics=assembled.diagnostics)
    with patch.object(
        MCPTraceCompletenessMetric, "measure", side_effect=AssertionError("Gate called scorer")
    ) as scorer:
        decision = validate_metric(roundtrip("mcp_trace_completeness", options), assembled.case, context=context)
        scorer.assert_not_called()
    assert decision.ready is observed
    if observed:
        assert metric.measure(assembled.case).value == 1.0
    else:
        assert {reason.code for reason in decision.reasons} == {ReasonCode.INVALID_EVIDENCE}


@pytest.mark.parametrize("kind", ["turn_latency", "turn_token_cost"])
@pytest.mark.parametrize("bad", ["invalid", "0", True, -1, float("nan"), float("inf")])
@pytest.mark.parametrize("reverse", [False, True])
def test_malformed_turn_value_is_not_an_admissible_partial(kind: str, bad: Any, reverse: bool) -> None:
    name = "latency_ms" if kind == "turn_latency" else "token_count"
    messages = [Message("assistant", **{name: 0}), Message("assistant", **{name: bad})]
    if reverse:
        messages.reverse()
    case = EvalCase(input="", output="", messages=messages)
    cls = TurnLatencyMetric if kind == "turn_latency" else TurnTokenCostMetric
    with patch.object(cls, "measure", side_effect=AssertionError("Gate called scorer")) as scorer:
        decision = validate_metric(roundtrip(kind), case)
        scorer.assert_not_called()
    assert not decision.ready
    assert {reason.code for reason in decision.reasons} == {ReasonCode.INVALID_EVIDENCE}


@pytest.mark.parametrize("bad", ["invalid", True, -1, float("nan"), float("inf")])
def test_latency_metadata_fallback_validates_the_value_consumed_by_scoring(bad: Any) -> None:
    case = EvalCase(
        input="",
        output="",
        messages=[Message("assistant", latency_ms=0), Message("assistant", metadata={"latency_ms": bad})],
    )
    decision = validate_metric(roundtrip("turn_latency"), case)
    assert not decision.ready
    assert {reason.code for reason in decision.reasons} == {ReasonCode.INVALID_EVIDENCE}
    case.messages[1].latency_ms = 0
    assert validate_metric(roundtrip("turn_latency"), case).ready
    assert TurnLatencyMetric().measure(case).value == 1.0


@pytest.mark.parametrize("kind", ["turn_latency", "turn_token_cost"])
@pytest.mark.parametrize("hydrated", [False, True])
def test_supported_missing_turn_values_are_skipped_not_invalid(kind: str, hydrated: bool) -> None:
    name = "latency_ms" if kind == "turn_latency" else "token_count"
    case = EvalCase(input="", output="", messages=[Message("assistant"), Message("assistant", **{name: 0})])
    if hydrated:
        case = EvalCase.from_dict(json.loads(json.dumps(case.to_dict())))
    original = deepcopy(case)
    decision = validate_metric(roundtrip(kind), case)
    assert decision.ready
    assert {reason.code for reason in decision.reasons} == {ReasonCode.PARTIAL_EVIDENCE}
    cls = TurnLatencyMetric if kind == "turn_latency" else TurnTokenCostMetric
    assert cls().measure(case).value == 1.0
    assert case == original


def test_fractional_turn_tokens_do_not_hide_behind_valid_turns() -> None:
    case = EvalCase(
        input="", output="", messages=[Message("assistant", token_count=0), Message("assistant", token_count=0.5)]
    )
    assert not validate_metric(roundtrip("turn_token_cost"), case).ready


def sub(name: str, weight: Any, *, missing: bool = False) -> dict[str, Any]:
    return {
        "name": name,
        "weight": weight,
        "skip_when_missing": missing,
        "check": {"type": "field_exists", "field": "output.absent" if missing else "output.ok"},
    }


@pytest.mark.parametrize("weights", [[0], [0, 0], [1e308, 1e308], [-1, 2], [True], [10**400]])
def test_composite_rejects_unusable_configuration_without_scoring(weights: list[Any]) -> None:
    subs = [sub(str(index), weight) for index, weight in enumerate(weights)]
    with patch.object(CompositeMetric, "measure", side_effect=AssertionError("Gate called scorer")) as scorer:
        configured = roundtrip("composite", {"sub_scores": subs})
        decision = validate_metric(configured, EvalCase(input="", output={"ok": True}))
        scorer.assert_not_called()
    assert not configured.reviewed and not decision.ready
    assert {reason.code for reason in configured.reasons} == {ReasonCode.INVALID_CONFIGURATION}
    assert ReasonCode.INVALID_CONFIGURATION in {reason.code for reason in decision.reasons}


def test_composite_requires_positive_remaining_weight_after_skips() -> None:
    subs = [sub("positive", 1, missing=True), sub("zero", 0)]
    configured = roundtrip("composite", {"sub_scores": subs})
    case = EvalCase(input="", output={"ok": True})
    decision = validate_metric(configured, case)
    assert configured.reviewed and not decision.ready
    assert {reason.code for reason in decision.reasons} == {
        ReasonCode.PARTIAL_EVIDENCE,
        ReasonCode.INSUFFICIENT_EVIDENCE,
    }
    case.output["absent"] = True
    assert validate_metric(configured, case).ready
    assert CompositeMetric(sub_scores=subs).measure(case).value == 1.0


@pytest.mark.parametrize("actual,expected", [(True, 1.0), (False, 0.0)])
def test_weighted_composite_quality_zero_is_still_usable_evidence(actual: bool, expected: float) -> None:
    subs = [sub("positive", 2), sub("skipped", 1, missing=True), sub("zero", 0)]
    for selected in subs:
        if selected["name"] != "skipped":
            selected["check"] = {"type": "equals", "field": "output.ok", "value": True}
    case = EvalCase(input="", output={"ok": actual})
    decision = validate_metric(roundtrip("composite", {"sub_scores": subs}), case)
    assert decision.ready
    assert {reason.code for reason in decision.reasons} == {ReasonCode.PARTIAL_EVIDENCE}
    assert CompositeMetric(sub_scores=subs).measure(case).value == expected


@pytest.mark.parametrize("weights", [[-1], [1, -2], [-1, 2], [1e308, 1e308]])
def test_factory_weighted_composite_rejects_negative_or_overflowing_weight(weights: list[float]) -> None:
    metric = build_metric(
        "composite",
        {
            "aggregation": "weighted_average",
            "metrics": [
                {"ref": str(index), "type": "heuristic", "config": {"kind": "exact_match"}, "weight": weight}
                for index, weight in enumerate(weights)
            ],
        },
        allow_code_loading=False,
    )
    configured = describe_configured_metric(metric)
    assert not configured.reviewed
    assert configured.reasons[0].code == ReasonCode.INVALID_CONFIGURATION


@pytest.mark.parametrize("arg_match", ["exact", "subset"])
def test_greedy_argument_subset_pairing_requires_established_order(arg_match: str) -> None:
    options = {"pair": "subset", "arg_match": arg_match}
    metric = ToolArgumentMatchMetric(**options)
    raw_scores = []
    for reverse in (False, True):
        rows = [
            {
                "span_id": sid,
                "start_timestamp": 1,
                "end_timestamp": 2,
                "attributes": {
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "lookup",
                    "gen_ai.tool.call.id": sid,
                    "gen_ai.tool.call.arguments": {"value": value},
                },
            }
            for sid, value in zip(("z", "a") if reverse else ("a", "z"), ("y", "z"), strict=True)
        ]
        assembled = traces_to_group_eval_case(
            [GroupTrace(rows, "one", True), GroupTrace([], "two", True)],
            authored=GroupAuthoredData(
                expected_tool_calls=[ToolCall("lookup", {"value": value}) for value in ("x", "y")]
            ),
            metrics=[metric],
        )
        decision = validate_metric(
            roundtrip("tool_argument_match", options),
            assembled.case,
            context=ValidationContext(mode="online", scope="group", group_diagnostics=assembled.diagnostics),
        )
        assert not decision.ready
        assert {reason.code for reason in decision.reasons} == {ReasonCode.AMBIGUOUS_ORDER}
        raw_scores.append(metric.measure(assembled.case).value)
    assert sorted(raw_scores) == [0.0, 0.5]


@pytest.mark.parametrize("kind", ["turn_latency", "turn_token_cost"])
def test_normalized_group_telemetry_remains_a_supported_partial(kind: str) -> None:
    name = "latency_ms" if kind == "turn_latency" else "token_count"
    row = {
        "span_id": "capture",
        "start_timestamp": 1,
        "end_timestamp": 2,
        "attributes": {
            "gen_ai.output.messages": [
                {"id": "good", "role": "assistant", name: 0},
                {"id": "bad", "role": "assistant", name: "invalid"},
            ]
        },
    }
    assembled = traces_to_group_eval_case([GroupTrace([row], "one", True), GroupTrace([], "two", True)])
    assert [getattr(message, name) for message in assembled.case.messages] == [0, None]
    decision = validate_metric(
        roundtrip(kind),
        assembled.case,
        context=ValidationContext(mode="online", scope="group", group_diagnostics=assembled.diagnostics),
    )
    assert decision.ready and ReasonCode.PARTIAL_EVIDENCE in {reason.code for reason in decision.reasons}
    cls = TurnLatencyMetric if kind == "turn_latency" else TurnTokenCostMetric
    assert cls().measure(assembled.case).value == 1.0
