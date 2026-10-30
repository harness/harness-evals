"""Focused Group contracts: identity, causality, usage and projection parity."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
from datetime import datetime, timedelta, timezone
from itertools import permutations
from typing import Any
from unittest.mock import patch

import pytest

import harness_evals.adapters.group as group_adapter
from harness_evals.adapters import (
    GroupAuthoredData,
    GroupErrorCode,
    GroupInputError,
    GroupIssue,
    GroupTrace,
    traces_to_group_eval_case,
)
from harness_evals.catalog import catalog
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension, EvaluationMode, MetricInputs, OnlineScope
from harness_evals.core.score import Score
from harness_evals.core.types import Message, ToolCall
from harness_evals.metrics.agent.tool_correctness import ToolCorrectnessMetric
from harness_evals.metrics.conversation.tool_use import ToolUseMetric
from harness_evals.metrics.operational.latency import LatencyMetric
from harness_evals.metrics.operational.token_cost import TokenCostMetric
from harness_evals.metrics.operational.turn_latency import TurnLatencyMetric
from harness_evals.metrics.operational.turn_token_cost import TurnTokenCostMetric
from harness_evals.validation import ReasonCode, ValidationContext, resolve_metric_contract, validate_metric
from tests.conftest import MockLLM

pytestmark = pytest.mark.unit


def test_group_assembly_does_not_advertise_metric_group_eligibility() -> None:
    assert all(OnlineScope.GROUP not in entry.applicability.online_scopes for entry in catalog())


def span(
    sid: str | None = "s",
    *,
    trace: str | None = None,
    parent: str | None = None,
    start: Any = 1,
    end: Any = 2,
    inputs: list[dict[str, Any]] | None = None,
    outputs: list[dict[str, Any]] | None = None,
    attrs: dict[str, Any] | None = None,
    **columns: Any,
) -> dict[str, Any]:
    attributes = dict(attrs or {})
    if inputs is not None:
        attributes["gen_ai.input.messages"] = inputs
    if outputs is not None:
        attributes["gen_ai.output.messages"] = outputs
    return {
        "span_id": sid,
        "trace_id": trace,
        "parent_span_id": parent,
        "start_timestamp": start,
        "end_timestamp": end,
        "attributes": attributes,
        **columns,
    }


def message(role: str, content: str | None = "", **fields: Any) -> dict[str, Any]:
    return {"role": role, "content": content, **fields}


def members(*rows: dict[str, Any], complete: bool | None = None) -> list[GroupTrace]:
    return [GroupTrace(list(rows), "one", complete), GroupTrace([], "two", complete)]


def issues(assembly: group_adapter.GroupAssembly) -> set[GroupIssue]:
    return {item.issue for item in assembly.diagnostics.issues}


def tool(
    sid: str,
    call_id: str | None,
    start: float,
    end: float,
    output: str | dict[str, Any] | None,
    *,
    parent: str | None = None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "lookup"}
    if call_id is not None:
        attrs["gen_ai.tool.call.id"] = call_id
    if output is not None:
        attrs["gen_ai.tool.call.result"] = output
    attrs["gen_ai.tool.call.arguments"] = {"key": "same"}
    return span(sid, parent=parent, start=start, end=end, attrs=attrs)


def intent(call_id: str | None, **fields: Any) -> dict[str, Any]:
    call: dict[str, Any] = {"function": {"name": "lookup", "arguments": '{"key":"same"}'}}
    if call_id is not None:
        call["id"] = call_id
    return message("assistant", None, tool_calls=[call], **fields)


class DeclaredMetric(BaseMetric):
    def __init__(self, required: tuple[str, ...] = (), optional: tuple[str, ...] = ()) -> None:
        super().__init__("declared", Dimension.PERFORMANCE)
        self.eval_inputs = MetricInputs(required, optional)

    def measure(self, eval_case: EvalCase) -> Score:
        raise AssertionError("Adaptation must not run a metric")


@pytest.mark.parametrize("count", [2, 64])
def test_public_bounds_and_explicit_empty_capture(count: int) -> None:
    result = traces_to_group_eval_case([GroupTrace([], f"t{i}", True) for i in range(count)])
    assert result.case.messages == result.case.tool_calls == []
    assert result.diagnostics.member_count == result.diagnostics.complete_member_count == count
    assert result.case.input == result.case.output == ""
    for field in ("runs", "expected", "context", "latency_ms", "retry_count", "confidence"):
        assert getattr(result.case, field) is None


@pytest.mark.parametrize("count", [0, 1, 65])
def test_reject_bounds_never_truncate(count: int) -> None:
    with pytest.raises(GroupInputError) as error:
        traces_to_group_eval_case([GroupTrace([]) for _ in range(count)])
    assert error.value.code is GroupErrorCode.MEMBER_COUNT


@pytest.mark.parametrize("complete", [None, False])
def test_unknown_or_incomplete_absence_is_not_empty(complete: bool | None) -> None:
    result = traces_to_group_eval_case(members(complete=complete))
    assert result.case.messages is result.case.tool_calls is None
    assert result.case.token_count is result.case.cost_usd is None


def test_roles_repeated_equal_turns_empty_and_tool_only_are_preserved() -> None:
    raw = [
        message(role, "same" if role == "user" else "") for role in ("system", "developer", "user", "user", "custom")
    ]
    raw += [intent("never-executed"), message("tool", "", tool_call_id="unmatched")]
    result = traces_to_group_eval_case(members(span(inputs=raw)))
    assert [msg.role for msg in result.case.messages] == [raw_msg["role"] for raw_msg in raw]
    assert [msg.content for msg in result.case.messages] == [raw_msg["content"] for raw_msg in raw]
    assert isinstance(result.case.messages[-2].tool_calls[0], ToolCall)
    assert result.case.tool_calls is None  # An intent is not an observed execution.
    assert GroupIssue.AMBIGUOUS_MESSAGE_IDENTITY in issues(result)
    assert GroupIssue.UNMATCHED_TOOL_RESULT in issues(result)


def test_span_and_message_ids_are_scoped_to_trace() -> None:
    first = span(trace="first", inputs=[message("user", "repeat", id="message")], input_tokens=2)
    second = deepcopy(first)
    second["trace_id"] = "second"
    result = traces_to_group_eval_case([GroupTrace([first]), GroupTrace([second])])
    assert len(result.case.messages) == 2
    assert result.case.input_tokens == 4
    assert result.diagnostics.duplicate_span_count == 0


def test_missing_trace_ids_use_member_local_scope() -> None:
    raw = span(inputs=[message("user", "repeat", id="m")])
    result = traces_to_group_eval_case([GroupTrace([raw]), GroupTrace([deepcopy(raw)])])
    assert len(result.case.messages) == 2
    assert GroupIssue.MISSING_TRACE_ID in issues(result)


@pytest.mark.parametrize(
    ("selected", "code"),
    [
        ([GroupTrace([span(trace="different")], "explicit"), GroupTrace([], "other")], GroupErrorCode.TRACE_SCOPE),
        ([GroupTrace([span(trace="a"), span("s2", trace="b")]), GroupTrace([])], GroupErrorCode.TRACE_SCOPE),
        ([GroupTrace([], "same"), GroupTrace([], "same")], GroupErrorCode.REPEATED_TRACE),
    ],
)
def test_scope_failures(selected: list[GroupTrace], code: GroupErrorCode) -> None:
    with pytest.raises(GroupInputError) as error:
        traces_to_group_eval_case(selected)
    assert error.value.code is code


def test_duplicate_exact_span_capture_is_counted_once() -> None:
    raw = span(inputs=[message("user", "x")], input_tokens=4, output_tokens=0)
    result = traces_to_group_eval_case(members(raw, deepcopy(raw)))
    assert result.diagnostics.span_count == result.diagnostics.duplicate_span_count == 1
    assert len(result.case.messages) == 1
    assert result.case.token_count == 4
    assert GroupIssue.DUPLICATE_CAPTURE in issues(result)


def test_identical_message_id_capture_is_counted_once_not_equal_text() -> None:
    raw = message("user", "x", message_id="m")
    result = traces_to_group_eval_case(members(span("a", inputs=[raw]), span("b", inputs=[deepcopy(raw)])))
    assert len(result.case.messages) == 1
    no_ids = traces_to_group_eval_case(
        members(span("a", inputs=[message("user", "x")]), span("b", inputs=[message("user", "x")]))
    )
    assert len(no_ids.case.messages) == 2


@pytest.mark.parametrize("identity", ["span", "message"])
def test_conflicting_identical_identity_fails_closed(identity: str) -> None:
    first = span("a", inputs=[message("user", "first", id="m")])
    second = span("a" if identity == "span" else "b", inputs=[message("user", "other", id="m")])
    with pytest.raises(GroupInputError) as error:
        traces_to_group_eval_case(members(first, second))
    assert error.value.code is GroupErrorCode.CONFLICTING_IDENTITY


@pytest.mark.parametrize("mode", ["exact", "subsequence", "subset"])
@pytest.mark.parametrize("layout", ["span_ids", "trace_ids", "overlap", "zero_duration"])
@pytest.mark.parametrize("projection", ["full", "selective", "mixed"])
def test_unestablished_execution_order_is_not_safe_for_ordered_scoring(mode: str, layout: str, projection: str) -> None:
    metric = ToolCorrectnessMetric(mode=mode)
    requested = {"full": None, "selective": [metric], "mixed": [metric, TokenCostMetric()]}[projection]
    contract = resolve_metric_contract("tool_correctness", {"mode": mode})
    raw_scores = []
    for reverse_ids in (False, True):
        identifiers = ["a", "z"] if not reverse_ids else ["z", "a"]
        first = tool(identifiers[0], "first-call", 1, 4 if layout == "overlap" else 2, "first result")
        second = tool(identifiers[1], "second-call", 2 if layout == "overlap" else 1, 3, "second result")
        if layout != "overlap":
            second["end_timestamp"] = first["end_timestamp"]
        if layout == "zero_duration":
            first["end_timestamp"] = second["end_timestamp"] = 1
        first["attributes"]["gen_ai.tool.name"] = "first"
        second["attributes"]["gen_ai.tool.name"] = "second"
        if layout == "trace_ids":
            first["span_id"] = second["span_id"] = "execution"
            selected = [GroupTrace([first], identifiers[0], True), GroupTrace([second], identifiers[1], True)]
        else:
            selected = members(first, second, complete=True)
        assembly = traces_to_group_eval_case(
            selected, authored=GroupAuthoredData(expected_tools=["first", "second"]), metrics=requested
        )
        assert issues(assembly) == {GroupIssue.AMBIGUOUS_TOOL_ORDER}
        assert assembly.diagnostics.complete_member_count == 2
        # Diagnose canonical evidence even when no tool/message fields are projected.
        assert traces_to_group_eval_case(selected, metrics=[]).diagnostics == assembly.diagnostics
        decision = validate_metric(
            contract,
            assembly.case,
            context=ValidationContext(
                mode=EvaluationMode.ONLINE, scope=OnlineScope.GROUP, group_diagnostics=assembly.diagnostics
            ),
        )
        admitted_score = metric.measure(assembly.case) if decision.ready else None
        if mode == "subset":
            assert admitted_score is not None and admitted_score.value == 1.0
        else:
            assert admitted_score is None
            assert {reason.code for reason in decision.reasons} == {ReasonCode.AMBIGUOUS_ORDER}
        # Demonstrate why directly scoring the deterministic presentation is unsafe.
        raw_scores.append(metric.measure(assembly.case).value)
    if layout in ("span_ids", "trace_ids", "zero_duration") and mode != "subset":
        assert raw_scores[0] == 1.0 and raw_scores[1] < 1.0


@pytest.mark.parametrize("mode", ["exact", "subsequence", "subset"])
@pytest.mark.parametrize("ordering", ["separated", "touching", "parent", "result_history"])
@pytest.mark.parametrize("selective", [False, True])
def test_established_execution_order_allows_real_gate_and_scorer(mode: str, ordering: str, selective: bool) -> None:
    first = tool("z-first", "first-call", 1, 2, "first result")
    second = tool("a-second", "second-call", 1, 2, "second result")
    if ordering == "separated":
        second.update(start_timestamp=3, end_timestamp=4)
    elif ordering == "touching":
        second.update(start_timestamp=2, end_timestamp=3)
    elif ordering == "parent":
        second["parent_span_id"] = first["span_id"]
    else:
        result = message("tool", "first result", id="result-event", tool_call_id="first-call")
        first["attributes"]["gen_ai.output.messages"] = [result]
        second["attributes"]["gen_ai.input.messages"] = [deepcopy(result)]
    first["attributes"]["gen_ai.tool.name"] = "first"
    second["attributes"]["gen_ai.tool.name"] = "second"
    metric = ToolCorrectnessMetric(mode=mode)
    assembly = traces_to_group_eval_case(
        members(second, first, complete=True),
        authored=GroupAuthoredData(expected_tools=["first", "second"]),
        metrics=[metric] if selective else None,
    )
    assert issues(assembly) == ({GroupIssue.DUPLICATE_CAPTURE} if ordering == "result_history" else set())
    decision = validate_metric(
        resolve_metric_contract("tool_correctness", {"mode": mode}),
        assembly.case,
        context=ValidationContext(
            mode=EvaluationMode.ONLINE, scope=OnlineScope.GROUP, group_diagnostics=assembly.diagnostics
        ),
    )
    assert decision.ready
    assert metric.measure(assembly.case).value == 1.0


@pytest.mark.parametrize("call_id", [None, "call"])
@pytest.mark.parametrize("output", ["observed", "", {}])
@pytest.mark.parametrize("selective", [False, True])
def test_identified_execution_results_have_derived_identity_and_are_admitted(
    call_id: str | None, output: Any, selective: bool
) -> None:
    selected = [
        GroupTrace([tool("same-span", call_id, 1, 2, output)], "one", True),
        GroupTrace([tool("same-span", call_id, 3, 4, output)], "two", True),
    ]
    metric = ToolCorrectnessMetric()
    assembly = traces_to_group_eval_case(
        selected,
        authored=GroupAuthoredData(expected_tools=["lookup", "lookup"]),
        metrics=[metric, DeclaredMetric(("messages",))] if selective else None,
    )
    assert issues(assembly) == set()
    identities = [msg.metadata["message_id"] for msg in assembly.case.messages]
    assert len(set(identities)) == 2  # Equal span/call IDs are trace-scoped.
    assert all(msg.metadata["source_span_id"] == "same-span" for msg in assembly.case.messages)
    assert identities == [
        msg.metadata["message_id"] for msg in traces_to_group_eval_case(list(reversed(selected))).case.messages
    ]
    decision = validate_metric(
        resolve_metric_contract("tool_correctness"),
        assembly.case,
        context=ValidationContext(
            mode=EvaluationMode.ONLINE, scope=OnlineScope.GROUP, group_diagnostics=assembly.diagnostics
        ),
    )
    assert decision.ready
    assert metric.measure(assembly.case).value == 1.0


@pytest.mark.parametrize("unidentified", ["source_result", "source_history", "execution_span"])
def test_derived_results_do_not_hide_genuinely_missing_source_identity(unidentified: str) -> None:
    row = tool("execution", "call", 1, 2, "observed")
    if unidentified == "execution_span":
        row["span_id"] = None
    elif unidentified == "source_result":
        row["attributes"]["gen_ai.output.messages"] = [message("tool", "observed", tool_call_id="call")]
    else:
        row["attributes"]["gen_ai.input.messages"] = [message("user", "unidentified history")]
    metric = ToolCorrectnessMetric(mode="subset")
    for requested in (None, [metric]):
        assembly = traces_to_group_eval_case(
            members(row, complete=True), authored=GroupAuthoredData(expected_tools=["lookup"]), metrics=requested
        )
        assert GroupIssue.AMBIGUOUS_MESSAGE_IDENTITY in issues(assembly)
        decision = validate_metric(
            resolve_metric_contract("tool_correctness", {"mode": "subset"}),
            assembly.case,
            context=ValidationContext(
                mode=EvaluationMode.ONLINE, scope=OnlineScope.GROUP, group_diagnostics=assembly.diagnostics
            ),
        )
        assert not decision.ready
        assert ReasonCode.AMBIGUOUS_TOOLS in {reason.code for reason in decision.reasons}


def test_derived_identity_does_not_resolve_ambiguous_result_ownership() -> None:
    rows = [tool("first", "shared", 1, 2, "first"), tool("second", "shared", 3, 4, "second")]
    metric = ToolCorrectnessMetric(mode="subset")
    for requested in (None, [metric, DeclaredMetric(("messages",))]):
        assembly = traces_to_group_eval_case(
            members(*rows, complete=True),
            authored=GroupAuthoredData(expected_tools=["lookup", "lookup"]),
            metrics=requested,
        )
        assert issues(assembly) == {GroupIssue.AMBIGUOUS_TOOL_LINK}
        assert len({msg.metadata["message_id"] for msg in assembly.case.messages}) == 2
        decision = validate_metric(
            resolve_metric_contract("tool_correctness", {"mode": "subset"}),
            assembly.case,
            context=ValidationContext(
                mode=EvaluationMode.ONLINE, scope=OnlineScope.GROUP, group_diagnostics=assembly.diagnostics
            ),
        )
        assert not decision.ready
        assert {reason.code for reason in decision.reasons} == {ReasonCode.AMBIGUOUS_TOOLS}


def test_derived_result_identity_cannot_collide_with_source_message_identity() -> None:
    row = tool("execution", "call", 1, 2, "observed")
    baseline = traces_to_group_eval_case(members(row))
    derived_id = baseline.case.messages[0].metadata["message_id"]
    source = span("source", start=3, end=4, inputs=[message("user", "distinct event", id=derived_id)])
    result = traces_to_group_eval_case(members(row, source))
    assert [(msg.role, msg.content) for msg in result.case.messages] == [
        ("tool", "observed"),
        ("user", "distinct event"),
    ]
    assert GroupIssue.AMBIGUOUS_MESSAGE_IDENTITY not in issues(result)
    assert result.diagnostics.message_count == 2


def test_tied_timestamps_use_ids_not_capture_order() -> None:
    rows = [span(sid, outputs=[message("assistant", sid)]) for sid in ("c", "a", "b")]
    result = traces_to_group_eval_case(members(*rows))
    shuffled = traces_to_group_eval_case(members(*reversed(rows)))
    assert result.case == shuffled.case
    assert [msg.content for msg in result.case.messages] == ["a", "b", "c"]


def test_source_array_order_overrides_timestamp_tie_breaks() -> None:
    raw = span(
        inputs=[message("user", "question"), message("assistant", "earlier")], outputs=[message("assistant", "answer")]
    )
    result = traces_to_group_eval_case(members(raw))
    assert [msg.content for msg in result.case.messages] == ["question", "earlier", "answer"]


def test_utc_datetime_and_numeric_timestamp_ordering() -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        span(
            "later",
            start=(base + timedelta(seconds=2)).timestamp(),
            end=(base + timedelta(seconds=3)).timestamp(),
            outputs=[message("assistant", "later")],
        ),
        span(
            "early",
            start="2026-01-01T01:00:00+01:00",
            end=base + timedelta(seconds=1),
            outputs=[message("assistant", "early")],
        ),
    ]
    result = traces_to_group_eval_case(members(*rows))
    assert [msg.content for msg in result.case.messages] == ["early", "later"]


@pytest.mark.parametrize("unknown", [None, "not-a-time", "1", float("nan"), float("inf"), True, 10**400])
def test_invalid_timestamps_remain_unknown_not_epoch(unknown: Any) -> None:
    rows = [
        span("unknown", start=unknown, end=unknown, outputs=[message("assistant", "unknown")]),
        span("known", start=1, end=2, outputs=[message("assistant", "known")]),
    ]
    result = traces_to_group_eval_case(members(*rows))
    assert [msg.content for msg in result.case.messages] == ["known", "unknown"]
    assert GroupIssue.UNKNOWN_TIMESTAMP in issues(result)


def test_concurrent_identical_name_and_args_link_by_id_reverse_completion() -> None:
    rows = [
        span("llm", start=0, end=0.5, outputs=[intent("a"), intent("b")]),
        tool("a-span", "a", 1, 10, "A"),
        tool("b-span", "b", 2, 3, "B"),
    ]
    result = traces_to_group_eval_case(members(*reversed(rows)))
    assert [call.output for call in result.case.tool_calls] == ["A", "B"]
    assert [msg.content for msg in result.case.messages if msg.role == "tool"] == ["B", "A"]
    assistants = [msg for msg in result.case.messages if msg.role == "assistant"]
    assert [msg.tool_calls[0].output for msg in assistants] == ["A", "B"]
    assert GroupIssue.CYCLIC_ORDER not in issues(result)
    assert traces_to_group_eval_case(members(*rows)).case == result.case


def test_tool_call_ids_never_link_across_members() -> None:
    result = traces_to_group_eval_case(
        [GroupTrace([span(outputs=[intent("same")])], "one"), GroupTrace([tool("tool", "same", 1, 2, "result")], "two")]
    )
    assistant = next(msg for msg in result.case.messages if msg.role == "assistant")
    assert assistant.tool_calls[0].output is None
    assert result.case.tool_calls[0].output == "result"


def test_nested_tool_completion_follows_child_not_parent_start() -> None:
    rows = [tool("parent", "p", 1, 10, "parent result"), tool("child", "c", 2, 3, "child result", parent="parent")]
    result = traces_to_group_eval_case(members(*rows))
    assert [call.output for call in result.case.tool_calls] == ["parent result", "child result"]
    assert [msg.content for msg in result.case.messages] == ["child result", "parent result"]
    assert all(msg.role == "tool" for msg in result.case.messages)
    assert GroupIssue.CYCLIC_ORDER not in issues(result)


def test_parent_agent_completion_not_flattened_before_child() -> None:
    rows = [
        span("parent", start=1, end=2, outputs=[message("assistant", "done")]),
        tool("child", "c", 3, 4, "result", parent="parent"),
    ]
    result = traces_to_group_eval_case(members(*rows))
    assert [msg.content for msg in result.case.messages] == ["result", "done"]


def test_unique_structural_parent_intent_link_without_id() -> None:
    rows = [
        span("parent", start=0, end=10, outputs=[intent(None)]),
        tool("child", None, 1, 2, "value", parent="parent"),
    ]
    result = traces_to_group_eval_case(members(*rows))
    assert result.case.messages[0].tool_calls[0].output == "value"
    assert GroupIssue.CYCLIC_ORDER not in issues(result)


def test_name_and_arguments_never_link_ambiguous_structural_siblings() -> None:
    rows = [
        span("parent", start=0, end=10, outputs=[intent(None)]),
        tool("a", None, 1, 2, "a", parent="parent"),
        tool("b", None, 1, 2, "b", parent="parent"),
    ]
    result = traces_to_group_eval_case(members(*rows))
    assistant = next(msg for msg in result.case.messages if msg.role == "assistant")
    assert assistant.tool_calls[0].output is None
    assert len(result.case.tool_calls) == 2
    assert GroupIssue.AMBIGUOUS_TOOL_LINK in issues(result)


@pytest.mark.parametrize("output", ["", {}])
def test_orphan_execution_retains_empty_output_without_fabricated_assistant(output: Any) -> None:
    result = traces_to_group_eval_case(members(tool("tool", "a", 1, 2, output)))
    assert result.case.tool_calls == [ToolCall("lookup", {"key": "same"}, output)]
    assert [msg.role for msg in result.case.messages] == ["tool"]
    assert result.case.messages[0].content == (output if isinstance(output, str) else "{}")


def test_external_explicit_tool_result_links_and_keeps_actual_role() -> None:
    rows = [
        span("intent", start=0, end=1, outputs=[intent("a")]),
        tool("execution", "a", 2, 3, None),
        span("result", start=4, end=5, outputs=[message("tool", "", tool_call_id="a")]),
    ]
    result = traces_to_group_eval_case(members(*rows))
    assert result.case.tool_calls[0].output == ""
    assert result.case.messages[0].tool_calls[0].output == ""
    assert result.case.messages[-1].role == "tool"
    assert result.case.messages[-1].metadata["tool_call_id"] == "a"
    assert GroupIssue.UNMATCHED_TOOL_RESULT not in issues(result)


def test_ambiguous_and_unmatched_results_are_retained_without_latest_assistant_matching() -> None:
    rows = [
        span("intent", outputs=[intent("a")]),
        tool("exec-one", "a", 2, 3, None),
        tool("exec-two", "a", 2, 3, None),
        span(
            "result",
            start=4,
            end=5,
            outputs=[message("tool", "result", tool_call_id="a"), message("tool", "orphan", tool_call_id="missing")],
        ),
    ]
    result = traces_to_group_eval_case(members(*rows))
    assert [msg.content for msg in result.case.messages if msg.role == "tool"] == ["result", "orphan"]
    assert all(call.output is None for call in result.case.tool_calls)
    assert {GroupIssue.AMBIGUOUS_TOOL_LINK, GroupIssue.UNMATCHED_TOOL_RESULT} <= issues(result)


def test_missing_parents_and_cycles_keep_evidence_with_safe_diagnostics() -> None:
    rows = [
        span("a", parent="b", outputs=[message("assistant", "a")], input_tokens=5),
        span("b", parent="a", outputs=[message("assistant", "b")], input_tokens=6),
        span("orphan", parent="missing", outputs=[message("custom", "orphan")], input_tokens=9),
        span("valid", input_tokens=2),
    ]
    result = traces_to_group_eval_case(members(*rows))
    assert sorted(msg.content for msg in result.case.messages) == ["a", "b", "orphan"]
    assert {GroupIssue.CYCLIC_ORDER, GroupIssue.MISSING_PARENT, GroupIssue.AMBIGUOUS_USAGE} <= issues(result)
    assert result.case.input_tokens == 2
    assert result.diagnostics.input_tokens.ambiguous == 3
    assert traces_to_group_eval_case(members(*rows)).case == result.case


@pytest.mark.parametrize(
    ("incoming", "outgoing", "expected"),
    [(None, None, None), (0, None, 0), (None, 0, 0), (2, None, 2), (None, 3, 3), (0, 0, 0)],
)
def test_usage_unknown_vs_observed_directional_zero(incoming: Any, outgoing: Any, expected: Any) -> None:
    result = traces_to_group_eval_case(members(span(input_tokens=incoming, output_tokens=outgoing)))
    assert result.case.token_count == expected
    assert result.case.input_tokens == incoming
    assert result.case.output_tokens == outgoing
    assert result.case.cost_usd is None


def test_usage_descendant_frontier_is_independent_per_dimension() -> None:
    rows = [
        span("root", input_tokens=100, output_tokens=12, cost_usd=5),
        span("child", parent="root", input_tokens=30),
        span("grandchild", parent="child", output_tokens=2, cost_usd=1),
        span("sibling", parent="root", input_tokens=10, output_tokens=3, cost_usd=0),
    ]
    result = traces_to_group_eval_case(members(*rows))
    assert (result.case.input_tokens, result.case.output_tokens, result.case.token_count, result.case.cost_usd) == (
        40,
        5,
        45,
        1,
    )
    assert asdict(result.diagnostics.input_tokens) == {
        "observed": 3,
        "selected": 2,
        "missing": 1,
        "invalid": 0,
        "overlapping": 1,
        "ambiguous": 0,
    }
    assert result.diagnostics.cost_usd.overlapping == 1
    assert result.diagnostics.cost_usd.selected == 2


def test_disjoint_members_and_siblings_sum_without_completeness_claim() -> None:
    result = traces_to_group_eval_case(
        [
            GroupTrace(
                [
                    span("root", input_tokens=100),
                    span("a", parent="root", input_tokens=10),
                    span("b", parent="root", input_tokens=20),
                ],
                "one",
            ),
            GroupTrace([span("root", input_tokens=5, cost_usd=0)], "two"),
        ]
    )
    assert result.case.input_tokens == 35
    assert result.case.cost_usd == 0
    assert result.diagnostics.unknown_member_count == 2
    assert result.diagnostics.input_tokens.selected == 3


@pytest.mark.parametrize(
    "invalid", [True, -1, 1.2, "1.2", "1.0", "1e0", "+1", " 1", "-0", "²", "bad", float("nan"), float("inf"), {}, []]
)
def test_invalid_raw_usage_does_not_fall_back_to_valid_attributes(invalid: Any) -> None:
    raw = span(input_tokens=invalid, attrs={"gen_ai.usage.input_tokens": 50, "gen_ai.usage.output_tokens": 0})
    result = traces_to_group_eval_case(members(raw))
    assert result.case.input_tokens is None
    assert result.case.output_tokens == result.case.token_count == 0
    assert result.diagnostics.input_tokens.invalid == 1
    assert GroupIssue.INVALID_TELEMETRY in issues(result)


@pytest.mark.parametrize("invalid", [True, -1, "bad", float("nan"), float("inf"), []])
def test_invalid_raw_cost_does_not_fall_back_or_estimate(invalid: Any) -> None:
    result = traces_to_group_eval_case(members(span(cost_usd=invalid, input_tokens=10, attrs={"gen_ai.usage.cost": 1})))
    assert result.case.cost_usd is None
    assert result.diagnostics.cost_usd.invalid == 1


def test_usage_identity_unknown_excludes_ambiguous_contributions() -> None:
    result = traces_to_group_eval_case(members(span(None, input_tokens=100), span("known", input_tokens=5)))
    assert result.case.input_tokens == 5
    assert result.diagnostics.input_tokens.ambiguous == 1
    assert GroupIssue.MISSING_SPAN_ID in issues(result)


def test_direct_message_telemetry_preserved_validated_and_not_double_counted() -> None:
    raw = span(
        outputs=[
            message("assistant", "one", latency_ms=7, token_count=4, cost_usd=0.2, input_tokens=2, output_tokens=2)
        ],
        input_tokens=10,
        output_tokens=5,
        cost_usd=1,
    )
    result = traces_to_group_eval_case(members(raw))
    observed = result.case.messages[0]
    assert (observed.latency_ms, observed.token_count, observed.cost_usd) == (7, 4, 0.2)
    assert (result.case.input_tokens, result.case.output_tokens, result.case.cost_usd) == (10, 5, 1)
    fallback = traces_to_group_eval_case(members(span(outputs=raw["attributes"]["gen_ai.output.messages"])))
    assert (fallback.case.input_tokens, fallback.case.output_tokens, fallback.case.cost_usd) == (2, 2, 0.2)


def test_single_message_span_telemetry_is_not_multiplied_across_messages() -> None:
    first = span(
        "single", outputs=[message("assistant", "one")], duration_ms=5, input_tokens=2, output_tokens=3, cost_usd=0
    )
    second = span(
        "multiple",
        outputs=[message("assistant", "two"), message("assistant", "three")],
        duration_ms=10,
        input_tokens=7,
        cost_usd=1,
    )
    result = traces_to_group_eval_case(members(first, second))
    by_content = {msg.content: msg for msg in result.case.messages}
    assert (by_content["one"].latency_ms, by_content["one"].token_count, by_content["one"].cost_usd) == (5, 5, 0)
    for content in ("two", "three"):
        assert by_content[content].latency_ms is by_content[content].token_count is by_content[content].cost_usd is None
    assert result.case.token_count == 12


def test_invalid_direct_message_telemetry_does_not_survive_typed_projection() -> None:
    raw = span(
        start=None,
        end=None,
        outputs=[message("assistant", "", latency_ms=float("inf"), token_count=1.5, cost_usd=True)],
    )
    result = traces_to_group_eval_case(members(raw))
    observed = result.case.messages[0]
    assert observed.latency_ms is observed.token_count is observed.cost_usd is None
    assert GroupIssue.INVALID_TELEMETRY in issues(result)


def test_authored_expectations_are_explicit_typed_and_deep_owned() -> None:
    expected = [ToolCall("expected", {"nested": []}, {})]
    metadata = {"labels": []}
    authored = GroupAuthoredData(
        expected_tools=["expected"], expected_tool_calls=expected, metadata=metadata, tags={"test": "authored"}
    )
    expected[0].input["nested"].append("original mutation")
    metadata["labels"].append("original mutation")
    result = traces_to_group_eval_case(members(tool("observed", "id", 1, 2, {})), authored=authored)
    assert result.case.expected_tools == ["expected"]
    assert result.case.expected_tool_calls == [ToolCall("expected", {"nested": []}, {})]
    assert result.case.metadata == {"labels": []}
    result.case.expected_tool_calls[0].input["nested"].append("output mutation")
    result.case.metadata["labels"].append("output mutation")
    assert authored.expected_tool_calls[0].input == {"nested": []}
    assert authored.metadata == {"labels": []}
    assert result.case.runs is None
    observed_only = traces_to_group_eval_case(members(tool("observed", "id", 1, 2, {})))
    assert (
        observed_only.case.expected_tools
        is observed_only.case.expected_tool_calls
        is observed_only.case.metadata
        is None
    )
    with pytest.raises(FrozenInstanceError):
        authored.tags = {}


def test_authored_dictionary_calls_are_hydrated() -> None:
    authored = GroupAuthoredData(expected_tool_calls=[{"name": "expected", "arguments": {"x": 1}, "result": ""}])
    result = traces_to_group_eval_case(members(), authored=authored)
    assert result.case.expected_tool_calls == [ToolCall("expected", {"x": 1}, "")]


def test_no_source_or_metric_mutation_and_typed_output_is_deep_owned() -> None:
    row = span(outputs=[intent("a", metadata={"nested": []})])
    execution = tool("tool", "a", 3, 4, {"nested": []})
    selected = members(row, execution)
    before = deepcopy(selected)
    metric = DeclaredMetric(("messages", "tool_calls"))
    metric_before = deepcopy(metric.__dict__)
    result = traces_to_group_eval_case(selected, metrics=[metric])
    assert isinstance(result.case.messages[0], Message)
    assert isinstance(result.case.tool_calls[0], ToolCall)
    result.case.tool_calls[0].input["key"] = "changed"
    result.case.tool_calls[0].output["nested"].append("changed")
    result.case.messages[0].metadata["nested"].append("changed")
    assert selected == before
    assert metric.__dict__ == metric_before
    assert result.case.messages[0].tool_calls[0].output == {"nested": []}


def test_selective_empty_and_telemetry_only_do_not_materialize_payload_types() -> None:
    selected = members(span(outputs=[intent("a")], input_tokens=2), tool("tool", "a", 3, 4, "result"))
    with (
        patch.object(group_adapter, "_message", side_effect=AssertionError("unexpected message projection")),
        patch.object(group_adapter, "_tool_call", side_effect=AssertionError("unexpected tool projection")),
    ):
        empty = traces_to_group_eval_case(selected, metrics=[])
        telemetry = traces_to_group_eval_case(selected, metrics=[TokenCostMetric()])
    assert empty.case.to_dict() == {"input": "", "output": ""}
    assert telemetry.case.to_dict() == {"input": "", "output": "", "token_count": 2}
    assert empty.diagnostics == telemetry.diagnostics
    assert telemetry.case.output == ""


def test_optional_fields_and_invalid_custom_declarations_fallback_to_full() -> None:
    selected = members(span(outputs=[message("assistant", "value")], input_tokens=2))
    optional = traces_to_group_eval_case(selected, metrics=[DeclaredMetric(("token_count",), ("messages",))])
    assert optional.case.messages is not None
    assert optional.case.token_count == 2
    metric = DeclaredMetric()
    metric.eval_inputs = "bad declaration"
    assert traces_to_group_eval_case(selected, metrics=[metric]).case == traces_to_group_eval_case(selected).case
    assert traces_to_group_eval_case(selected, metrics=[object()]).case == traces_to_group_eval_case(selected).case


@pytest.mark.parametrize(
    "metric",
    [TurnLatencyMetric(max_ms_per_turn=5), TurnTokenCostMetric(max_tokens_per_turn=10), TokenCostMetric(max_tokens=20)],
)
def test_full_single_and_mixed_projection_evidence_and_real_score_parity(metric: BaseMetric) -> None:
    selected = members(
        span("a", outputs=[message("assistant", "one")], duration_ms=3, input_tokens=2, output_tokens=1),
        span("b", start=3, end=4, outputs=[message("assistant", "two")], duration_ms=8, input_tokens=4),
    )
    results = [
        traces_to_group_eval_case(selected, metrics=requested)
        for requested in (None, [metric], [metric, LatencyMetric()])
    ]
    scores = [metric.measure(result.case) for result in results]
    assert all(
        score.value == scores[0].value and score.reason == scores[0].reason and score.metadata == scores[0].metadata
        for score in scores
    )
    for field in metric.eval_inputs.fields:
        assert all(getattr(result.case, field.value) == getattr(results[0].case, field.value) for result in results)
    assert results[0].diagnostics == results[1].diagnostics == results[2].diagnostics
    assert all(result.case.latency_ms is None for result in results)


def test_concurrent_calls_share_no_state_even_with_same_metric_instance() -> None:
    metric = DeclaredMetric(("messages", "tool_calls", "token_count"))

    def adapt(index: int) -> group_adapter.GroupAssembly:
        return traces_to_group_eval_case(
            members(
                span(outputs=[message("assistant", str(index))], input_tokens=index),
                tool("tool", "id", 3, 4, str(index)),
            ),
            metrics=[metric],
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(adapt, range(20)))
    for index, result in enumerate(results):
        assert result.case.messages[0].content == str(index)
        assert result.case.tool_calls[0].output == str(index)
        assert result.case.token_count == index
        assert result.diagnostics.message_count == 2


def test_safe_diagnostic_and_error_serialization_has_no_payload_ids_or_logs(caplog: pytest.LogCaptureFixture) -> None:
    sentinel = "SENSITIVE_SENTINEL"
    result = traces_to_group_eval_case(
        [
            GroupTrace(
                [
                    span(
                        None,
                        parent=sentinel,
                        start=None,
                        end=None,
                        outputs=[message("tool", sentinel, tool_call_id=sentinel)],
                        input_tokens=sentinel,
                    )
                ],
                sentinel,
            ),
            GroupTrace([], "second"),
        ]
    )
    assert sentinel not in json.dumps(asdict(result.diagnostics))
    assert sentinel not in repr(result)
    with pytest.raises(GroupInputError) as error:
        traces_to_group_eval_case([GroupTrace([span(trace=sentinel)], "mismatch"), GroupTrace([])])
    assert sentinel not in str(error.value)
    assert sentinel not in repr(error.value)
    assert sentinel not in json.dumps(vars(error.value))
    assert caplog.records == []


def test_langfuse_and_canonical_blocks_keep_roles_call_ids_and_empty_results() -> None:
    generation = span(
        "generation",
        start=0,
        end=1,
        attrs={
            "langfuse.observation.type": "generation",
            "langfuse.observation.input": {"messages": [message("developer", "policy"), message("user", "go")]},
            "langfuse.observation.output": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "a", "name": "lookup", "input": {"x": 1}}],
            },
        },
    )
    execution = span(
        "tool",
        start=2,
        end=3,
        attrs={
            "langfuse.observation.type": "tool",
            "langfuse.observation.name": "lookup",
            "gen_ai.tool.call.id": "a",
            "langfuse.observation.input": {"x": 1},
            "langfuse.observation.output": {},
        },
    )
    execution["attributes"] = json.dumps(execution["attributes"])
    result = traces_to_group_eval_case(members(generation, execution))
    assert [msg.role for msg in result.case.messages] == ["developer", "user", "assistant", "tool"]
    assert result.case.messages[2].tool_calls == [ToolCall("lookup", {"x": 1}, {})]
    assert result.case.tool_calls == [ToolCall("lookup", {"x": 1}, {})]


def test_replayed_message_timestamp_and_tool_ownership_are_shuffle_invariant() -> None:
    raw = intent(None, id="intent-event")
    rows = [
        span("parent", start=0, end=10, outputs=[raw]),
        span("history", start=20, end=21, inputs=[deepcopy(raw)]),
        tool("execution", None, 1, 2, "result", parent="parent"),
    ]
    forward = traces_to_group_eval_case(members(*rows))
    backward = traces_to_group_eval_case(members(*reversed(rows)))
    assert forward.case == backward.case
    assert forward.diagnostics == backward.diagnostics
    assert forward.case.messages[0].tool_calls[0].output == "result"


def test_replayed_message_id_uses_deterministic_event_time() -> None:
    raw = message("assistant", "shared", id="m")
    rows = [
        span("output", start=0, end=1, outputs=[raw]),
        span("history", start=10, end=11, inputs=[deepcopy(raw)]),
        span("other", start=2, end=3, outputs=[message("assistant", "middle")]),
    ]
    forward = traces_to_group_eval_case(members(*rows))
    backward = traces_to_group_eval_case(members(*reversed(rows)))
    assert forward.case == backward.case
    assert [msg.content for msg in forward.case.messages] == ["shared", "middle"]


def test_conversation_judge_sees_identical_real_evidence_for_all_projections() -> None:
    judge = MockLLM(default={"score": 0.8, "reasoning": "observed evidence"})
    metric = ToolUseMetric(llm=judge)
    selected = members(
        span("intent", start=0, end=1, inputs=[message("user", "request")], outputs=[intent("a")]),
        tool("execution", "a", 2, 3, "observed result"),
    )
    assemblies = [
        traces_to_group_eval_case(selected, metrics=requested)
        for requested in (None, [metric], [metric, LatencyMetric()])
    ]
    assert judge.prompts == []  # Adaptation does not call the judge.
    scores = [metric.measure(result.case) for result in assemblies]
    assert judge.prompts[0] == judge.prompts[1] == judge.prompts[2]
    assert "request" in judge.prompts[0] and "observed result" in judge.prompts[0]
    assert all(score.value == 0.8 and score.metadata["n_tool_calls"] == 1 for score in scores)
    assert all(result.case.messages == assemblies[0].case.messages for result in assemblies)
    assert all(result.case.tool_calls == assemblies[0].case.tool_calls for result in assemblies)
    assert assemblies[0].case.messages[1].metadata["tool_call_ids"] == ["a"]


def test_output_duration_used_when_end_timestamp_is_missing() -> None:
    rows = [tool("a", "a", 1, 10, "a"), tool("b", "b", 2, 3, "b")]
    for row in rows:
        row["duration_ms"] = (row["end_timestamp"] - row["start_timestamp"]) * 1000
        row["end_timestamp"] = None
    result = traces_to_group_eval_case(members(*rows))
    assert [call.output for call in result.case.tool_calls] == ["a", "b"]
    assert [msg.content for msg in result.case.messages] == ["b", "a"]


@pytest.mark.parametrize("output", ["", {}])
def test_canonical_tool_result_parts_are_one_result_not_two(output: Any) -> None:
    raw = span(
        "execution",
        inputs=[
            {"role": "assistant", "parts": [{"type": "tool_call", "id": "a", "name": "lookup", "arguments": {"x": 1}}]}
        ],
        outputs=[{"role": "tool", "parts": [{"type": "tool_call_response", "id": "a", "result": output}]}],
        attrs={"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "lookup"},
    )
    result = traces_to_group_eval_case(members(raw))
    assert result.case.tool_calls == [ToolCall("lookup", {"x": 1}, output)]
    assert result.case.messages[0].tool_calls[0].output == output
    assert result.case.messages[1].role == "tool"
    assert GroupIssue.AMBIGUOUS_TOOL_LINK not in issues(result)


def test_unique_structural_child_result_can_link_to_parent_execution() -> None:
    rows = [
        tool("execution", None, 1, 10, None),
        span("result", parent="execution", start=2, end=3, outputs=[message("tool", "result")]),
    ]
    result = traces_to_group_eval_case(members(*rows))
    assert result.case.tool_calls[0].output == "result"
    assert GroupIssue.UNMATCHED_TOOL_RESULT not in issues(result)


def test_unidentified_tool_history_is_not_current_execution_result() -> None:
    raw = tool("execution", "new", 1, 2, None)
    raw["attributes"]["gen_ai.input.messages"] = [message("tool", "historical")]
    result = traces_to_group_eval_case(members(raw))
    assert result.case.tool_calls[0].output is None
    assert result.case.messages[0].content == "historical"
    assert GroupIssue.UNMATCHED_TOOL_RESULT in issues(result)


def test_empty_self_parent_cycle_does_not_claim_known_absence() -> None:
    result = traces_to_group_eval_case(members(span("self", parent="self"), complete=True))
    assert GroupIssue.CYCLIC_ORDER in issues(result)
    assert result.case.messages is result.case.tool_calls is None


def test_replayed_direct_message_usage_is_one_event_across_span_captures() -> None:
    raw = message("assistant", "observed", id="m", input_tokens=7, cost_usd=0.25)
    rows = [span("a", outputs=[raw]), span("b", inputs=[deepcopy(raw)])]
    result = traces_to_group_eval_case(members(*rows))
    assert result.case.input_tokens == 7
    assert result.case.cost_usd == 0.25
    assert result.diagnostics.input_tokens.selected == 1
    assert result.diagnostics.input_tokens.observed == 1
    assert len(result.case.messages) == 1
    assert traces_to_group_eval_case(members(*reversed(rows))).case == result.case


def test_replayed_message_does_not_recover_invalid_span_column() -> None:
    raw = message("assistant", "observed", id="m", input_tokens=7)
    result = traces_to_group_eval_case(
        members(span("a", outputs=[raw], input_tokens="invalid"), span("b", inputs=[deepcopy(raw)]))
    )
    assert result.case.input_tokens is None
    assert result.diagnostics.input_tokens.invalid == 1
    assert result.diagnostics.input_tokens.overlapping == 1


def test_partial_direct_message_usage_retains_known_subtotal_and_invalid_coverage() -> None:
    raw = span(
        outputs=[
            message("assistant", "valid", id="a", input_tokens=5),
            message("assistant", "invalid", id="b", input_tokens=1.5),
        ]
    )
    result = traces_to_group_eval_case(members(raw))
    assert result.case.input_tokens == 5
    assert result.diagnostics.input_tokens.selected == result.diagnostics.input_tokens.invalid == 1
    assert GroupIssue.INVALID_TELEMETRY in issues(result)


def test_shared_message_does_not_receive_arbitrary_span_telemetry() -> None:
    raw = message("assistant", "observed", id="m")
    rows = [
        span("a", outputs=[raw], input_tokens=7, duration_ms=1),
        span("b", outputs=[deepcopy(raw)], input_tokens=9, duration_ms=2),
    ]
    first = traces_to_group_eval_case(members(*rows))
    second = traces_to_group_eval_case(members(*reversed(rows)))
    assert first.case == second.case
    assert first.case.messages[0].token_count is first.case.messages[0].latency_ms is None


@pytest.mark.parametrize(
    ("value", "expected"), [("9007199254740993", 9007199254740993), (1.0, 1), ("01", 1), ("١٢", 12), (10**400, 10**400)]
)
def test_integral_token_reports_preserve_exact_values(value: Any, expected: int) -> None:
    result = traces_to_group_eval_case(members(span(input_tokens=value, outputs=[message("assistant", "answer")])))
    assert result.case.input_tokens == result.case.messages[0].token_count == expected


def test_invalid_metadata_telemetry_cannot_bypass_typed_validation() -> None:
    raw = span(
        start=None,
        end=None,
        outputs=[
            message("assistant", "value", metadata={"latency_ms": float("inf"), "token_count": True, "cost_usd": -1})
        ],
    )
    result = traces_to_group_eval_case(members(raw))
    observed = result.case.messages[0]
    assert observed.latency_ms is observed.token_count is observed.cost_usd is observed.metadata is None


@pytest.mark.parametrize("attributes", [None, [], 1, True, "malformed", "[]", "1", "null", '"string"', ""])
def test_malformed_attributes_normalize_to_absent_without_losing_other_fields(attributes: Any) -> None:
    raw = span(
        "tool", start=0, end=None, duration_ms=0, tool_name="narrow-tool", input_tokens=0, output_tokens=0, cost_usd=0
    )
    raw["attributes"] = attributes
    selected = members(raw, span("message", outputs=[message("assistant", "kept")]))
    before = deepcopy(selected)
    result = traces_to_group_eval_case(selected)
    assert result.case.tool_calls == [ToolCall("narrow-tool")]
    assert result.case.messages == [Message("assistant", "kept", latency_ms=1000)]
    assert result.case.input_tokens == result.case.output_tokens == result.case.token_count == result.case.cost_usd == 0
    assert result.diagnostics.span_count == 2
    assert GroupIssue.UNKNOWN_TIMESTAMP not in issues(result)
    assert selected == before


def test_unknown_message_identity_excludes_ambiguous_history_usage_only() -> None:
    raw = span(
        inputs=[message("assistant", "history", input_tokens=50)],
        outputs=[message("assistant", "new", id="new", input_tokens=5)],
    )
    result = traces_to_group_eval_case(members(raw))
    assert result.case.input_tokens == 5
    assert result.diagnostics.input_tokens.ambiguous == 1
    assert len(result.case.messages) == 2


@pytest.mark.parametrize(("context", "expected"), [(["chunk", ""], "answer"), ([], ""), (None, None)])
def test_explicit_message_rag_fields_are_typed_owned_and_projection_invariant(
    context: list[str] | None, expected: str | None
) -> None:
    selected = members(span(outputs=[message("assistant", "observed", retrieval_context=context, expected=expected)]))
    before = deepcopy(selected)
    assemblies = [
        traces_to_group_eval_case(selected, metrics=requested)
        for requested in (None, [DeclaredMetric(("messages",))], [DeclaredMetric((), ("messages",)), TokenCostMetric()])
    ]
    for assembly in assemblies:
        observed = assembly.case.messages[0]
        assert isinstance(observed, Message)
        assert observed.retrieval_context == context
        assert observed.expected == expected
        assert assembly.case.context is assembly.case.expected is None
        assert observed.to_dict().get("retrieval_context") == context
        assert observed.to_dict().get("expected") == expected
    assert assemblies[0].case.messages == assemblies[1].case.messages == assemblies[2].case.messages
    if context is not None:
        assemblies[0].case.messages[0].retrieval_context.append("output mutation")
        assert assemblies[1].case.messages[0].retrieval_context == context
    assert selected == before


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("retrieval_context", "not-a-list"),
        ("retrieval_context", ("chunk",)),
        ("retrieval_context", ["valid", 1]),
        ("retrieval_context", {}),
        ("expected", []),
        ("expected", 1),
        ("expected", True),
    ],
)
def test_invalid_message_rag_fields_are_dropped_without_losing_valid_evidence(field: str, invalid: Any) -> None:
    raw = message("assistant", "observed", retrieval_context=["valid context"], expected="valid expected")
    raw[field] = invalid
    selected = members(span(outputs=[raw]), complete=True)
    full = traces_to_group_eval_case(selected)
    projected = traces_to_group_eval_case(selected, metrics=[DeclaredMetric(("messages",))])
    empty = traces_to_group_eval_case(selected, metrics=[])
    observed = full.case.messages[0]
    assert getattr(observed, field) is None
    assert observed.content == "observed"
    if field == "expected":
        assert observed.retrieval_context == ["valid context"]
    else:
        assert observed.expected == "valid expected"
    assert full.case.messages == projected.case.messages
    assert full.diagnostics == projected.diagnostics == empty.diagnostics
    assert GroupIssue.UNSUPPORTED_EVIDENCE in issues(full)
    assert full.diagnostics.complete_member_count == full.diagnostics.incomplete_member_count == 1


@pytest.mark.parametrize("capture_complete", [True, False, None])
@pytest.mark.parametrize(
    ("rows", "issue"),
    [
        ([span(parent="missing")], GroupIssue.MISSING_PARENT),
        ([span("self", parent="self")], GroupIssue.CYCLIC_ORDER),
        ([span("a", parent="b"), span("b", parent="a")], GroupIssue.CYCLIC_ORDER),
        ([span(attrs={"gen_ai.output.messages": [1]})], GroupIssue.UNSUPPORTED_EVIDENCE),
        ([span(attrs={"gen_ai.output.messages": "malformed"})], GroupIssue.UNSUPPORTED_EVIDENCE),
        ([span(attrs={"langfuse.observation.output": [1]})], GroupIssue.UNSUPPORTED_EVIDENCE),
        ([span(attrs={"langfuse.observation.output": {"messages": 1}})], GroupIssue.UNSUPPORTED_EVIDENCE),
        ([span(outputs=[message("assistant", "kept", tool_calls=[1])])], GroupIssue.UNSUPPORTED_EVIDENCE),
        ([span(outputs=[message("assistant", "kept", tool_calls={})])], GroupIssue.UNSUPPORTED_EVIDENCE),
        ([tool("execution", "call", 1, 2, [])], GroupIssue.UNSUPPORTED_EVIDENCE),
        (
            [tool("execution", "call", 1, 2, None), span("result", outputs=[message("tool", 1, tool_call_id="call")])],
            GroupIssue.UNSUPPORTED_EVIDENCE,
        ),
    ],
)
def test_contradictions_downgrade_only_affected_explicit_complete_member(
    rows: list[dict[str, Any]], issue: GroupIssue, capture_complete: bool | None
) -> None:
    selected = [
        GroupTrace(rows, "affected", capture_complete),
        GroupTrace([], "complete", True),
        GroupTrace([], "unknown"),
        GroupTrace([], "incomplete", False),
    ]
    full = traces_to_group_eval_case(selected)
    empty = traces_to_group_eval_case(selected, metrics=[])
    assert issue in issues(full)
    assert full.diagnostics == empty.diagnostics
    assert full.diagnostics.complete_member_count == 1
    assert full.diagnostics.incomplete_member_count == 1 + (capture_complete is not None)
    assert full.diagnostics.unknown_member_count == 1 + (capture_complete is None)
    assert full.diagnostics.member_count == 4


@pytest.mark.parametrize(
    "row", [span(parent="missing"), span("self", parent="self"), span(attrs={"gen_ai.output.messages": [1]})]
)
def test_contradicted_complete_member_cannot_establish_known_empty_evidence(row: dict[str, Any]) -> None:
    result = traces_to_group_eval_case(members(row, complete=True))
    assert result.case.messages is result.case.tool_calls is None
    assert result.diagnostics.complete_member_count == result.diagnostics.incomplete_member_count == 1
    assert result.diagnostics.unknown_member_count == 0


@pytest.mark.parametrize("serialized", [False, True])
def test_valid_empty_message_arrays_do_not_contradict_explicit_complete_capture(serialized: bool) -> None:
    value = "[]" if serialized else []
    raw = span(attrs={"gen_ai.input.messages": value, "gen_ai.output.messages": value})
    result = traces_to_group_eval_case(members(raw, complete=True))
    assert result.case.messages == result.case.tool_calls == []
    assert result.diagnostics.complete_member_count == 2
    assert result.diagnostics.incomplete_member_count == 0
    assert GroupIssue.UNSUPPORTED_EVIDENCE not in issues(result)


def test_connected_tree_does_not_establish_capture_completeness() -> None:
    result = traces_to_group_eval_case(members(span("root"), span("child", parent="root")))
    assert result.diagnostics.complete_member_count == result.diagnostics.incomplete_member_count == 0
    assert result.diagnostics.unknown_member_count == 2
    assert result.case.messages is result.case.tool_calls is None


@pytest.mark.parametrize("invalid", ["1.0", "1e0", "+1", " 1", "²"])
def test_message_token_validation_matches_span_token_validation(invalid: str) -> None:
    raw = span(
        start=None,
        end=None,
        outputs=[
            message(
                "assistant",
                "observed",
                token_count=invalid,
                input_tokens=invalid,
                output_tokens=0,
                metadata={"token_count": invalid},
            )
        ],
    )
    result = traces_to_group_eval_case(members(raw))
    assert result.case.input_tokens is None
    assert result.case.token_count == result.case.output_tokens == 0
    assert result.case.messages[0].token_count is result.case.messages[0].metadata is None
    assert result.diagnostics.input_tokens.invalid == 1


def test_finite_cost_reports_overflow_remains_unknown_not_infinite() -> None:
    result = traces_to_group_eval_case(members(span("a", cost_usd=1e308), span("b", cost_usd=1e308)))
    assert result.case.cost_usd is None
    assert result.diagnostics.cost_usd.observed == result.diagnostics.cost_usd.selected == 2
    assert GroupIssue.INVALID_TELEMETRY in issues(result)


@pytest.mark.parametrize("row", [None, [], "SENSITIVE_SENTINEL", 1])
def test_malformed_span_structure_still_raises_static_safe_error(row: Any, caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(GroupInputError) as error:
        traces_to_group_eval_case([GroupTrace([row], "one"), GroupTrace([], "two")])
    assert error.value.code is GroupErrorCode.INVALID_INPUT
    assert "SENSITIVE_SENTINEL" not in str(error.value)
    assert caplog.records == []


@pytest.mark.parametrize("call_id", [None, "shared-call"])
def test_shared_canonical_result_is_not_assigned_to_multiple_executions(call_id: str | None) -> None:
    raw = message("tool", "once", id="shared-result")
    if call_id is not None:
        raw["tool_call_id"] = call_id
    rows = [
        span(
            sid, outputs=[deepcopy(raw)], attrs={"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "lookup"}
        )
        for sid in ("x", "y")
    ]
    forward = traces_to_group_eval_case(members(*rows, complete=True))
    backward = traces_to_group_eval_case(members(*reversed(rows), complete=True))
    assert forward.case == backward.case
    assert forward.diagnostics == backward.diagnostics
    assert forward.case.tool_calls == [ToolCall("lookup"), ToolCall("lookup")]
    assert [(msg.role, msg.content) for msg in forward.case.messages] == [("tool", "once")]
    assert forward.diagnostics.message_count == 1
    assert GroupIssue.AMBIGUOUS_TOOL_LINK in issues(forward)


@pytest.mark.parametrize("contradiction", ["name", "arguments"])
@pytest.mark.parametrize("location", ["local", "parent"])
def test_incompatible_historical_intent_cannot_supply_execution_identity_or_arguments(
    contradiction: str, location: str
) -> None:
    old_call = {"name": "old_tool" if contradiction == "name" else "lookup", "arguments": {"old": True}}
    if location == "local":
        old_call["id"] = "old"
    old = message("assistant", None, tool_calls=[old_call])
    attrs: dict[str, Any] = {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": "lookup",
        "gen_ai.tool.call.result": "new",
    }
    if contradiction == "arguments":
        attrs["gen_ai.tool.call.arguments"] = {"new": True}
    execution = span("execution", inputs=[old] if location == "local" else None, attrs=attrs)
    rows = [execution]
    if location == "parent":
        execution["parent_span_id"] = "parent"
        rows.append(span("parent", outputs=[old]))
    result = traces_to_group_eval_case(members(*rows))
    assistant = next(msg for msg in result.case.messages if msg.role == "assistant")
    assert assistant.tool_calls == [ToolCall(old_call["name"], {"old": True})]
    assert result.case.tool_calls == [
        ToolCall("lookup", {"new": True} if contradiction == "arguments" else None, "new")
    ]
    tool_result = next(msg for msg in result.case.messages if msg.role == "tool")
    # The result has its execution-derived message identity, but no call
    # identity may be inferred from incompatible history.
    assert "tool_call_id" not in tool_result.metadata
    assert tool_result.metadata["source_span_id"] == "execution"
    assert GroupIssue.AMBIGUOUS_TOOL_LINK in issues(result)


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_cycle_recovery_preserves_downstream_intent_execution_order(order: tuple[int, ...]) -> None:
    rows = [
        span("a", parent="b", outputs=[intent("c")]),
        span("b", parent="a"),
        tool("0-child", "c", 1, 2, "result", parent="a"),
    ]
    expected = traces_to_group_eval_case(members(*rows, complete=True))
    result = traces_to_group_eval_case(members(*(rows[index] for index in order), complete=True))
    assert result.case == expected.case
    assert result.diagnostics == expected.diagnostics
    assert [msg.role for msg in result.case.messages] == ["assistant", "tool"]
    assert result.case.messages[0].tool_calls[0].output == "result"
    assert result.case.tool_calls[0].output == "result"
    assert GroupIssue.CYCLIC_ORDER in issues(result)
    assert result.diagnostics.complete_member_count == result.diagnostics.incomplete_member_count == 1


def test_deep_cycle_recovery_is_iterative_and_retains_evidence() -> None:
    count = 1100
    rows = [span(f"s{index}", parent=f"s{(index + 1) % count}") for index in range(count)]
    rows[-1]["attributes"]["gen_ai.output.messages"] = [message("assistant", "kept")]
    result = traces_to_group_eval_case(members(*rows, complete=True))
    assert [msg.content for msg in result.case.messages] == ["kept"]
    assert result.diagnostics.span_count == count
    assert GroupIssue.CYCLIC_ORDER in issues(result)


@pytest.mark.parametrize("output", ["answer", '"answer"', "42", "true", 0, 1.5, True])
def test_roleless_langfuse_generation_output_is_unknown_not_complete_empty(output: Any) -> None:
    raw = span(attrs={"langfuse.observation.type": "generation", "langfuse.observation.output": output})
    selected = members(raw, complete=True)
    full = traces_to_group_eval_case(selected)
    projected = traces_to_group_eval_case(selected, metrics=[DeclaredMetric(("messages", "tool_calls"))])
    empty = traces_to_group_eval_case(selected, metrics=[])
    assert full.case.messages is full.case.tool_calls is None
    assert projected.case == full.case
    assert full.diagnostics == projected.diagnostics == empty.diagnostics
    assert GroupIssue.UNSUPPORTED_EVIDENCE in issues(full)
    assert full.diagnostics.complete_member_count == full.diagnostics.incomplete_member_count == 1
    assert full.diagnostics.message_count == 0


@pytest.mark.parametrize("output", ["A", "", {}])
def test_conflicting_local_result_cannot_overwrite_explicit_span_result(output: Any) -> None:
    raw = tool("execution", "c", 1, 2, output)
    raw["attributes"]["gen_ai.output.messages"] = [message("tool", "B", tool_call_id="c")]
    for requested in (None, [], [DeclaredMetric(("tool_calls",))]):
        with pytest.raises(GroupInputError) as error:
            traces_to_group_eval_case(members(raw), metrics=requested)
        assert error.value.code is GroupErrorCode.CONFLICTING_IDENTITY
        assert str(error.value) == "Captures with the same identity disagree."


@pytest.mark.parametrize("output", ["A", "", {}])
def test_matching_local_result_and_explicit_span_result_remain_supported(output: Any) -> None:
    raw = tool("execution", "c", 1, 2, output)
    raw["attributes"]["gen_ai.output.messages"] = [
        {"role": "tool", "parts": [{"type": "tool_call_response", "id": "c", "result": output}]}
    ]
    result = traces_to_group_eval_case(members(raw))
    assert result.case.tool_calls == [ToolCall("lookup", {"key": "same"}, output)]
    assert [msg.role for msg in result.case.messages] == ["tool"]
    assert GroupIssue.AMBIGUOUS_TOOL_LINK not in issues(result)
