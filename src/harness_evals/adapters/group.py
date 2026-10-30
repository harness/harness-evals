"""Pure adaptation of two through 64 already-selected traces into Group evidence.

Conversation and execution evidence lives in ``messages`` and ``tool_calls``;
``input`` and ``output`` are deliberately empty placeholders, not an answer.
This module does not select members, invoke targets, or enable Group metrics.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, MetricInput, MetricInputs
from harness_evals.core.types import Message, ToolCall


class GroupErrorCode(str, Enum):
    """Stable, payload-free input rejection codes."""

    MEMBER_COUNT = "member_count"
    INVALID_INPUT = "invalid_input"
    TRACE_SCOPE = "trace_scope"
    REPEATED_TRACE = "repeated_trace"
    CONFLICTING_IDENTITY = "conflicting_identity"


_ERROR_MESSAGES = {
    GroupErrorCode.MEMBER_COUNT: "A Group requires between 2 and 64 member traces.",
    GroupErrorCode.INVALID_INPUT: "Group input has an unsupported structure.",
    GroupErrorCode.TRACE_SCOPE: "A member contains inconsistent trace scope.",
    GroupErrorCode.REPEATED_TRACE: "A known trace occurs in more than one member.",
    GroupErrorCode.CONFLICTING_IDENTITY: "Captures with the same identity disagree.",
}


class GroupInputError(ValueError):
    """Rejected input, with a fixed code and static, safe error message."""

    def __init__(self, code: GroupErrorCode) -> None:
        self.code = GroupErrorCode(code)
        super().__init__(_ERROR_MESSAGES[self.code])


class GroupIssue(str, Enum):
    """Observational limitations; never source identifiers or payloads."""

    MISSING_TRACE_ID = "missing_trace_id"
    MISSING_SPAN_ID = "missing_span_id"
    AMBIGUOUS_MESSAGE_IDENTITY = "ambiguous_message_identity"
    DUPLICATE_CAPTURE = "duplicate_capture"
    MISSING_PARENT = "missing_parent"
    CYCLIC_ORDER = "cyclic_order"
    UNKNOWN_TIMESTAMP = "unknown_timestamp"
    INVALID_TELEMETRY = "invalid_telemetry"
    UNSUPPORTED_EVIDENCE = "unsupported_evidence"
    AMBIGUOUS_TOOL_LINK = "ambiguous_tool_link"
    # Stable presentation order is not proof of execution order.
    AMBIGUOUS_TOOL_ORDER = "ambiguous_tool_order"
    UNMATCHED_TOOL_RESULT = "unmatched_tool_result"
    AMBIGUOUS_USAGE = "ambiguous_usage"


@dataclass(frozen=True)
class GroupIssueCount:
    issue: GroupIssue
    count: int


@dataclass(frozen=True)
class GroupUsageCoverage:
    """Per-dimension event coverage, not a claim of capture completeness.

    A report is a canonical span observation or a direct message observation;
    identifiable message replays count once. ``observed`` counts valid reports
    before exclusions, including zero. These partition into ``selected``,
    ``overlapping`` and ``ambiguous``. ``invalid`` counts rejected reports;
    ``missing`` counts canonical spans with no report for this dimension.
    """

    observed: int = 0
    selected: int = 0
    missing: int = 0
    invalid: int = 0
    overlapping: int = 0
    ambiguous: int = 0


@dataclass(frozen=True)
class GroupDiagnostics:
    """Payload-free evidence counts and effective source completeness assertions.

    ``complete_member_count`` counts explicit ``capture_complete=True`` assertions
    not contradicted by missing parents, cyclic order or unsupported evidence.
    Contradicted True assertions join explicit False assertions in
    ``incomplete_member_count``. None assertions stay unknown, even for connected
    trees; neither usable evidence nor valid usage establishes completeness.
    """

    member_count: int
    span_count: int
    duplicate_span_count: int
    message_count: int
    tool_execution_count: int
    complete_member_count: int
    incomplete_member_count: int
    unknown_member_count: int
    issues: tuple[GroupIssueCount, ...]
    input_tokens: GroupUsageCoverage
    output_tokens: GroupUsageCoverage
    cost_usd: GroupUsageCoverage


@dataclass(frozen=True)
class GroupTrace:
    """One member's span capture with an optional source completeness assertion.

    None is unknown, not inferred from connectivity. True can establish known
    empty evidence only when no observed structural/evidence issue contradicts
    it; such contradictions downgrade the assertion to incomplete diagnostics.
    """

    spans: Sequence[dict[str, Any]] = field(repr=False)
    trace_id: str | None = field(default=None, repr=False)
    capture_complete: bool | None = None


@dataclass(frozen=True)
class GroupAuthoredData:
    """Explicit expectations and labels, independent of observed evidence.

    Construction snapshots mutable values; adaptation takes another snapshot
    so neither the caller nor the returned EvalCase shares mutable data.
    """

    expected_tools: list[str] | None = field(default=None, repr=False)
    expected_tool_calls: list[ToolCall] | None = field(default=None, repr=False)
    metadata: dict[str, Any] | None = field(default=None, repr=False)
    tags: dict[str, str] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        try:
            for name in ("expected_tools", "expected_tool_calls", "metadata", "tags"):
                object.__setattr__(self, name, deepcopy(getattr(self, name)))
            if self.expected_tools is not None and (
                not isinstance(self.expected_tools, list)
                or not all(isinstance(name, str) for name in self.expected_tools)
            ):
                raise GroupInputError(GroupErrorCode.INVALID_INPUT)
            if self.metadata is not None and not isinstance(self.metadata, dict):
                raise GroupInputError(GroupErrorCode.INVALID_INPUT)
            if self.tags is not None and (
                not isinstance(self.tags, dict)
                or not all(isinstance(key, str) and isinstance(value, str) for key, value in self.tags.items())
            ):
                raise GroupInputError(GroupErrorCode.INVALID_INPUT)
            if self.expected_tool_calls is not None:
                if not isinstance(self.expected_tool_calls, list):
                    raise GroupInputError(GroupErrorCode.INVALID_INPUT)
                calls = [
                    call if isinstance(call, ToolCall) else ToolCall.from_dict(call)
                    for call in self.expected_tool_calls
                ]
                if not all(
                    isinstance(call.name, str)
                    and (call.input is None or isinstance(call.input, dict))
                    and (call.output is None or isinstance(call.output, (str, dict)))
                    for call in calls
                ):
                    raise GroupInputError(GroupErrorCode.INVALID_INPUT)
                object.__setattr__(self, "expected_tool_calls", calls)
        except Exception:
            raise GroupInputError(GroupErrorCode.INVALID_INPUT) from None


@dataclass(frozen=True)
class GroupAssembly:
    case: EvalCase = field(repr=False)
    diagnostics: GroupDiagnostics


def _requested_fields(metrics: Iterable[BaseMetric] | None) -> set[MetricInput]:
    if metrics is None:
        return set(MetricInput)
    fields: set[MetricInput] = set()
    for metric in metrics:
        try:
            declaration = getattr(metric, "eval_inputs", None)
            if not isinstance(declaration, MetricInputs) or not all(
                isinstance(value, MetricInput) for value in declaration.fields
            ):
                return set(MetricInput)
            fields.update(declaration.fields)
        except Exception:
            return set(MetricInput)
    return fields


def _tool_call(data: dict[str, Any]) -> ToolCall:
    return ToolCall(name=data["name"], input=deepcopy(data["input"]), output=deepcopy(data["output"]))


def _message(data: dict[str, Any]) -> Message:
    return Message(
        role=data["role"],
        content=data["content"],
        tool_calls=[_tool_call(call) for call in data["tool_calls"]] if data["tool_calls"] is not None else None,
        latency_ms=data["latency_ms"],
        token_count=data["token_count"],
        cost_usd=data["cost_usd"],
        metadata=deepcopy(data["metadata"]),
        retrieval_context=deepcopy(data["retrieval_context"]),
        expected=data["expected"],
    )


def traces_to_group_eval_case(
    members: Sequence[GroupTrace],
    *,
    authored: GroupAuthoredData | None = None,
    metrics: Iterable[BaseMetric] | None = None,
) -> GroupAssembly:
    """Snapshot, assemble canonical evidence, then project requested fields.

    ``None`` metrics requests all supported fields; an empty iterable requests
    only placeholders. Required and optional ``MetricInputs`` are both read.
    Undeclared or invalid declarations conservatively request all fields.
    Diagnostics are independent of this projection. Missing evidence remains
    ``None``; explicitly complete captures can establish empty evidence lists.
    """
    from harness_evals.adapters._group import assemble

    if not isinstance(members, Sequence) or isinstance(members, (str, bytes)):
        raise GroupInputError(GroupErrorCode.INVALID_INPUT)
    if not 2 <= len(members) <= 64:
        raise GroupInputError(GroupErrorCode.MEMBER_COUNT)
    if authored is not None and not isinstance(authored, GroupAuthoredData):
        raise GroupInputError(GroupErrorCode.INVALID_INPUT)
    try:
        snapshot = deepcopy(members)
        authored_snapshot = deepcopy(authored)
    except Exception:
        raise GroupInputError(GroupErrorCode.INVALID_INPUT) from None
    try:
        evidence = assemble(snapshot)
    except GroupInputError:
        raise
    except Exception:
        # Malformed source values must never escape as payload-bearing errors.
        raise GroupInputError(GroupErrorCode.INVALID_INPUT) from None
    fields = _requested_fields(metrics)
    case = EvalCase(input="", output="")
    if MetricInput.MESSAGES in fields and evidence.messages is not None:
        case.messages = [_message(message) for message in evidence.messages]
    if MetricInput.TOOL_CALLS in fields and evidence.tool_calls is not None:
        case.tool_calls = [_tool_call(call) for call in evidence.tool_calls]
    for name, value in evidence.usage.items():
        if MetricInput(name) in fields:
            setattr(case, name, value)
    if authored_snapshot is not None:
        for name in ("expected_tools", "expected_tool_calls", "metadata", "tags"):
            if MetricInput(name) in fields:
                setattr(case, name, getattr(authored_snapshot, name))
    return GroupAssembly(case=case, diagnostics=evidence.diagnostics)
