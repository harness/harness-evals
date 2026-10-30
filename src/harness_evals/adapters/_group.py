"""Private canonical Group evidence graph and nonoverlapping usage frontier.

Nodes retain identity and source coordinates until after causal ordering and
linking. No Message/ToolCall is materialized here, and no Trace assembly logic
is reused. All state belongs to one adaptation call.
"""

from __future__ import annotations

import heapq
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from harness_evals.adapters.group import (
    GroupDiagnostics,
    GroupErrorCode,
    GroupInputError,
    GroupIssue,
    GroupIssueCount,
    GroupTrace,
    GroupUsageCoverage,
)
from harness_evals.adapters.trace import (
    SpanType,
    _langfuse_messages,
    _message_blocks,
    _message_text,
    _nonnegative_finite_number,
    _parse_json,
    _parse_messages,
    _parse_token_count,
    _raw_token_counts,
    _span_start_seconds,
    classify_span,
    normalize_span,
)

_DIMENSIONS = ("input_tokens", "output_tokens", "cost_usd")


@dataclass
class _Node:
    key: tuple[Any, ...]
    member_index: int
    message: dict[str, Any] | None = None
    execution: dict[str, Any] | None = None
    edges: set[int] = field(default_factory=set)


@dataclass
class _Span:
    row: dict[str, Any]
    scope: tuple[str, str]
    sid: str | None
    parent: str | None
    coordinate: tuple[int, int]
    start: float | None
    end: float | None
    start_node: int = -1
    end_node: int = -1
    inputs: list[int] = field(default_factory=list)
    outputs: list[int] = field(default_factory=list)
    completions: list[int] = field(default_factory=list)

    @property
    def identity(self) -> tuple[tuple[str, str], str] | None:
        return (self.scope, self.sid) if self.sid is not None else None


@dataclass
class _Call:
    span: int
    node: int
    call_id: str | None
    data: dict[str, Any]


@dataclass
class _Result:
    span: int
    node: int
    call_id: str | None
    value: Any
    is_output: bool


@dataclass
class _Evidence:
    messages: list[dict[str, Any]] | None
    tool_calls: list[dict[str, Any]] | None
    usage: dict[str, int | float | None]
    diagnostics: GroupDiagnostics


def _id(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _first(source: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in source and source[key] is not None:
            return source[key]
    return None


def _span_usage(row: dict[str, Any], dimension: str) -> Any:
    if dimension == "cost_usd":
        raw = row.get("cost_usd")
        return row["attributes"].get("gen_ai.usage.cost") if raw is None else raw
    return _raw_token_counts(row)[0 if dimension == "input_tokens" else 1]


def _message_usage(message: dict[str, Any], dimension: str) -> Any:
    raw = message.get(dimension)
    if raw is None and isinstance(message.get("usage"), dict):
        raw = message["usage"].get(dimension)
    if raw is None and dimension == "cost_usd" and isinstance(message.get("metadata"), dict):
        raw = message["metadata"].get(dimension)
    return raw


def _same(left: Any, right: Any) -> bool:
    try:
        return bool(left == right)
    except Exception:
        return False


def _require_same(left: Any, right: Any) -> None:
    if not _same(left, right):
        raise GroupInputError(GroupErrorCode.CONFLICTING_IDENTITY)


def _compatible_call_data(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Known names/arguments constrain a link; they never establish identity."""
    return not (
        (left["name"] and right["name"] and left["name"] != right["name"])
        or (left["input"] is not None and right["input"] is not None and not _same(left["input"], right["input"]))
    )


class _Builder:
    def __init__(self) -> None:
        self.issues: Counter[GroupIssue] = Counter()
        self.spans: list[_Span] = []
        self.by_identity: dict[tuple[tuple[str, str], str], int] = {}
        self.nodes: list[_Node] = []
        self.message_ids: dict[tuple[Any, ...], tuple[dict[str, Any], int]] = {}
        self.intents: list[_Call] = []
        self.executions: list[_Call] = []
        self.results: list[_Result] = []
        self.duplicate_spans = 0
        self.message_owners: dict[int, set[int]] = defaultdict(set)
        self.raw_messages: dict[int, dict[str, Any]] = {}
        self.ambiguous_execution_spans: set[int] = set()
        self.linked_intents: dict[int, list[_Call]] = defaultdict(list)
        self.contradicted_members: set[int] = set()

    def issue(self, issue: GroupIssue, member_index: int | None = None) -> None:
        self.issues[issue] += 1
        if member_index is not None and issue in (
            GroupIssue.MISSING_PARENT,
            GroupIssue.CYCLIC_ORDER,
            GroupIssue.UNSUPPORTED_EVIDENCE,
        ):
            self.contradicted_members.add(member_index)

    def capture(self, members: Sequence[GroupTrace]) -> None:
        trace_ids: set[str] = set()
        for member_index, member in enumerate(members):
            if (
                not isinstance(member, GroupTrace)
                or not isinstance(member.spans, Sequence)
                or isinstance(member.spans, (str, bytes))
                or (member.trace_id is not None and _id(member.trace_id) is None)
                or (member.capture_complete is not None and not isinstance(member.capture_complete, bool))
            ):
                raise GroupInputError(GroupErrorCode.INVALID_INPUT)
            rows = []
            known = {member.trace_id} if member.trace_id is not None else set()
            for row in member.spans:
                if not isinstance(row, dict):
                    raise GroupInputError(GroupErrorCode.INVALID_INPUT)
                row = normalize_span(row)
                raw_trace = row.get("trace_id")
                if raw_trace is not None and _id(raw_trace) is None:
                    raise GroupInputError(GroupErrorCode.INVALID_INPUT)
                if raw_trace is not None:
                    known.add(raw_trace)
                rows.append(row)
            if len(known) > 1:
                raise GroupInputError(GroupErrorCode.TRACE_SCOPE)
            trace_id = next(iter(known), None)
            if trace_id is not None:
                if trace_id in trace_ids:
                    raise GroupInputError(GroupErrorCode.REPEATED_TRACE)
                trace_ids.add(trace_id)
                scope = ("trace", trace_id)
            else:
                self.issue(GroupIssue.MISSING_TRACE_ID)
                scope = ("member", str(member_index))
            for row_index, row in enumerate(rows):
                sid = _id(row.get("span_id"))
                identity = (scope, sid) if sid is not None else None
                if identity in self.by_identity:
                    _require_same(self.spans[self.by_identity[identity]].row, row)
                    self.duplicate_spans += 1
                    self.issue(GroupIssue.DUPLICATE_CAPTURE)
                    continue
                if sid is None:
                    self.issue(GroupIssue.MISSING_SPAN_ID)
                start = _span_start_seconds(row)
                end = _span_start_seconds({"start_timestamp": row.get("end_timestamp")})
                if end is None and row.get("end_timestamp") is None and start is not None:
                    duration = _nonnegative_finite_number(row.get("duration_ms"))
                    if duration is not None:
                        end = start + duration / 1000
                        if not math.isfinite(end):
                            end = None
                if start is None or end is None:
                    self.issue(GroupIssue.UNKNOWN_TIMESTAMP)
                if start is not None and end is not None and end < start:
                    end = None
                    self.issue(GroupIssue.UNKNOWN_TIMESTAMP)
                span = _Span(row, scope, sid, _id(row.get("parent_span_id")), (member_index, row_index), start, end)
                if identity is not None:
                    self.by_identity[identity] = len(self.spans)
                self.spans.append(span)

    @staticmethod
    def node_key(span: _Span, timestamp: float | None, phase: int, position: int) -> tuple[Any, ...]:
        return (
            timestamp is None,
            timestamp if timestamp is not None else 0.0,
            span.scope,
            span.sid or "",
            phase,
            position,
            span.coordinate,
        )

    def node(self, span: _Span, timestamp: float | None, phase: int, position: int) -> int:
        self.nodes.append(_Node(self.node_key(span, timestamp, phase, position), span.coordinate[0]))
        return len(self.nodes) - 1

    def edge(self, before: int, after: int) -> None:
        if before != after:
            self.nodes[before].edges.add(after)

    def records(self, span: _Span, phase: str) -> list[dict[str, Any]]:
        attrs = span.row["attributes"]
        for key in (f"gen_ai.{phase}.messages", f"gen_ai.{phase}_messages"):
            if key in attrs:
                records = _parse_messages(attrs[key])
                if not isinstance(attrs[key], list) and not records and not isinstance(_parse_json(attrs[key]), list):
                    self.issue(GroupIssue.UNSUPPORTED_EVIDENCE, span.coordinate[0])
                break
        else:
            raw_source = attrs.get(f"langfuse.observation.{phase}")
            source = _parse_json(raw_source) if isinstance(raw_source, (str, dict, list)) else raw_source
            if isinstance(source, list):
                records = source
            elif isinstance(source, dict) and "messages" in source:
                records = source["messages"]
                if not isinstance(records, list):
                    self.issue(GroupIssue.UNSUPPORTED_EVIDENCE, span.coordinate[0])
                    records = []
            else:
                records = _langfuse_messages(source)
                if not records and source not in (None, "", [], {}) and classify_span(span.row) != SpanType.TOOL_CALL:
                    # Scalar tool payloads are handled by execution(); other
                    # roleless payloads cannot establish an empty conversation.
                    self.issue(GroupIssue.UNSUPPORTED_EVIDENCE, span.coordinate[0])
        valid = [record for record in records if isinstance(record, dict) and isinstance(record.get("role"), str)]
        if len(valid) != len(records):
            self.issue(GroupIssue.UNSUPPORTED_EVIDENCE, span.coordinate[0])
        return valid

    def call_data(self, raw: dict[str, Any], span_index: int) -> tuple[str | None, dict[str, Any]]:
        function = raw.get("function")
        source = function if isinstance(function, dict) else raw
        arguments = _parse_json(_first(source, ("input", "arguments")))
        if arguments is not None and not isinstance(arguments, dict):
            self.issue(GroupIssue.UNSUPPORTED_EVIDENCE, self.spans[span_index].coordinate[0])
            arguments = None
        name = source.get("name", "")
        if not isinstance(name, str):
            self.issue(GroupIssue.UNSUPPORTED_EVIDENCE, self.spans[span_index].coordinate[0])
            name = ""
        return _id(_first(raw, ("id", "tool_call_id", "call_id"))), {
            "name": name,
            "input": arguments,
            "output": None,
        }

    def message(
        self, span_index: int, raw: dict[str, Any], phase: int, position: int, *, derived_result: bool = False
    ) -> int:
        span = self.spans[span_index]
        message_id = _id(_first(raw, ("message_id", "id")))
        identity = (span.scope, message_id) if message_id is not None else None
        if derived_result and span.identity is not None:
            # A single span result is an identified observation, not an
            # unidentified source message. Keep its namespace disjoint from
            # caller-supplied message IDs, even if one mimics the rendered ID.
            identity = (span.scope, span.sid, "tool_result")
            message_id = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
        if identity in self.message_ids:
            previous, node = self.message_ids[identity]
            _require_same(previous, raw)
            self.issue(GroupIssue.DUPLICATE_CAPTURE)
            self.message_owners[node].add(span_index)
            key = self.node_key(span, span.start if phase == 0 else span.end, phase, position)
            self.nodes[node].key = min(self.nodes[node].key, key)
            for result in self.results:
                if result.node == node and phase == 1:
                    result.is_output = True
            return node
        if message_id is None:
            self.issue(GroupIssue.AMBIGUOUS_MESSAGE_IDENTITY)
        node = self.node(span, span.start if phase == 0 else span.end, phase, position)
        if identity is not None:
            self.message_ids[identity] = (raw, node)
        calls = []
        raw_calls = raw.get("tool_calls")
        if isinstance(raw_calls, list):
            calls.extend(call for call in raw_calls if isinstance(call, dict))
            if len(calls) != len(raw_calls):
                self.issue(GroupIssue.UNSUPPORTED_EVIDENCE, span.coordinate[0])
        elif raw_calls is not None:
            self.issue(GroupIssue.UNSUPPORTED_EVIDENCE, span.coordinate[0])
        blocks = _message_blocks(raw)
        calls.extend(block for block in blocks if block.get("type") in ("tool_call", "tool_use"))
        typed_calls = []
        call_ids = []
        for call in calls:
            call_id, data = self.call_data(call, span_index)
            typed_calls.append(data)
            call_ids.append(call_id)
            self.intents.append(_Call(span_index, node, call_id, data))
        metadata = dict(raw["metadata"]) if isinstance(raw.get("metadata"), dict) else {}
        if message_id is not None:
            metadata["message_id"] = message_id
            if derived_result:
                metadata["source_span_id"] = span.sid
        if any(call_id is not None for call_id in call_ids):
            metadata["tool_call_ids"] = call_ids
        result_id = _id(_first(raw, ("tool_call_id", "tool_use_id", "call_id")))
        if result_id is not None:
            metadata["tool_call_id"] = result_id
        telemetry = {}
        for name in ("latency_ms", "token_count", "cost_usd"):
            # Recognized metadata telemetry must pass the same validation as
            # direct fields; metrics may otherwise read it as an unsafe fallback.
            metadata_value = self.telemetry(metadata.get(name), integral=name == "token_count")
            if name in metadata:
                if metadata_value is None:
                    metadata.pop(name)
                else:
                    metadata[name] = metadata_value
            source = raw.get(name, metadata_value)
            telemetry[name] = self.telemetry(source, integral=name == "token_count")
        retrieval_context = raw.get("retrieval_context")
        if retrieval_context is not None and (
            not isinstance(retrieval_context, list) or not all(isinstance(chunk, str) for chunk in retrieval_context)
        ):
            self.issue(GroupIssue.UNSUPPORTED_EVIDENCE, span.coordinate[0])
            retrieval_context = None
        expected = raw.get("expected")
        if expected is not None and not isinstance(expected, str):
            self.issue(GroupIssue.UNSUPPORTED_EVIDENCE, span.coordinate[0])
            expected = None
        data = {
            "role": raw["role"],
            "content": _message_text(raw),
            "tool_calls": typed_calls if calls or raw_calls == [] else None,
            "metadata": metadata or None,
            "retrieval_context": retrieval_context,
            "expected": expected,
            **telemetry,
        }
        self.nodes[node].message = data
        self.message_owners[node].add(span_index)
        self.raw_messages[node] = raw
        result_blocks = [block for block in blocks if block.get("type") in ("tool_call_response", "tool_result")]
        if raw["role"] == "tool" and not result_blocks:
            value = raw.get("content")
            if isinstance(value, list):
                value = _message_text(raw)
            self.results.append(_Result(span_index, node, result_id, value, phase == 1))
        for block in result_blocks:
            result_id = _id(_first(block, ("tool_call_id", "tool_use_id", "id", "call_id")))
            value = _first(block, ("result", "output", "content"))
            self.results.append(_Result(span_index, node, result_id, value, phase == 1))
        return node

    def telemetry(self, raw: Any, *, integral: bool = False) -> int | float | None:
        value = _parse_token_count(raw) if integral else _nonnegative_finite_number(raw)
        if raw is not None and value is None:
            self.issue(GroupIssue.INVALID_TELEMETRY)
        return value

    def execution(self, span_index: int) -> None:
        span = self.spans[span_index]
        row, attrs = span.row, span.row["attributes"]
        raw: dict[str, Any] = {
            "id": _first(attrs, ("gen_ai.tool.call.id", "gen_ai.tool.call_id", "tool.call.id", "tool_call_id"))
            or row.get("tool_call_id"),
            "name": _first(attrs, ("gen_ai.tool.name", "tool.name", "langfuse.observation.name"))
            or row.get("tool_name"),
            "arguments": _first(attrs, ("gen_ai.tool.call.arguments", "gen_ai.tool.input", "tool.input")),
        }
        if raw["name"] is None:
            name = row.get("span_name") or row.get("name") or ""
            raw["name"] = name.removeprefix("execute_tool ") if isinstance(name, str) else ""
        if raw["arguments"] is None:
            value = _parse_json(attrs.get("langfuse.observation.input"))
            if isinstance(value, dict):
                raw["arguments"] = _first(value, ("arguments", "input"))
                if raw["arguments"] is None and "messages" not in value and "role" not in value:
                    raw["arguments"] = value
        call_id, data = self.call_data(raw, span_index)
        local_intents = [
            intent
            for intent in self.intents
            if span_index in self.message_owners[intent.node] and (call_id is None or intent.call_id in (None, call_id))
        ]
        if len(local_intents) == 1:
            intent = local_intents[0]
            if _compatible_call_data(data, intent.data):
                call_id = call_id or intent.call_id
                data["input"] = data["input"] if data["input"] is not None else intent.data["input"]
                data["name"] = data["name"] or intent.data["name"]
            else:
                self.issue(GroupIssue.AMBIGUOUS_TOOL_LINK)
        output = _first(attrs, ("gen_ai.tool.call.result", "gen_ai.tool.output", "tool.output"))
        if output is None and "langfuse.observation.output" in attrs:
            output = _parse_json(attrs["langfuse.observation.output"])
            if isinstance(output, dict) and "content" in output and "role" not in output:
                output = output["content"]
            if _langfuse_messages(attrs["langfuse.observation.output"]):
                output = None
        local_results = [result for result in self.results if result.node in span.outputs]
        if len(local_results) == 1:
            result = local_results[0]
            output_owners = {
                owner
                for owner in self.message_owners[result.node]
                if result.node in self.spans[owner].outputs
                and classify_span(self.spans[owner].row) == SpanType.TOOL_CALL
            }
            if call_id is not None and result.call_id is not None and call_id != result.call_id:
                self.issue(GroupIssue.AMBIGUOUS_TOOL_LINK)
            elif output_owners == {span_index}:
                call_id = call_id or result.call_id
            # Message results are assigned only after global ownership/link
            # resolution. Never replace an explicit span result here.
        data["output"] = self.result_value(output, span_index)
        self.nodes[span.start_node].execution = data
        execution = _Call(span_index, span.start_node, call_id, data)
        self.executions.append(execution)
        if output is not None and not local_results and data["output"] is not None:
            content = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False, sort_keys=True)
            raw_result = {"role": "tool", "content": content}
            if call_id is not None:
                raw_result["tool_call_id"] = call_id
            result_node = self.message(span_index, raw_result, 1, len(span.outputs), derived_result=True)
            # The source payload is retained in ToolCall, not its rendered text.
            self.results[-1].value = output
            span.outputs.append(result_node)
            span.completions.append(result_node)
            self.edge(span.start_node, result_node)
            self.edge(result_node, span.end_node)

    def result_value(self, value: Any, span_index: int) -> str | dict[str, Any] | None:
        if value is None or isinstance(value, (str, dict)):
            return value
        self.issue(GroupIssue.UNSUPPORTED_EVIDENCE, self.spans[span_index].coordinate[0])
        return None

    def build_nodes(self) -> None:
        for index, span in enumerate(self.spans):
            span.start_node = self.node(span, span.start, 0, -1)
            span.end_node = self.node(span, span.end, 2, -1)
            span.inputs = [self.message(index, raw, 0, pos) for pos, raw in enumerate(self.records(span, "input"))]
            span.outputs = [self.message(index, raw, 1, pos) for pos, raw in enumerate(self.records(span, "output"))]
            intent_nodes = {intent.node for intent in self.intents if index in self.message_owners[intent.node]}
            span.completions = [node for node in span.outputs if node not in intent_nodes]
            chain = [*span.inputs, span.start_node, *span.outputs, span.end_node]
            for before, after in zip(chain, chain[1:], strict=False):
                self.edge(before, after)
        # Capture every message owner before inferring local execution identity.
        for index, span in enumerate(self.spans):
            if classify_span(span.row) == SpanType.TOOL_CALL:
                self.execution(index)
        for span_index, span in enumerate(self.spans):
            # A span-wide observation can be attributed to a single output
            # message only when that message belongs to exactly one span.
            if len(set(span.outputs)) == 1 and self.message_owners[span.outputs[0]] == {span_index}:
                message = self.nodes[span.outputs[0]].message
                if message is not None:
                    duration = span.row.get("duration_ms")
                    if duration is None and span.start is not None and span.end is not None:
                        duration = (span.end - span.start) * 1000
                    incoming, outgoing = _raw_token_counts(span.row)
                    tokens = [self.telemetry(value, integral=True) for value in (incoming, outgoing)]
                    known_tokens = [value for value in tokens if value is not None]
                    raw_cost = _first(span.row, ("cost_usd",))
                    if raw_cost is None:
                        raw_cost = span.row["attributes"].get("gen_ai.usage.cost")
                    observed = {
                        "latency_ms": self.telemetry(duration),
                        "token_count": sum(known_tokens) if known_tokens else None,
                        "cost_usd": self.telemetry(raw_cost),
                    }
                    for name, value in observed.items():
                        if message[name] is None:
                            message[name] = value
        for span in self.spans:
            if span.parent is None:
                continue
            parent_index = self.by_identity.get((span.scope, span.parent))
            if parent_index is None:
                self.issue(GroupIssue.MISSING_PARENT, span.coordinate[0])
                continue
            parent = self.spans[parent_index]
            if parent is span:
                self.issue(GroupIssue.CYCLIC_ORDER, span.coordinate[0])
            self.edge(parent.start_node, span.start_node)
            self.edge(span.end_node, parent.end_node)
            for completion in parent.completions:
                self.edge(span.end_node, completion)

    def link_tools(self) -> None:
        intents: dict[tuple[tuple[str, str], str], list[_Call]] = defaultdict(list)
        executions: dict[tuple[tuple[str, str], str], list[_Call]] = defaultdict(list)
        for items, target in ((self.intents, intents), (self.executions, executions)):
            for call in items:
                if call.call_id is not None:
                    target[(self.spans[call.span].scope, call.call_id)].append(call)
        for execution in self.executions:
            span = self.spans[execution.span]
            matches: list[_Call] = []
            if execution.call_id is not None:
                key = (span.scope, execution.call_id)
                if len(executions[key]) == 1:
                    matches = intents[key]
                else:
                    self.issue(GroupIssue.AMBIGUOUS_TOOL_LINK)
                    self.ambiguous_execution_spans.update(call.span for call in executions[key])
            elif span.parent is not None:
                parent_index = self.by_identity.get((span.scope, span.parent))
                siblings = [
                    call
                    for call in self.executions
                    if self.spans[call.span].scope == span.scope and self.spans[call.span].parent == span.parent
                ]
                if parent_index is not None and len(siblings) == 1:
                    matches = [
                        call
                        for call in self.intents
                        if parent_index in self.message_owners[call.node] and call.call_id is None
                    ]
                elif parent_index is not None and any(
                    parent_index in self.message_owners[call.node] for call in self.intents
                ):
                    self.issue(GroupIssue.AMBIGUOUS_TOOL_LINK)
            if len(matches) == 1:
                intent = matches[0]
                if not _compatible_call_data(execution.data, intent.data):
                    self.issue(GroupIssue.AMBIGUOUS_TOOL_LINK)
                    continue
                if intent.node != execution.node:
                    self.edge(intent.node, execution.node)
                intent.data["output"] = execution.data["output"]
                self.linked_intents[execution.node].append(intent)
            elif len(matches) > 1:
                self.issue(GroupIssue.AMBIGUOUS_TOOL_LINK)
        result_matches: dict[int, list[_Result]] = defaultdict(list)
        for result in self.results:
            span = self.spans[result.span]
            matches = executions.get((span.scope, result.call_id), []) if result.call_id is not None else []
            if not matches and result.is_output:
                local = [
                    execution
                    for execution in self.executions
                    if execution.span in self.message_owners[result.node]
                    and result.node in self.spans[execution.span].outputs
                ]
                if not local:
                    parents = {
                        self.by_identity.get((self.spans[owner].scope, self.spans[owner].parent))
                        for owner in self.message_owners[result.node]
                        if result.node in self.spans[owner].outputs and self.spans[owner].parent is not None
                    }
                    local = [execution for execution in self.executions if execution.span in parents]
                matches = [
                    execution
                    for execution in local
                    if result.call_id is None or execution.call_id in (None, result.call_id)
                ]
            if len(matches) == 1:
                execution = matches[0]
                self.edge(execution.node, result.node)
                result_matches[execution.node].append(result)
            else:
                self.issue(GroupIssue.AMBIGUOUS_TOOL_LINK if matches else GroupIssue.UNMATCHED_TOOL_RESULT)
        for execution in self.executions:
            results = result_matches[execution.node]
            if len(results) == 1:
                value = self.result_value(results[0].value, results[0].span)
                if execution.data["output"] is not None and value is not None:
                    _require_same(execution.data["output"], value)
                if value is not None:
                    execution.data["output"] = value
                for intent in self.linked_intents[execution.node]:
                    intent.data["output"] = execution.data["output"]
            elif len(results) > 1:
                self.issue(GroupIssue.AMBIGUOUS_TOOL_LINK)

    def cycle_root(self, remaining: set[int]) -> int:
        """Choose a node in a root strongly connected component, iteratively.

        A stalled topological sort can contain downstream acyclic nodes too.
        Kosaraju's two passes isolate components without recursion; only a root
        component may lose an incoming dependency without violating other
        components' causal edges.
        """
        predecessors: dict[int, set[int]] = {index: set() for index in remaining}
        for index in remaining:
            for after in self.nodes[index].edges:
                if after in remaining:
                    predecessors[after].add(index)
        visited: set[int] = set()
        finished: list[int] = []
        for root in remaining:
            stack = [(root, False)]
            while stack:
                index, exiting = stack.pop()
                if exiting:
                    finished.append(index)
                elif index not in visited:
                    visited.add(index)
                    stack.append((index, True))
                    stack.extend(
                        (after, False)
                        for after in self.nodes[index].edges
                        if after in remaining and after not in visited
                    )
        components: dict[int, int] = {}
        for root in reversed(finished):
            if root in components:
                continue
            components[root] = root
            pending = [root]
            while pending:
                index = pending.pop()
                for before in predecessors[index]:
                    if before not in components:
                        components[before] = root
                        pending.append(before)
        blocked = {
            components[after]
            for after in remaining
            for before in predecessors[after]
            if components[before] != components[after]
        }
        return min(
            (index for index in remaining if components[index] not in blocked),
            key=lambda index: self.nodes[index].key,
        )

    def ordered(self) -> list[int]:
        incoming = [0] * len(self.nodes)
        for node in self.nodes:
            for after in node.edges:
                incoming[after] += 1
        ready = [(node.key, index) for index, node in enumerate(self.nodes) if incoming[index] == 0]
        heapq.heapify(ready)
        remaining = set(range(len(self.nodes)))
        ordered = []
        while remaining:
            if not ready:
                # Break only a root cycle, retaining downstream dependencies.
                index = self.cycle_root(remaining)
                self.issue(GroupIssue.CYCLIC_ORDER, self.nodes[index].member_index)
                heapq.heappush(ready, (self.nodes[index].key, index))
            _, index = heapq.heappop(ready)
            if index not in remaining:
                continue
            remaining.remove(index)
            ordered.append(index)
            for after in self.nodes[index].edges:
                incoming[after] -= 1
                if incoming[after] == 0 and after in remaining:
                    heapq.heappush(ready, (self.nodes[after].key, after))
        return ordered

    def check_tool_order(self, ordered: list[int]) -> None:
        """Diagnose a presentation sequence that is not an established order.

        Consecutive executions suffice: a total order requires each adjacent
        pair to be ordered. Propagate the latest execution ancestor through
        the evidence graph in linear time instead of searching every pair.
        Independent overlapping intervals (including tied starts) do not
        establish an order; parent/intent/result dependencies still can.
        """
        if len(self.executions) < 2:
            return
        execution_spans = {execution.node: self.spans[execution.span] for execution in self.executions}
        latest_ancestor = [-1] * len(self.nodes)
        previous: _Span | None = None
        rank = -1
        for index in ordered:
            latest = latest_ancestor[index]
            current = execution_spans.get(index)
            if current is not None:
                if previous is not None and latest != rank:
                    separated = (
                        previous.start is not None
                        and previous.end is not None
                        and current.start is not None
                        and previous.start < current.start
                        and previous.end <= current.start
                    )
                    if not separated:
                        self.issue(GroupIssue.AMBIGUOUS_TOOL_ORDER)
                rank += 1
                latest = rank
                previous = current
            for after in self.nodes[index].edges:
                latest_ancestor[after] = max(latest_ancestor[after], latest)

    def ancestors(self, index: int) -> set[int] | None:
        ancestors = set()
        current = index
        while True:
            span = self.spans[current]
            if span.sid is None or current in ancestors:
                return None
            ancestors.add(current)
            if span.parent is None:
                ancestors.remove(index)
                return ancestors
            parent = self.by_identity.get((span.scope, span.parent))
            if parent is None:
                return None
            current = parent

    def usage(self) -> tuple[dict[str, int | float | None], dict[str, GroupUsageCoverage]]:
        totals: dict[str, int | float | None] = {}
        coverage = {}
        ancestry = [self.ancestors(index) for index in range(len(self.spans))]
        for dimension in _DIMENSIONS:
            counts: Counter[str] = Counter()
            span_reports = [_span_usage(span.row, dimension) for span in self.spans]
            # Reports retain *all* source owners: a message captured in two
            # histories is one event, but can overlap either history's ancestry.
            reports: list[tuple[set[int], Any, bool, bool]] = [
                ({index}, raw, True, False) for index, raw in enumerate(span_reports) if raw is not None
            ]
            message_coverage: set[int] = set()
            for node, message in self.raw_messages.items():
                raw = _message_usage(message, dimension)
                if raw is None:
                    continue
                owners = self.message_owners[node]
                message_coverage.update(owners)
                explicit_id = _id(_first(message, ("message_id", "id"))) is not None
                single_output = len(owners) == 1 and all(
                    node in self.spans[owner].outputs and len(set(self.spans[owner].outputs)) == 1 for owner in owners
                )
                # Raw span reports take precedence over message reports, even
                # when invalid. Never recover an invalid column through an
                # attribute or a replayed message from a different capture.
                blocked = any(span_reports[owner] is not None for owner in owners)
                reports.append((owners, raw, explicit_id or single_output, blocked))
            counts["missing"] = sum(
                raw is None and index not in message_coverage for index, raw in enumerate(span_reports)
            )
            candidates: list[tuple[set[int], set[int], int | float]] = []
            for owners, raw, identified, blocked in reports:
                value = self.telemetry(raw, integral=dimension != "cost_usd")
                if value is None:
                    counts["invalid"] += 1
                    continue
                counts["observed"] += 1
                if blocked:
                    counts["overlapping"] += 1
                elif not identified or any(
                    ancestry[owner] is None or owner in self.ambiguous_execution_spans for owner in owners
                ):
                    counts["ambiguous"] += 1
                    self.issue(GroupIssue.AMBIGUOUS_USAGE)
                else:
                    ancestors = set().union(*(ancestry[owner] for owner in owners))
                    ancestors.difference_update(owners)
                    candidates.append((owners, ancestors, value))
            selected = []
            for index, (owners, _, value) in enumerate(candidates):
                if any(owners & ancestors for other, (_, ancestors, _) in enumerate(candidates) if other != index):
                    counts["overlapping"] += 1
                else:
                    selected.append(value)
                    counts["selected"] += 1
            total = sum(selected) if selected else None
            if dimension == "cost_usd" and selected:
                try:
                    total = math.fsum(selected)
                except OverflowError:
                    # Even finite reports can overflow when aggregated.
                    self.issue(GroupIssue.INVALID_TELEMETRY)
                    total = None
            totals[dimension] = total
            coverage[dimension] = GroupUsageCoverage(**counts)
        directions = [totals[name] for name in ("input_tokens", "output_tokens") if totals[name] is not None]
        totals["token_count"] = sum(directions) if directions else None
        return totals, coverage


def assemble(members: Sequence[GroupTrace]) -> _Evidence:
    builder = _Builder()
    builder.capture(members)
    builder.build_nodes()
    builder.link_tools()
    ordered = builder.ordered()
    builder.check_tool_order(ordered)
    usage, coverage = builder.usage()
    messages = [builder.nodes[node].message for node in ordered if builder.nodes[node].message is not None]
    executions = [builder.nodes[node].execution for node in ordered if builder.nodes[node].execution is not None]
    completeness = [
        False if member.capture_complete is True and index in builder.contradicted_members else member.capture_complete
        for index, member in enumerate(members)
    ]
    known_empty = all(complete is True for complete in completeness)
    diagnostics = GroupDiagnostics(
        member_count=len(members),
        span_count=len(builder.spans),
        duplicate_span_count=builder.duplicate_spans,
        message_count=len(messages),
        tool_execution_count=len(executions),
        complete_member_count=sum(complete is True for complete in completeness),
        incomplete_member_count=sum(complete is False for complete in completeness),
        unknown_member_count=sum(complete is None for complete in completeness),
        issues=tuple(GroupIssueCount(issue, count) for issue, count in sorted(builder.issues.items())),
        **coverage,
    )
    return _Evidence(
        messages if messages or known_empty else None,
        executions if executions or known_empty else None,
        usage,
        diagnostics,
    )
