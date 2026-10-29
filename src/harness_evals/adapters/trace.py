"""Trace adapter — converts assembled trace spans into EvalCase for scoring.

One adaptation shared by API-side and batch scoring paths: same EvalCase,
same parity. Follows the OTel GenAI agent-spans semantic conventions, with
dialect coalescing for the attribute shapes seen in deployed agent traffic.

- Span normalization: sources may carry `attributes` as a JSON string or a
  dict; start_timestamp may be a datetime or ISO string. Both are normalized
  before extraction.
- Dialect coalescing: classify_span falls back from semconv
  `gen_ai.operation.name` to the Langfuse-instrumentation markers —
  `langfuse.observation.type` (agent/generation/tool) — and a narrow
  `span_type` column, before the root fallback.
- Message coalescing: canonical `gen_ai.*.messages` takes precedence;
  documented Langfuse agent, generation, and tool observations supply input,
  output, and tool payloads when those canonical fields are absent.
- Token counts coalesce narrow input_tokens/output_tokens columns with the
  gen_ai.usage.* attributes (columns win).

Missing-content rule: empty `eval_case.output` is not a blanket skip signal.
Selective adaptation leaves unrequested output empty, including valid
telemetry-only cases. Callers must check the inputs their metrics need;
requested but missing output is reported in the returned warnings.
"""

from __future__ import annotations

import ast
import enum
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import MetricInput, MetricInputs
from harness_evals.core.types import Message, ToolCall

if TYPE_CHECKING:
    from harness_evals.core.metric import BaseMetric

logger = logging.getLogger("harness_evals.adapters.trace")

# Well-known gen_ai.operation.name values from the spec
_OP_INVOKE_AGENT = "invoke_agent"
_OP_CHAT = "chat"
_OP_GENERATE_CONTENT = "generate_content"
_OP_EXECUTE_TOOL = "execute_tool"
_OP_INVOKE_WORKFLOW = "invoke_workflow"


class SpanType(enum.Enum):
    LLM_TURN = "llm_turn"
    TOOL_CALL = "tool_call"
    AGENT_ROOT = "agent_root"
    OTHER = "other"


def normalize_span(span: dict[str, Any]) -> dict[str, Any]:
    """Normalize one span row: attributes JSON string → dict."""
    span = dict(span)
    attrs = span.get("attributes")
    if isinstance(attrs, str):
        try:
            span["attributes"] = json.loads(attrs or "{}")
        except (json.JSONDecodeError, TypeError):
            span["attributes"] = {}
    elif attrs is None:
        span["attributes"] = {}
    return span


def _attrs(span: dict[str, Any]) -> dict[str, Any]:
    attrs = span.get("attributes")
    if isinstance(attrs, str):
        try:
            return json.loads(attrs or "{}")
        except (json.JSONDecodeError, TypeError):
            return {}
    return attrs or {}


def classify_span(span: dict[str, Any]) -> SpanType:
    """Classify a span: semconv → langfuse marker → span_type column → root.

    Tolerates attributes as a JSON string (un-normalized rows) so it is safe
    to call standalone (e.g. duration fallback scans).
    """
    attrs = _attrs(span)
    operation = attrs.get("gen_ai.operation.name", "")

    if operation in (_OP_INVOKE_AGENT, _OP_INVOKE_WORKFLOW):
        return SpanType.AGENT_ROOT
    if operation in (_OP_CHAT, _OP_GENERATE_CONTENT):
        return SpanType.LLM_TURN
    if operation == _OP_EXECUTE_TOOL:
        return SpanType.TOOL_CALL

    # Langfuse's explicit observation type is more reliable than a span name.
    # Generic `span` records deliberately remain unclassified.
    langfuse_type = attrs.get("langfuse.observation.type", "")
    if langfuse_type == "agent":
        return SpanType.AGENT_ROOT
    if langfuse_type == "generation":
        return SpanType.LLM_TURN
    if langfuse_type == "tool":
        return SpanType.TOOL_CALL

    span_name = str(span.get("span_name") or span.get("name") or "")
    if span_name.startswith("execute_tool"):
        return SpanType.TOOL_CALL
    tool_name = span.get("tool_name")
    if tool_name and str(tool_name) not in ("", "NA"):
        return SpanType.TOOL_CALL
    if span_name == "chat":
        return SpanType.LLM_TURN

    span_type = (span.get("span_type") or "").lower()
    if span_type in ("agent", "workflow"):
        return SpanType.AGENT_ROOT
    if span_type == "tool":
        return SpanType.TOOL_CALL
    if span_type == "llm":
        return SpanType.LLM_TURN

    # Fallback: root span (no parent) is treated as agent root
    parent = span.get("parent_span_id")
    if not parent:
        return SpanType.AGENT_ROOT

    return SpanType.OTHER


def _extract_text_from_parts(parts: list[dict[str, Any]]) -> str | None:
    """Extract concatenated text content from a message's parts array."""
    if not parts:
        return None
    texts = []
    for part in parts:
        ptype = part.get("type", "")
        if ptype == "text":
            value = part.get("text")
            if value is None:
                value = part.get("content")
            if isinstance(value, str):
                texts.append(value)
        elif not ptype and isinstance(part.get("text"), str):
            texts.append(part["text"])
        elif ptype == "" and isinstance(part.get("content"), str):
            texts.append(part["content"])
    return "\n".join(texts) if texts else None


def _parse_json(value: Any) -> Any:
    """Parse an already-structured value or a JSON string without raising."""
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return None


def _message_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Return canonical parts or Langfuse content blocks from one message."""
    blocks = message.get("parts")
    if not isinstance(blocks, list):
        blocks = message.get("content")
    return [block for block in blocks if isinstance(block, dict)] if isinstance(blocks, list) else []


def _message_text(message: dict[str, Any]) -> str | None:
    """Extract text from canonical parts or Langfuse content."""
    blocks = _message_blocks(message)
    if blocks:
        return _extract_text_from_parts(blocks)
    content = message.get("content")
    return content if isinstance(content, str) else None


def _extract_tool_calls_from_parts(parts: list[dict[str, Any]]) -> list[ToolCall]:
    """Extract tool calls from a message's parts array."""
    tool_calls = []
    for part in parts:
        if part.get("type") in {"tool_call", "tool_use"}:
            tool_calls.append(
                ToolCall(
                    name=part.get("name", ""),
                    input=part.get("arguments", part.get("input")),
                    output=None,
                )
            )
    return tool_calls


def _extract_tool_results_from_parts(parts: list[dict[str, Any]]) -> dict[str, str]:
    """Extract tool results keyed by call ID from a message's parts array."""
    results = {}
    for part in parts:
        if part.get("type") == "tool_call_response":
            call_id = part.get("id", "")
            results[call_id] = part.get("result", "")
    return results


def _parse_messages(messages_attr: Any) -> list[dict[str, Any]]:
    """Parse gen_ai.input.messages or gen_ai.output.messages attribute.

    Handles both valid JSON (double quotes) and Python repr format
    (single quotes) that some instrumentations emit via str() instead
    of json.dumps().
    """
    if isinstance(messages_attr, list):
        return messages_attr
    if isinstance(messages_attr, str):
        try:
            parsed = json.loads(messages_attr)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            parsed = ast.literal_eval(messages_attr)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return []


def _langfuse_messages(value: Any) -> list[dict[str, Any]]:
    """Normalize documented Langfuse input/output values to message records."""
    parsed = _parse_json(value)
    if isinstance(parsed, list):
        return [message for message in parsed if isinstance(message, dict)]
    if not isinstance(parsed, dict):
        return []
    messages = parsed.get("messages")
    if isinstance(messages, list):
        return [message for message in messages if isinstance(message, dict)]
    return [parsed] if isinstance(parsed.get("role"), str) else []


def _messages_from_attrs(attrs: dict[str, Any], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """Read canonical messages first, then a Langfuse message value."""
    for key in keys:
        messages = _parse_messages(attrs.get(key))
        if messages:
            return messages
    return []


def _append_user_messages(messages: list[Message], source_messages: list[dict[str, Any]]) -> None:
    """Append unseen user turns while preserving the trace's chronological order."""
    known = {message.content for message in messages if message.role == "user" and message.content}
    for source in source_messages:
        if source.get("role") != "user":
            continue
        content = _message_text(source)
        if content and content not in known:
            messages.append(Message(role="user", content=content))
            known.add(content)


def _has_matching_message_tool_call(messages: list[Message], target: ToolCall) -> bool:
    """Return whether an LLM message already records this executed tool call."""
    for message in messages:
        for tool_call in message.tool_calls or []:
            if tool_call.name == target.name and tool_call.input == target.input:
                return True
    return False


def _extract_output_from_span(span: dict[str, Any]) -> tuple[str | None, list[ToolCall]]:
    """Extract assistant output text and tool calls from an LLM span's output messages."""
    attrs = _attrs(span)
    output_messages = _messages_from_attrs(attrs, ("gen_ai.output.messages", "gen_ai.output_messages"))
    if not output_messages:
        langfuse_output = _parse_json(attrs.get("langfuse.observation.output"))
        if isinstance(langfuse_output, str) and langfuse_output:
            return langfuse_output, []
        output_messages = _langfuse_messages(langfuse_output)

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for msg in output_messages:
        role = msg.get("role", "")
        if role == "assistant" or not role:
            text = _message_text(msg)
            if text:
                text_parts.append(text)
            tcs = _extract_tool_calls_from_parts(_message_blocks(msg))
            tool_calls.extend(tcs)

    content = "\n".join(text_parts) if text_parts else None
    return content, tool_calls


def _extract_input_from_span(span: dict[str, Any]) -> str | None:
    """Extract the user's input from an LLM span's input messages."""
    attrs = _attrs(span)
    input_messages = _messages_from_attrs(attrs, ("gen_ai.input.messages", "gen_ai.input_messages"))
    if not input_messages:
        input_messages = _langfuse_messages(attrs.get("langfuse.observation.input"))

    for msg in reversed(input_messages):
        role = msg.get("role", "")
        if role == "user":
            text = _message_text(msg)
            if text:
                return text
    for key in ("langfuse.trace.input", "langfuse.observation.input"):
        value = _parse_json(attrs.get(key))
        if isinstance(value, dict):
            text = value.get("user_message") or value.get("prompt") or value.get("query")
            if isinstance(text, str) and text:
                return text
        if isinstance(value, str) and value:
            return value
    return None


def _extract_tool_from_span(span: dict[str, Any]) -> ToolCall:
    """Extract tool call info from an execute_tool span."""
    attrs = _attrs(span)
    span_name = span.get("span_name") or span.get("name") or ""

    # Per spec: span name is "execute_tool {gen_ai.tool.name}"
    name = (
        attrs.get("gen_ai.tool.name")
        or attrs.get("tool.name")
        or attrs.get("langfuse.observation.name")
        or span.get("tool_name")
        or ""
    )
    name = name or (span_name[len("execute_tool ") :] if span_name.startswith("execute_tool ") else span_name)

    input_messages = _messages_from_attrs(attrs, ("gen_ai.input.messages", "gen_ai.input_messages"))
    tool_input: Any = None
    for msg in input_messages:
        parts = msg.get("parts", [])
        for part in parts:
            if part.get("type") == "tool_call":
                if not name:
                    name = part.get("name", "")
                tool_input = part.get("arguments")
                break

    output_messages = _messages_from_attrs(attrs, ("gen_ai.output.messages", "gen_ai.output_messages"))
    tool_output: Any = None
    for msg in output_messages:
        parts = msg.get("parts", [])
        results = _extract_tool_results_from_parts(parts)
        if results:
            tool_output = next(iter(results.values()))
            break

    if tool_input is None:
        langfuse_input = _parse_json(attrs.get("langfuse.observation.input"))
        if isinstance(langfuse_input, dict):
            name = name or str(langfuse_input.get("name") or "")
            tool_input = langfuse_input.get("arguments", langfuse_input.get("input", langfuse_input))
    if tool_output is None:
        langfuse_output = _parse_json(attrs.get("langfuse.observation.output"))
        if isinstance(langfuse_output, dict) and "content" in langfuse_output:
            tool_output = langfuse_output["content"]
        elif isinstance(langfuse_output, str):
            tool_output = langfuse_output

    return ToolCall(name=name, input=tool_input, output=tool_output)


def _get_token_counts(span: dict[str, Any]) -> tuple[int, int]:
    """Input/output tokens: narrow columns win, gen_ai.usage.* attrs fallback."""
    input_tokens, output_tokens = _raw_token_counts(span)
    input_tokens = input_tokens or 0
    output_tokens = output_tokens or 0
    if isinstance(input_tokens, str):
        input_tokens = int(input_tokens) if input_tokens.isdigit() else 0
    if isinstance(output_tokens, str):
        output_tokens = int(output_tokens) if output_tokens.isdigit() else 0
    return int(input_tokens), int(output_tokens)


def _raw_token_counts(span: dict[str, Any]) -> tuple[Any, Any]:
    """Return source token values before coercion, preserving zero."""
    attrs = _attrs(span)
    input_tokens = span.get("input_tokens")
    output_tokens = span.get("output_tokens")
    if input_tokens is None:
        input_tokens = attrs.get("gen_ai.usage.input_tokens")
    if output_tokens is None:
        output_tokens = attrs.get("gen_ai.usage.output_tokens")
    return input_tokens, output_tokens


def _is_valid_token_count(value: Any) -> bool:
    """Return whether a reported token value can safely represent usage."""
    if isinstance(value, str):
        return value.isdigit()
    if isinstance(value, bool):
        return False
    try:
        return int(value) >= 0
    except (TypeError, ValueError):
        return False


def _has_token_usage(span: dict[str, Any]) -> tuple[bool, bool]:
    """Return whether each token dimension was validly reported, including zero."""
    input_tokens, output_tokens = _raw_token_counts(span)
    return _is_valid_token_count(input_tokens), _is_valid_token_count(output_tokens)


def _observed_token_count(span: dict[str, Any]) -> int | None:
    """Return a span total using only validly reported token dimensions."""
    return _sum_token_usage([span]).total


def _trace_cost_usd(spans: list[dict[str, Any]]) -> float | None:
    """Return observed span cost, preferring child spans over rolled-up roots."""
    child_cost = 0.0
    root_cost = 0.0
    saw_child_cost = False
    saw_root_cost = False
    for span in spans:
        raw_cost = _attrs(span).get("gen_ai.usage.cost")
        try:
            cost = float(raw_cost)
        except (TypeError, ValueError):
            continue
        if cost < 0:
            continue
        if classify_span(span) == SpanType.AGENT_ROOT:
            root_cost += cost
            saw_root_cost = True
        else:
            child_cost += cost
            saw_child_cost = True
    if saw_child_cost:
        return child_cost
    if saw_root_cost:
        return root_cost
    return None


def _sort_key(span: dict[str, Any]) -> float:
    ts = span.get("start_timestamp")
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.timestamp()
    if isinstance(ts, str) and ts:
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    if isinstance(ts, (int, float)):
        return float(ts)
    return 0.0


def _decode_telemetry_value(value: Any) -> Any:
    """Unwrap JSON and common OTLP AnyValue-style wrapper shapes."""
    if isinstance(value, str):
        try:
            return _decode_telemetry_value(json.loads(value))
        except (json.JSONDecodeError, TypeError):
            try:
                return _decode_telemetry_value(ast.literal_eval(value))
            except (ValueError, SyntaxError):
                return value
    if isinstance(value, list):
        return [_decode_telemetry_value(item) for item in value]
    if not isinstance(value, dict):
        return value

    wrapper_type = str(value.get("_type") or value.get("type") or "").lower()
    if wrapper_type.endswith("arrayvalue"):
        return _decode_telemetry_value(value.get("values", []))
    if wrapper_type.endswith("kvlistvalue"):
        return _decode_key_value_list(value.get("values", []))
    if wrapper_type.endswith("stringvalue"):
        return value.get("value", "")
    if wrapper_type.endswith(("boolvalue", "intvalue", "doublevalue")):
        return value.get("value")
    for key in (
        "stringValue",
        "string_value",
        "boolValue",
        "bool_value",
        "intValue",
        "int_value",
        "doubleValue",
        "double_value",
    ):
        if key in value:
            return _decode_telemetry_value(value[key])
    for key in ("arrayValue", "array_value"):
        if key in value:
            array_value = value[key]
            if isinstance(array_value, dict):
                return _decode_telemetry_value(array_value.get("values", []))
            return _decode_telemetry_value(array_value)
    for key in ("kvlistValue", "kvlist_value"):
        if key in value:
            key_value_list = value[key]
            if isinstance(key_value_list, dict):
                return _decode_key_value_list(key_value_list.get("values", []))
            return _decode_key_value_list(key_value_list)
    return {key: _decode_telemetry_value(item) for key, item in value.items()}


def _decode_key_value_list(value: Any) -> dict[str, Any]:
    """Decode an OTLP KeyValueList-style map without assuming a schema."""
    if not isinstance(value, list):
        return {}
    decoded: dict[str, Any] = {}
    for entry in value:
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        if isinstance(key, str) and "value" in entry:
            decoded[key] = _decode_telemetry_value(entry["value"])
    return decoded


def _tool_call_from_aggregate(value: Any) -> ToolCall | None:
    """Convert a generic aggregate entry into a ToolCall without guessing data."""
    if isinstance(value, str) and value:
        return ToolCall(name=value)
    if not isinstance(value, dict):
        return None

    function = value.get("function")
    function_data = function if isinstance(function, dict) else {}
    name = value.get("name") or value.get("tool_name") or value.get("toolName") or function_data.get("name") or ""
    tool_input = (
        value.get("arguments")
        if "arguments" in value
        else value.get("input", value.get("args", function_data.get("arguments")))
    )
    tool_output = value.get("output", value.get("result"))
    if not isinstance(name, str) or not name:
        return None
    return ToolCall(name=name, input=tool_input, output=tool_output)


def _same_tool_call(left: ToolCall, right: ToolCall) -> bool:
    """Treat a name-only aggregate entry as a duplicate of its concrete span."""
    if not left.name or left.name != right.name:
        return False
    return left.input == right.input or left.input is None or right.input is None


def _aggregate_tool_calls(
    spans: list[dict[str, Any]],
    attribute_keys: tuple[str, ...],
    warnings: list[str],
) -> list[ToolCall]:
    """Read caller-configured root/agent aggregate tool-call attributes."""
    if not attribute_keys:
        return []

    tool_calls: list[ToolCall] = []
    for span in spans:
        attrs = _attrs(span)
        for key in attribute_keys:
            if key not in attrs:
                continue
            decoded = _decode_telemetry_value(attrs[key])
            entries = decoded if isinstance(decoded, list) else [decoded]
            for entry in entries:
                tool_call = _tool_call_from_aggregate(entry)
                if tool_call is None:
                    warnings.append(f"Configured aggregate tool-call attribute {key!r} contained an unreadable entry.")
                    continue
                tool_calls.append(tool_call)
    return tool_calls


def spans_to_eval_case(
    spans: list[dict[str, Any]],
    *,
    aggregate_tool_call_attribute_keys: tuple[str, ...] = (),
    metrics: Iterable[BaseMetric] | None = None,
) -> tuple[EvalCase, list[str]]:
    """Convert a list of span dicts into an EvalCase.

    Returns (EvalCase, warnings); warnings report requested content that was
    not observed. Empty ``eval_case.output`` can mean output was not requested
    or was not observed; it does not make telemetry-only metrics unscorable.
    Callers must check sufficiency for each metric rather than skip all metrics.

    ``aggregate_tool_call_attribute_keys`` is opt-in so callers can map
    source-specific aggregate attributes without embedding their convention in
    the shared adapter.

    When ``metrics`` is supplied, only the fields in those metrics'
    ``eval_inputs`` are materialized from the trace. Internal dependencies,
    such as message-level tool intents used to reconcile aggregate calls, are
    still read without materializing unrequested fields. Omitting ``metrics``
    materializes every field; an empty iterable requests none. Normalization
    and sorting still process every supplied span.
    """
    if not spans:
        return EvalCase(input="", output=""), ["No spans found in trace"]

    projection = _TraceProjection.for_metrics(metrics)
    sorted_spans = sorted((normalize_span(span) for span in spans), key=_sort_key)
    warnings: list[str] = []

    content = _TraceContent(projection)
    if projection.reads_span_content:
        for span in sorted_spans:
            content.add_span(span)
        content.add_aggregate_tool_calls(sorted_spans, aggregate_tool_call_attribute_keys, warnings)

    if projection.output and not content.output:
        warnings.append(
            "No LLM output content found in trace spans. "
            "Metrics requiring output text may produce low-confidence scores."
        )

    tokens = _sum_token_usage(sorted_spans) if projection.reads_token_usage else _TokenUsage()

    eval_case = EvalCase(
        input=(content.user_input or "") if projection.input else "",
        output=content.output or "",
        expected=None,
        messages=content.conversation() or None,
        tool_calls=content.tool_calls or None,
        latency_ms=_compute_total_duration(sorted_spans) if projection.latency_ms else None,
        token_count=tokens.total if projection.token_count else None,
        input_tokens=tokens.input_tokens if projection.input_tokens else None,
        output_tokens=tokens.output_tokens if projection.output_tokens else None,
        cost_usd=_trace_cost_usd(sorted_spans) if projection.cost_usd else None,
        metadata=_trace_metadata(spans[0].get("trace_id"), content.root_span) if projection.metadata else None,
    )

    return eval_case, warnings


@dataclass(frozen=True)
class _TraceProjection:
    """Requested EvalCase fields and the extraction dependencies they imply.

    Reading evidence is separate from exposing a field: tool-call projection
    needs message-level intents but need not build a public conversation.
    """

    input: bool
    output: bool
    messages: bool
    tool_calls: bool
    metadata: bool
    latency_ms: bool
    cost_usd: bool
    token_count: bool
    input_tokens: bool
    output_tokens: bool

    @classmethod
    def for_metrics(cls, metrics: Iterable[BaseMetric] | None) -> _TraceProjection:
        fields = (
            frozenset(MetricInput)
            if metrics is None
            else frozenset().union(*(_declared_fields(metric) for metric in metrics))
        )
        return cls(
            input=MetricInput.INPUT in fields,
            output=MetricInput.OUTPUT in fields,
            messages=MetricInput.MESSAGES in fields,
            tool_calls=MetricInput.TOOL_CALLS in fields,
            metadata=MetricInput.METADATA in fields,
            latency_ms=MetricInput.LATENCY_MS in fields,
            cost_usd=MetricInput.COST_USD in fields,
            token_count=MetricInput.TOKEN_COUNT in fields,
            input_tokens=MetricInput.INPUT_TOKENS in fields,
            output_tokens=MetricInput.OUTPUT_TOKENS in fields,
        )

    @property
    def reads_user_input(self) -> bool:
        return self.input or self.messages

    @property
    def reads_assistant_output(self) -> bool:
        return self.output or self.reads_tool_activity

    @property
    def reads_tool_activity(self) -> bool:
        return self.tool_calls or self.messages

    @property
    def reads_span_content(self) -> bool:
        """Whether spans must be classified and their payloads parsed."""
        return self.reads_user_input or self.reads_assistant_output or self.reads_tool_activity or self.metadata

    @property
    def reads_token_usage(self) -> bool:
        return self.token_count or self.input_tokens or self.output_tokens


def _declared_fields(metric: BaseMetric) -> frozenset[MetricInput]:
    """Return a metric's declared fields, or every field when the declaration is invalid."""
    inputs = getattr(metric, "eval_inputs", None)
    return inputs.fields if isinstance(inputs, MetricInputs) else frozenset(MetricInput)


class _TraceContent:
    """Accumulates the span-derived EvalCase fields a projection requests."""

    def __init__(self, projection: _TraceProjection) -> None:
        self._projection = projection
        self.user_input: str | None = None
        self.output: str | None = None
        self.messages: list[Message] = []
        self.tool_calls: list[ToolCall] = []
        # Evidence must not depend on which public fields are materialized.
        self._known_tool_calls: list[ToolCall] = []
        self.root_span: dict[str, Any] | None = None

    def add_span(self, span: dict[str, Any]) -> None:
        span_type = classify_span(span)
        if span_type == SpanType.AGENT_ROOT:
            self.root_span = span
            self._add_user_input(span)
        elif span_type == SpanType.LLM_TURN:
            self._add_user_input(span)
            self._add_llm_output(span)
        elif span_type == SpanType.TOOL_CALL:
            self._add_tool_call(span)

    def add_aggregate_tool_calls(
        self,
        spans: list[dict[str, Any]],
        attribute_keys: tuple[str, ...],
        warnings: list[str],
    ) -> None:
        """Add aggregate calls not already recorded by concrete spans or messages.

        Concrete spans and message-level calls are authoritative when they
        overlap, but aggregate telemetry may still contain additional calls.
        """
        if not self._projection.reads_tool_activity:
            return
        additional = [
            candidate
            for candidate in _aggregate_tool_calls(spans, attribute_keys, warnings)
            if not any(_same_tool_call(existing, candidate) for existing in self._known_tool_calls)
        ]
        if not additional:
            return
        if self._projection.tool_calls:
            self.tool_calls.extend(additional)
        if self._projection.messages:
            self.messages.append(Message(role="assistant", content=None, tool_calls=additional))

    def conversation(self) -> list[Message]:
        """Return the messages, led by the user input when no user turn was recorded."""
        if (
            self._projection.messages
            and self.user_input
            and not any(message.role == "user" for message in self.messages)
        ):
            return [Message(role="user", content=self.user_input), *self.messages]
        return self.messages

    def _add_user_input(self, span: dict[str, Any]) -> None:
        if self._projection.reads_user_input and not self.user_input:
            self.user_input = _extract_input_from_span(span)
        if self._projection.messages:
            attrs = _attrs(span)
            _append_user_messages(
                self.messages,
                _messages_from_attrs(attrs, ("gen_ai.input.messages", "gen_ai.input_messages"))
                or _langfuse_messages(attrs.get("langfuse.observation.input")),
            )

    def _add_llm_output(self, span: dict[str, Any]) -> None:
        if not self._projection.reads_assistant_output:
            return
        # Intents inform aggregate deduplication even without public messages;
        # execute_tool spans remain the authoritative record (with results).
        content, turn_tool_calls = _extract_output_from_span(span)
        if self._projection.reads_tool_activity:
            self._known_tool_calls.extend(turn_tool_calls)
        if self._projection.output and content:
            self.output = content
        if self._projection.messages:
            self.messages.append(
                Message(
                    role="assistant",
                    content=content if content or turn_tool_calls else "",
                    tool_calls=turn_tool_calls or None,
                    latency_ms=_span_duration_ms(span),
                    token_count=_observed_token_count(span),
                )
            )

    def _add_tool_call(self, span: dict[str, Any]) -> None:
        if not self._projection.reads_tool_activity:
            return
        tool_call = _extract_tool_from_span(span)
        self._known_tool_calls.append(tool_call)
        if self._projection.tool_calls:
            self.tool_calls.append(tool_call)
        if self._projection.messages:
            if not _has_matching_message_tool_call(self.messages, tool_call):
                self.messages.append(Message(role="assistant", content=None, tool_calls=[tool_call]))
            if tool_call.output:
                self.messages.append(Message(role="tool", content=str(tool_call.output)))


@dataclass(frozen=True)
class _TokenUsage:
    """Token totals; ``None`` means the dimension was never validly reported."""

    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def total(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)


def _sum_token_usage(spans: list[dict[str, Any]]) -> _TokenUsage:
    """Sum validly reported token usage across spans, keeping unreported dimensions ``None``."""
    input_total: int | None = None
    output_total: int | None = None
    for span in spans:
        input_tokens, output_tokens = _get_token_counts(span)
        has_input, has_output = _has_token_usage(span)
        if has_input:
            input_total = (input_total or 0) + input_tokens
        if has_output:
            output_total = (output_total or 0) + output_tokens
    return _TokenUsage(input_tokens=input_total, output_tokens=output_total)


def _trace_metadata(trace_id: Any, root_span: dict[str, Any] | None) -> dict[str, Any]:
    """Trace correlation plus the service, model, and agent labels on the root span."""
    metadata: dict[str, Any] = {"trace_id": trace_id, "source": "online_eval"}
    if not root_span:
        return metadata
    root_attrs = root_span.get("attributes") or {}
    metadata["service_name"] = root_span.get("service_name")
    model = (
        root_attrs.get("gen_ai.request.model")
        or root_attrs.get("gen_ai.response.model")
        or root_attrs.get("langfuse.observation.model.name")
    )
    if model:
        metadata["model"] = model
    agent_name = (
        root_attrs.get("gen_ai.agent.name")
        or root_attrs.get("agent.name")
        or root_attrs.get("langfuse.observation.name")
    )
    if agent_name:
        metadata["agent_name"] = agent_name
    return metadata


def extract_span_subtree(spans: list[dict[str, Any]], target_span_id: str) -> list[dict[str, Any]]:
    """Extract the subtree rooted at target_span_id (inclusive), preserving trace order."""
    span_by_id: dict[str, dict[str, Any]] = {}
    for span in spans:
        sid = span.get("span_id")
        if sid:
            span_by_id[sid] = span

    if target_span_id not in span_by_id:
        return []

    children: dict[str, list[str]] = {}
    for span in spans:
        parent = span.get("parent_span_id")
        if parent:
            children.setdefault(parent, []).append(span.get("span_id", ""))

    subtree_ids: set[str] = set()
    queue = [target_span_id]
    while queue:
        current = queue.pop()
        if current in subtree_ids:
            continue
        subtree_ids.add(current)
        queue.extend(children.get(current, []))

    return [span for span in spans if span.get("span_id") in subtree_ids]


def spans_to_eval_case_for_span(
    spans: list[dict[str, Any]],
    target_span_id: str,
    *,
    aggregate_tool_call_attribute_keys: tuple[str, ...] = (),
    metrics: Iterable[BaseMetric] | None = None,
) -> tuple[EvalCase, list[str]]:
    """Convert the subtree rooted at target_span_id into an EvalCase (span scope)."""
    subtree = extract_span_subtree(spans, target_span_id)
    if not subtree:
        return EvalCase(input="", output=""), [f"Span {target_span_id} not found in trace"]

    eval_case, warnings = spans_to_eval_case(
        subtree,
        aggregate_tool_call_attribute_keys=aggregate_tool_call_attribute_keys,
        metrics=metrics,
    )

    # The span association is stamped even when the projection omits metadata.
    if eval_case.metadata:
        eval_case.metadata["span_id"] = target_span_id
    else:
        eval_case.metadata = {"span_id": target_span_id, "source": "online_eval"}

    return eval_case, warnings


def _compute_total_duration(sorted_spans: list[dict[str, Any]]) -> float | None:
    """Total trace duration: first span start to last span end."""
    if not sorted_spans:
        return None
    t0 = _sort_key(sorted_spans[0])
    t1 = _sort_key(sorted_spans[-1])
    last_duration = sorted_spans[-1].get("duration_ms") or 0
    if t1 > t0 > 0:
        return (t1 - t0) * 1000 + float(last_duration)
    # Fallback: agent root span duration
    for span in sorted_spans:
        if classify_span(span) == SpanType.AGENT_ROOT:
            d = span.get("duration_ms")
            if d:
                return float(d)
    return None


def _span_duration_ms(span: dict[str, Any]) -> float | None:
    """Return a non-negative span duration when the source reported one."""
    duration = span.get("duration_ms")
    if duration is None:
        return None
    try:
        parsed = float(duration)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None
