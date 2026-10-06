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

Unscorable-content rule: callers should treat a trace whose adapted
`eval_case.output` is empty as unscorable and skip it before any metric
runs (no judge spend, no persisted score).
"""

from __future__ import annotations

import ast
import enum
import json
import logging
from datetime import datetime, timezone
from typing import Any

from harness_evals import EvalCase, Message, ToolCall

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
    attrs = _attrs(span)
    input_tokens = span.get("input_tokens")
    output_tokens = span.get("output_tokens")
    if input_tokens is None:
        input_tokens = attrs.get("gen_ai.usage.input_tokens") or 0
    if output_tokens is None:
        output_tokens = attrs.get("gen_ai.usage.output_tokens") or 0
    if isinstance(input_tokens, str):
        input_tokens = int(input_tokens) if input_tokens.isdigit() else 0
    if isinstance(output_tokens, str):
        output_tokens = int(output_tokens) if output_tokens.isdigit() else 0
    return int(input_tokens), int(output_tokens)


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
) -> tuple[EvalCase, list[str]]:
    """Convert a list of span dicts into an EvalCase.

    Returns (EvalCase, warnings); warnings mark missing content that may
    affect metric applicability. An empty `eval_case.output` means the trace
    is unscorable — callers skip it (no metric runs, no judge spend).

    ``aggregate_tool_call_attribute_keys`` is opt-in so callers can map
    source-specific aggregate attributes without embedding their convention in
    the shared adapter.
    """
    warnings: list[str] = []

    if not spans:
        return EvalCase(input="", output=""), ["No spans found in trace"]

    normalized = [normalize_span(s) for s in spans]
    sorted_spans = sorted(normalized, key=_sort_key)

    classified: list[tuple[SpanType, dict[str, Any]]] = []
    for span in sorted_spans:
        classified.append((classify_span(span), span))

    user_input: str | None = None
    last_output: str | None = None
    messages: list[Message] = []
    tool_calls: list[ToolCall] = []
    root_span: dict[str, Any] | None = None

    for span_type, span in classified:
        if span_type == SpanType.AGENT_ROOT:
            root_span = span
            if not user_input:
                user_input = _extract_input_from_span(span)
            _append_user_messages(
                messages,
                _messages_from_attrs(_attrs(span), ("gen_ai.input.messages", "gen_ai.input_messages"))
                or _langfuse_messages(_attrs(span).get("langfuse.observation.input")),
            )

        elif span_type == SpanType.LLM_TURN:
            if not user_input:
                user_input = _extract_input_from_span(span)
            _append_user_messages(
                messages,
                _messages_from_attrs(_attrs(span), ("gen_ai.input.messages", "gen_ai.input_messages"))
                or _langfuse_messages(_attrs(span).get("langfuse.observation.input")),
            )

            # Tool call intents from chat spans go to the trajectory only —
            # execute_tool spans are the authoritative record (with results).
            content, turn_tool_calls = _extract_output_from_span(span)

            if content:
                last_output = content

            if content or turn_tool_calls:
                messages.append(Message(role="assistant", content=content, tool_calls=turn_tool_calls or None))
            else:
                messages.append(Message(role="assistant", content=""))

        elif span_type == SpanType.TOOL_CALL:
            tc = _extract_tool_from_span(span)
            tool_calls.append(tc)
            if not _has_matching_message_tool_call(messages, tc):
                messages.append(Message(role="assistant", content=None, tool_calls=[tc]))
            if tc.output:
                messages.append(Message(role="tool", content=str(tc.output)))

    # Concrete spans and message-level calls are authoritative when they
    # overlap, but aggregate telemetry may still contain additional calls.
    message_tool_calls = [tool_call for message in messages for tool_call in message.tool_calls or []]
    aggregate_tool_calls = _aggregate_tool_calls(sorted_spans, aggregate_tool_call_attribute_keys, warnings)
    additional_aggregate_calls = [
        candidate
        for candidate in aggregate_tool_calls
        if not any(_same_tool_call(existing, candidate) for existing in [*tool_calls, *message_tool_calls])
    ]
    if additional_aggregate_calls:
        tool_calls.extend(additional_aggregate_calls)
        messages.append(
            Message(
                role="assistant",
                content=None,
                tool_calls=additional_aggregate_calls,
            )
        )

    if not last_output:
        warnings.append(
            "No LLM output content found in trace spans. "
            "Metrics requiring output text may produce low-confidence scores."
        )

    if user_input and not any(message.role == "user" for message in messages):
        messages.insert(0, Message(role="user", content=user_input))

    total_input_tokens = 0
    total_output_tokens = 0
    for _, span in classified:
        inp, out = _get_token_counts(span)
        total_input_tokens += inp
        total_output_tokens += out

    metadata: dict[str, Any] = {
        "trace_id": spans[0].get("trace_id"),
        "source": "online_eval",
    }
    if root_span:
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

    eval_case = EvalCase(
        input=user_input or "",
        output=last_output or "",
        expected=None,
        messages=messages if messages else None,
        tool_calls=tool_calls if tool_calls else None,
        latency_ms=_compute_total_duration(sorted_spans),
        token_count=total_input_tokens + total_output_tokens,
        input_tokens=total_input_tokens or None,
        output_tokens=total_output_tokens or None,
        metadata=metadata,
    )

    return eval_case, warnings


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
) -> tuple[EvalCase, list[str]]:
    """Convert the subtree rooted at target_span_id into an EvalCase (span scope)."""
    subtree = extract_span_subtree(spans, target_span_id)
    if not subtree:
        return EvalCase(input="", output=""), [f"Span {target_span_id} not found in trace"]

    eval_case, warnings = spans_to_eval_case(
        subtree,
        aggregate_tool_call_attribute_keys=aggregate_tool_call_attribute_keys,
    )

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
