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

    # Langfuse-instrumented dialect: langfuse.observation.type is a closed
    # enum {span, generation, agent, tool}.
    langfuse_type = attrs.get("langfuse.observation.type", "")
    if langfuse_type == "agent":
        return SpanType.AGENT_ROOT
    if langfuse_type == "generation":
        return SpanType.LLM_TURN
    if langfuse_type == "tool":
        return SpanType.TOOL_CALL

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


def _extract_tool_calls_from_parts(parts: list[dict[str, Any]]) -> list[ToolCall]:
    """Extract tool calls from a message's parts array."""
    tool_calls = []
    for part in parts:
        if part.get("type") == "tool_call":
            tool_calls.append(
                ToolCall(
                    name=part.get("name", ""),
                    input=part.get("arguments"),
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


def _extract_output_from_span(span: dict[str, Any]) -> tuple[str | None, list[ToolCall]]:
    """Extract assistant output text and tool calls from an LLM span's output messages."""
    output_messages = _parse_messages(_attrs(span).get("gen_ai.output.messages"))

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for msg in output_messages:
        role = msg.get("role", "")
        parts = msg.get("parts", [])
        if role == "assistant" or not role:
            text = _extract_text_from_parts(parts)
            if text:
                text_parts.append(text)
            tcs = _extract_tool_calls_from_parts(parts)
            tool_calls.extend(tcs)

    content = "\n".join(text_parts) if text_parts else None
    return content, tool_calls


def _extract_input_from_span(span: dict[str, Any]) -> str | None:
    """Extract the user's input from an LLM span's input messages."""
    input_messages = _parse_messages(_attrs(span).get("gen_ai.input.messages"))

    for msg in reversed(input_messages):
        role = msg.get("role", "")
        if role == "user":
            parts = msg.get("parts", [])
            text = _extract_text_from_parts(parts)
            if text:
                return text
            if "content" in msg and isinstance(msg["content"], str):
                return msg["content"]
    return None


def _extract_tool_from_span(span: dict[str, Any]) -> ToolCall:
    """Extract tool call info from an execute_tool span."""
    attrs = _attrs(span)
    span_name = span.get("span_name") or span.get("name") or ""

    # Per spec: span name is "execute_tool {gen_ai.tool.name}"
    name = ""
    if span_name.startswith("execute_tool "):
        name = span_name[len("execute_tool ") :]

    input_messages = _parse_messages(attrs.get("gen_ai.input.messages"))
    tool_input: Any = None
    for msg in input_messages:
        parts = msg.get("parts", [])
        for part in parts:
            if part.get("type") == "tool_call":
                if not name:
                    name = part.get("name", "")
                tool_input = part.get("arguments")
                break

    output_messages = _parse_messages(attrs.get("gen_ai.output.messages"))
    tool_output: Any = None
    for msg in output_messages:
        parts = msg.get("parts", [])
        results = _extract_tool_results_from_parts(parts)
        if results:
            tool_output = next(iter(results.values()))
            break

    return ToolCall(name=name, input=tool_input, output=tool_output)


def _get_token_counts(span: dict[str, Any]) -> tuple[int, int]:
    """Input/output tokens: narrow columns win, gen_ai.usage.* attrs fallback."""
    attrs = _attrs(span)
    input_tokens = span.get("input_tokens") or attrs.get("gen_ai.usage.input_tokens") or 0
    output_tokens = span.get("output_tokens") or attrs.get("gen_ai.usage.output_tokens") or 0
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


def spans_to_eval_case(spans: list[dict[str, Any]]) -> tuple[EvalCase, list[str]]:
    """Convert a list of span dicts into an EvalCase.

    Returns (EvalCase, warnings); warnings mark missing content that may
    affect metric applicability. An empty `eval_case.output` means the trace
    is unscorable — callers skip it (no metric runs, no judge spend).
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

        elif span_type == SpanType.LLM_TURN:
            if not user_input:
                user_input = _extract_input_from_span(span)

            # Tool call intents from chat spans go to the trajectory only —
            # execute_tool spans are the authoritative record (with results).
            content, turn_tool_calls = _extract_output_from_span(span)

            if content:
                last_output = content
                messages.append(Message(role="assistant", content=content))

            if turn_tool_calls:
                messages.append(Message(role="assistant", content=None, tool_calls=turn_tool_calls))

            if not content and not turn_tool_calls:
                messages.append(Message(role="assistant", content=""))

        elif span_type == SpanType.TOOL_CALL:
            tc = _extract_tool_from_span(span)
            tool_calls.append(tc)
            messages.append(Message(role="assistant", content=None, tool_calls=[tc]))
            if tc.output:
                messages.append(Message(role="tool", content=str(tc.output)))

    if not last_output:
        warnings.append(
            "No LLM output content found in trace spans. "
            "Metrics requiring output text may produce low-confidence scores."
        )

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


def _extract_subtree(spans: list[dict[str, Any]], target_span_id: str) -> list[dict[str, Any]]:
    """Extract the subtree rooted at target_span_id (inclusive)."""
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
        subtree_ids.add(current)
        queue.extend(children.get(current, []))

    return [span_by_id[sid] for sid in subtree_ids if sid in span_by_id]


def spans_to_eval_case_for_span(spans: list[dict[str, Any]], target_span_id: str) -> tuple[EvalCase, list[str]]:
    """Convert the subtree rooted at target_span_id into an EvalCase (span scope)."""
    subtree = _extract_subtree(spans, target_span_id)
    if not subtree:
        return EvalCase(input="", output=""), [f"Span {target_span_id} not found in trace"]

    eval_case, warnings = spans_to_eval_case(subtree)

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
