"""OTEL eval-case importer — hydrate EvalCases from OpenTelemetry span data.

Requires: pip install harness-evals[otlp]
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import json
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.types import Message, ToolCall
from harness_evals.importers.base import BaseEvalCaseSource
from harness_evals.importers.trace_batch import (
    SpanTrace,
    TraceCatalog,
    complete_trace,
    filter_traces,
    group_traces,
    merge_span_groups,
    parse_bool,
    parse_group_by,
    parse_limit,
    parse_tags,
    time_window_from_extra,
    traces_from_json,
)
from harness_evals.plugins import register_eval_case_source
from harness_evals.refs import ResourceRef

# ------------------------------------------------------------------
# Span classification
# ------------------------------------------------------------------

_OP_INVOKE_AGENT = "invoke_agent"
_OP_INVOKE_WORKFLOW = "invoke_workflow"  # proposed extension, not yet in ratified spec
_OP_CHAT = "chat"
_OP_GENERATE_CONTENT = "generate_content"
_OP_TEXT_COMPLETION = "text_completion"
_OP_EXECUTE_TOOL = "execute_tool"


class SpanType(enum.Enum):
    AGENT_ROOT = "agent_root"
    LLM_TURN = "llm_turn"
    TOOL_CALL = "tool_call"
    OTHER = "other"


def classify_span(span: dict[str, Any]) -> SpanType:
    """Classify a span using gen_ai.operation.name (OTel GenAI semconv).

    Falls back to heuristic name matching for legacy traces.
    """
    attrs = span.get("attributes") or {}
    operation = attrs.get("gen_ai.operation.name", "")

    if operation in (_OP_INVOKE_AGENT, _OP_INVOKE_WORKFLOW):
        return SpanType.AGENT_ROOT

    if operation in (_OP_CHAT, _OP_GENERATE_CONTENT, _OP_TEXT_COMPLETION):
        return SpanType.LLM_TURN

    if operation == _OP_EXECUTE_TOOL:
        return SpanType.TOOL_CALL

    # Legacy heuristics
    name = (span.get("name") or span.get("span_name") or "").lower()

    if not span.get("parent_span_id"):
        # Root span: check if it looks like an LLM call (has messages/completion)
        has_llm_content = (
            "gen_ai.input_messages" in attrs
            or "gen_ai.input.messages" in attrs
            or "gen_ai.prompt" in attrs
            or "gen_ai.completion" in attrs
            or "gen_ai.output_messages" in attrs
            or "gen_ai.output.messages" in attrs
        )
        if has_llm_content:
            return SpanType.LLM_TURN
        return SpanType.AGENT_ROOT

    if "gen_ai.tool.name" in attrs or "tool.name" in attrs:
        return SpanType.TOOL_CALL
    # mcp.* spans are transport-layer (HTTP calls under execute_tool) — skip them
    if name.startswith("mcp."):
        return SpanType.OTHER
    if name.startswith("execute_tool") or ("tool" in name and "gen_ai" not in name):
        return SpanType.TOOL_CALL

    # Langfuse-instrumented agent run spans
    if attrs.get("langfuse.observation.type") == "agent":
        return SpanType.AGENT_ROOT

    # Langfuse-instrumented LLM turn spans (e.g. "llm_turn_N")
    if attrs.get("langfuse.observation.type") == "generation":
        return SpanType.LLM_TURN

    # Langfuse-instrumented tool spans (e.g. "tool:Read", "tool:Skill")
    if (
        attrs.get("langfuse.observation.type") == "span"
        and "gen_ai.tool.name" not in attrs
        and name.startswith("tool:")
    ):
        return SpanType.TOOL_CALL

    if (
        "gen_ai.request.model" in attrs
        or "gen_ai.system" in attrs
        or "gen_ai.provider.name" in attrs
        or "gen_ai" in name
        or "llm" in name
    ):
        return SpanType.LLM_TURN

    return SpanType.OTHER


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


@register_eval_case_source("otel")
class OTELEvalCaseSource(BaseEvalCaseSource):
    """Fetch EvalCases from OpenTelemetry span data.

    Produces a conversation ``EvalCase`` (one per trace, or one per session when
    ``group_by=session_id``) with a deduplicated trajectory suitable for both
    per-turn and whole-conversation metrics.

    Follows the `OpenTelemetry Semantic Conventions for Generative AI
    <https://github.com/open-telemetry/semantic-conventions-genai>`_.

    **Uniform entry point** — ``fetch(ref)``::

        source = OTELEvalCaseSource()
        cases = await source.fetch(resolve("otel://./trace_spans.json"))

        # Optional TraceCatalog (e.g. Langfuse) lists traces in a time window:
        source = OTELEvalCaseSource(catalog=catalog)
        cases = await source.fetch(ResourceRef(
            source="otel",
            id="",
            extra={"lookback_days": 7, "group_by": "session_id", "limit": 200},
        ))

    ``ref.extra`` (all optional, vendor-neutral)::

        lookback_days       Rolling window ending at ``now`` (int).
        from_timestamp      Inclusive ISO-8601 / datetime start.
        to_timestamp        Exclusive ISO-8601 / datetime end.
        group_by            ``session_id`` / ``conversation_id`` to merge traces
                            that share ``gen_ai.conversation.id`` (or ``session.id``).
                            Default is one EvalCase per trace.
        expand_sessions     If true and the catalog implements
                            ``list_session_trace_ids``, hydrate every trace in
                            each in-window session (HITL/resume siblings).
        limit               Cap listed traces (catalog) or output cases (files).
        name, tags, user_id, session_id
                            Forwarded to ``TraceCatalog.list_traces``.

    **Convenience methods**::

        # From exported JSON span data → single EvalCase with full trajectory
        ec = OTELEvalCaseSource.from_span_json(json.load(f))

        # From SDK ReadableSpan objects
        ec = OTELEvalCaseSource.from_spans(sdk_spans)
    """

    name = "otel"

    def __init__(
        self,
        catalog: TraceCatalog | None = None,
        *,
        now: Callable[[], datetime] | None = None,
        concurrency: int = 10,
    ) -> None:
        self._catalog = catalog
        self._now = now
        self._concurrency = concurrency

    async def fetch(self, ref: ResourceRef) -> list[EvalCase]:
        """Load traces from a file/directory or a ``TraceCatalog``, then convert.

        ``ref.id`` is a JSON file or directory of JSON files when no catalog
        listing is requested. Catalog listing is used when ``catalog`` is set
        and ``ref.id`` is empty, or when a time/filter extra is present.
        """
        extra = ref.extra
        group_by = parse_group_by(extra.get("group_by"))
        window = time_window_from_extra(extra, now=self._now)
        expand = parse_bool(extra.get("expand_sessions"))
        has_list_filters = any(
            extra.get(k) not in (None, "")
            for k in ("lookback_days", "from_timestamp", "to_timestamp", "name", "tags", "user_id", "session_id")
        )
        use_catalog = self._catalog is not None and (not ref.id or has_list_filters)

        if use_catalog:
            if self._catalog is None:
                raise ValueError(
                    "OTELEvalCaseSource.fetch() time/filter extras require a TraceCatalog "
                    "(pass catalog=... when constructing the source)"
                )
            limit = parse_limit(extra.get("limit"), default=100) or 100
            listed = await asyncio.to_thread(
                self._catalog.list_traces,
                from_timestamp=window.from_timestamp,
                to_timestamp=window.to_timestamp,
                session_id=extra.get("session_id") or None,
                name=extra.get("name") or None,
                tags=parse_tags(extra.get("tags")),
                user_id=extra.get("user_id") or None,
                limit=limit,
            )
            traces = await self._hydrate_catalog_traces(listed, expand_sessions=expand)
        elif ref.id:
            traces = await asyncio.to_thread(_load_traces_from_path, ref.id)
            traces = filter_traces(traces, window)
            file_limit = parse_limit(extra.get("limit"), default=None)
            if file_limit is not None:
                traces = traces[:file_limit]
        else:
            raise ValueError(
                "OTELEvalCaseSource.fetch() requires a file path in ref.id, or a TraceCatalog "
                "with lookback_days / from_timestamp / filter extras"
            )

        return self.from_span_traces(traces, group_by=group_by)

    @staticmethod
    def from_spans(spans: list[Any]) -> EvalCase:
        """Convert a list of OTel ``ReadableSpan`` objects to a single EvalCase.

        Requires ``opentelemetry-sdk`` to be installed.
        """
        converted = _convert_sdk_spans(spans)
        return _build_conversation_eval_case(converted)

    @staticmethod
    def from_span_json(data: list[dict[str, Any]] | dict[str, Any]) -> EvalCase:
        """Convert exported OTEL JSON span data to a single EvalCase.

        Builds a deduplicated conversation trajectory from all LLM and tool
        spans, suitable for both per-turn metrics (via ``messages``) and
        whole-conversation metrics.

        - ``input``: the user's original query
        - ``output``: the agent's final text response
        - ``messages``: the full conversation trajectory (no duplicates)
        - ``tool_calls``: flattened list from execute_tool spans (authoritative)
        - ``token_count``: aggregated across all LLM turns
        - ``metadata``: trace_id, model, provider, per-turn token breakdown

        For several traces (and optional session merge), use
        :meth:`from_span_traces` or :meth:`fetch`.
        """
        traces = traces_from_json(data)
        if not traces:
            return _build_conversation_eval_case([])
        spans = [span for trace in traces for span in trace.spans]
        return _build_conversation_eval_case(spans)

    @staticmethod
    def from_span_traces(
        traces: Sequence[SpanTrace | dict[str, Any]],
        *,
        group_by: str | None = None,
    ) -> list[EvalCase]:
        """Convert one or more traces to EvalCases, optionally merging by session."""
        normalized: list[SpanTrace] = []
        for item in traces:
            if isinstance(item, SpanTrace):
                normalized.append(complete_trace(item))
            elif isinstance(item, dict):
                normalized.extend(traces_from_json([item]))
            else:
                raise TypeError(f"expected SpanTrace or dict, got {type(item).__name__}")

        cases: list[EvalCase] = []
        for group in group_traces(normalized, group_by=group_by):
            spans, grouping_meta = merge_span_groups(group)
            case = _build_conversation_eval_case(spans)
            if grouping_meta:
                meta = dict(case.metadata or {})
                meta.update(grouping_meta)
                case.metadata = meta
            cases.append(case)
        return cases

    async def _hydrate_catalog_traces(
        self,
        listed: Sequence[SpanTrace],
        *,
        expand_sessions: bool,
    ) -> list[SpanTrace]:
        catalog = self._catalog
        assert catalog is not None
        stubs = [complete_trace(t) for t in listed]
        if expand_sessions:
            expander = getattr(catalog, "list_session_trace_ids", None)
            if callable(expander):
                seen: dict[str, SpanTrace] = {t.trace_id: t for t in stubs if t.trace_id}
                session_ids = {t.session_id for t in stubs if t.session_id}
                for session_id in session_ids:
                    extra_ids = expander(session_id) or []
                    for tid in extra_ids:
                        if tid not in seen:
                            seen[tid] = SpanTrace(spans=[], trace_id=tid, session_id=session_id)
                stubs = list(seen.values())

        sem = asyncio.Semaphore(self._concurrency)

        async def _load(stub: SpanTrace) -> SpanTrace:
            if stub.spans:
                return complete_trace(stub)
            if not stub.trace_id:
                return stub
            async with sem:
                spans = await asyncio.to_thread(catalog.load_spans, stub.trace_id)
            filled = SpanTrace(
                spans=spans,
                trace_id=stub.trace_id,
                session_id=stub.session_id,
                start_time=stub.start_time,
                metadata=stub.metadata,
            )
            return complete_trace(filled)

        return list(await asyncio.gather(*[_load(s) for s in stubs]))


# ------------------------------------------------------------------
# Core builder: conversation trajectory (single EvalCase)
# ------------------------------------------------------------------


def _build_conversation_eval_case(spans: list[dict[str, Any]]) -> EvalCase:
    """Build a single EvalCase with a deduplicated conversation trajectory.

    Strategy:
    - Sort spans chronologically
    - Classify each span
    - Extract user input from the agent root or first LLM turn
    - Build trajectory from *output messages only* of each LLM turn
    - Tool calls come from execute_tool spans (authoritative, with results)
    - Final output = last assistant text content
    """
    if not spans:
        return EvalCase(input="", output="")

    sorted_spans = sorted(spans, key=_span_sort_key)
    classified = [(classify_span(s), s) for s in sorted_spans]

    user_input: str | None = None
    last_output: str | None = None
    messages: list[Message] = []
    tool_calls: list[ToolCall] = []
    total_input_tokens = 0
    total_output_tokens = 0
    root_span: dict[str, Any] | None = None
    turn_details: list[dict[str, Any]] = []

    has_llm_turns = any(st == SpanType.LLM_TURN for st, _ in classified)

    for span_type, span in classified:
        attrs = span.get("attributes") or {}

        if span_type == SpanType.AGENT_ROOT:
            root_span = span
            if not user_input:
                user_input = _extract_user_input_from_span(attrs)
            if not has_llm_turns:
                content, turn_tool_calls = _extract_output_from_span(attrs)
                if content:
                    last_output = content
                    messages.append(Message(role="assistant", content=content))
                if turn_tool_calls:
                    messages.append(Message(role="assistant", content=None, tool_calls=turn_tool_calls))

        elif span_type == SpanType.LLM_TURN:
            if not user_input:
                user_input = _extract_user_input_from_span(attrs)

            _recover_intermediate_user_messages(attrs, messages)

            content, turn_tool_calls = _extract_output_from_span(attrs)

            if content:
                last_output = content
                messages.append(Message(role="assistant", content=content))

            if turn_tool_calls:
                messages.append(Message(role="assistant", content=None, tool_calls=turn_tool_calls))

            inp_tok = _safe_int(attrs.get("gen_ai.usage.input_tokens", 0))
            out_tok = _safe_int(attrs.get("gen_ai.usage.output_tokens", 0))
            total_input_tokens += inp_tok
            total_output_tokens += out_tok

            turn_details.append(
                {
                    "span_id": span.get("span_id"),
                    "input_tokens": inp_tok,
                    "output_tokens": out_tok,
                    "latency_ms": _span_latency(span),
                }
            )

        elif span_type == SpanType.TOOL_CALL:
            tc = _extract_tool_from_span(span)
            tool_calls.append(tc)
            if tc.output is not None:
                # Prefer structured JSON so hallucination grounding keeps list payloads.
                if isinstance(tc.output, (dict, list)):
                    content = json.dumps(tc.output, ensure_ascii=False)
                else:
                    content = str(tc.output)
                messages.append(
                    Message(
                        role="tool",
                        content=content,
                        tool_calls=[ToolCall(name=tc.name, input=tc.input, output=tc.output)],
                    )
                )

    # If we found a user input, prepend it as the first message
    if user_input and (not messages or messages[0].role != "user"):
        messages.insert(0, Message(role="user", content=user_input))

    total_tokens = total_input_tokens + total_output_tokens

    metadata: dict[str, Any] = {}
    # Use root span for metadata; fall back to first LLM turn if no root
    meta_span = root_span
    if not meta_span:
        for st, s in classified:
            if st == SpanType.LLM_TURN:
                meta_span = s
                break
    meta_attrs = (meta_span or spans[0]).get("attributes") or {}
    _set_if(metadata, "trace_id", (meta_span or spans[0]).get("trace_id"))
    _set_if(
        metadata,
        "provider",
        meta_attrs.get("gen_ai.provider.name") or meta_attrs.get("gen_ai.system"),
    )
    _set_if(
        metadata,
        "model",
        meta_attrs.get("gen_ai.response.model") or meta_attrs.get("gen_ai.request.model"),
    )
    _set_if(metadata, "operation", meta_attrs.get("gen_ai.operation.name"))
    _set_if(metadata, "agent_name", meta_attrs.get("gen_ai.agent.name"))
    _set_if(metadata, "conversation_id", meta_attrs.get("gen_ai.conversation.id"))
    cache_read = _safe_int(meta_attrs.get("gen_ai.usage.cache_read.input_tokens"))
    if cache_read:
        metadata["cache_read_tokens"] = cache_read

    # Langfuse trace tags → EvalCase.tags (parsed from JSON array string)
    raw_tags = meta_attrs.get("langfuse.trace.tags")
    tags: dict[str, str] | None = None
    if raw_tags:
        tag_list = raw_tags if isinstance(raw_tags, list) else _try_json(raw_tags)
        if isinstance(tag_list, list):
            parsed_tags: dict[str, str] = {}
            for tag in tag_list:
                if isinstance(tag, str) and ":" in tag:
                    k, _, v = tag.partition(":")
                    parsed_tags[k] = v
                elif isinstance(tag, str):
                    parsed_tags[tag] = "true"
            if parsed_tags:
                tags = parsed_tags

    if turn_details:
        metadata["turns"] = turn_details

    latency_ms = _compute_trace_latency(sorted_spans)

    # Prefer summing observation-level costs; fall back to a single trace total.
    cost_usd: float | None = None
    obs_cost_total = 0.0
    saw_obs_cost = False
    trace_cost: float | None = None
    for span in sorted_spans:
        attrs = span.get("attributes") or {}
        raw_obs = attrs.get("gen_ai.usage.cost")
        if raw_obs is not None:
            with contextlib.suppress(TypeError, ValueError):
                obs_cost_total += float(raw_obs)
                saw_obs_cost = True
        raw_trace = attrs.get("langfuse.trace.total_cost")
        if raw_trace is not None and trace_cost is None:
            with contextlib.suppress(TypeError, ValueError):
                trace_cost = float(raw_trace)
    if saw_obs_cost and obs_cost_total > 0:
        cost_usd = obs_cost_total
    elif trace_cost is not None and trace_cost > 0:
        cost_usd = trace_cost
    else:
        raw_cost = meta_attrs.get("gen_ai.usage.cost")
        if raw_cost is not None:
            with contextlib.suppress(TypeError, ValueError):
                parsed = float(raw_cost)
                if parsed > 0:
                    cost_usd = parsed

    # Prefer Langfuse trace-level I/O when generation spans omitted payloads.
    # For session merges, earliest non-empty prompt is the initial user input;
    # latest non-empty text is the final agent answer.
    if not user_input:
        for span in sorted_spans:
            attrs = span.get("attributes") or {}
            text = _text_from_langfuse_io(
                attrs.get("langfuse.trace.input"),
                # Prefer short `prompt` (the user question) over full `user_message`
                # which includes <current_context> + URL JSON for UA traces.
                keys=("prompt", "user_message", "text", "input"),
            )
            if text:
                user_input = text
                break
    if not last_output:
        for span in reversed(sorted_spans):
            attrs = span.get("attributes") or {}
            text = _text_from_langfuse_io(
                attrs.get("langfuse.trace.output"),
                keys=("text", "output", "content"),
            )
            if text:
                last_output = text
                break

    if user_input and (not messages or messages[0].role != "user"):
        messages.insert(0, Message(role="user", content=user_input))
    elif user_input and messages and messages[0].role == "user" and _looks_like_injected_user_context(
        messages[0].content or ""
    ):
        messages[0] = Message(role="user", content=user_input)

    return EvalCase(
        input=user_input or "",
        output=last_output or "",
        messages=messages or None,
        tool_calls=tool_calls or None,
        latency_ms=latency_ms,
        token_count=total_tokens if total_tokens > 0 else None,
        cost_usd=cost_usd,
        tags=tags,
        metadata=metadata or None,
    )


# ------------------------------------------------------------------
# Span content extraction
# ------------------------------------------------------------------


def _extract_user_input_from_span(attrs: dict) -> str:
    """Extract the user's text input from span attributes.

    Prefer Langfuse *trace*-level input (the real user prompt) over generation
    ``input_messages``, which often end with injected ``<system-reminder>`` /
    skill blobs that are role=user but are not the session prompt.
    """
    # Langfuse trace I/O: {user_message|prompt|...} on unified-agent traces.
    for key in ("langfuse.trace.input", "langfuse.observation.input"):
        raw = attrs.get(key)
        if not raw:
            continue
        text = _text_from_langfuse_io(
            raw, keys=("prompt", "user_message", "text", "input")
        )
        if text:
            return text

    # New semconv: gen_ai.input_messages or gen_ai.input.messages
    for key in ("gen_ai.input_messages", "gen_ai.input.messages"):
        raw = attrs.get(key)
        if not raw:
            continue
        parsed = raw if isinstance(raw, list) else _try_json(raw)
        if not isinstance(parsed, list):
            continue
        # Walk in reverse to find the last *real* user message (skip skill injections).
        for msg in reversed(parsed):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            text = _text_from_parts(msg.get("parts", []))
            if not text and "content" in msg and isinstance(msg["content"], str):
                text = msg["content"]
            if text and not _looks_like_injected_user_context(text):
                return text

    # Legacy: gen_ai.prompt / gen_ai.input
    prompt = attrs.get("gen_ai.prompt") or attrs.get("gen_ai.input")
    if isinstance(prompt, str):
        try:
            parsed = json.loads(prompt)
            if isinstance(parsed, list):
                for entry in reversed(parsed):
                    if isinstance(entry, dict) and entry.get("role") == "user":
                        content = entry.get("content", "")
                        if isinstance(content, str) and content and not _looks_like_injected_user_context(
                            content
                        ):
                            return content
        except (json.JSONDecodeError, TypeError):
            if not _looks_like_injected_user_context(prompt):
                return prompt

    return ""


def _text_from_langfuse_io(raw: object, *, keys: tuple[str, ...]) -> str:
    """Pull a human prompt/answer string out of Langfuse trace input/output shapes."""
    if isinstance(raw, str) and raw.strip():
        return raw
    parsed = raw if isinstance(raw, dict) else _try_json(raw)
    if not isinstance(parsed, dict):
        return ""
    for key in keys:
        val = parsed.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _looks_like_injected_user_context(text: str) -> bool:
    """True for Claude Agent SDK skill / system-reminder blobs role=user."""
    head = text.lstrip()[:80].lower()
    return head.startswith("<system-reminder>") or head.startswith("base directory for this skill:")


def _recover_intermediate_user_messages(attrs: dict, messages: list[Message]) -> None:
    """Append user messages from this turn's input that aren't already in the trajectory.

    In multi-turn conversations, each LLM turn's input_messages contains the full
    history including new user messages. We detect new user messages by checking the
    tail of input_messages for user content not already present in the trajectory.
    """
    for key in ("gen_ai.input_messages", "gen_ai.input.messages"):
        raw = attrs.get(key)
        if not raw:
            continue
        parsed = raw if isinstance(raw, list) else _try_json(raw)
        if not isinstance(parsed, list):
            continue

        existing_user_texts = {m.content for m in messages if m.role == "user" and m.content}

        for msg in parsed:
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            text = _text_from_parts(msg.get("parts", []))
            if not text and "content" in msg and isinstance(msg["content"], str):
                text = msg["content"]
            if text and text not in existing_user_texts:
                messages.append(Message(role="user", content=text))
                existing_user_texts.add(text)
        return


def _extract_output_from_span(attrs: dict) -> tuple[str | None, list[ToolCall]]:
    """Extract assistant output text and tool calls from an LLM span's output.

    Returns (text_content, tool_calls).
    """
    tool_calls: list[ToolCall] = []

    # New semconv: gen_ai.output_messages or gen_ai.output.messages
    for key in ("gen_ai.output_messages", "gen_ai.output.messages"):
        raw = attrs.get(key)
        if not raw:
            continue
        parsed = raw if isinstance(raw, list) else _try_json(raw)
        if not isinstance(parsed, list):
            continue

        text_parts: list[str] = []
        for msg in parsed:
            if not isinstance(msg, dict):
                continue
            parts = msg.get("parts", [])
            text = _text_from_parts(parts)
            if text:
                text_parts.append(text)
            tcs = _tool_calls_from_parts(parts)
            tool_calls.extend(tcs)
            # Fallback: direct content
            if not parts and "content" in msg:
                text_parts.append(msg["content"])

        content = "\n".join(text_parts) if text_parts else None
        return content, tool_calls

    # Legacy: gen_ai.completion / gen_ai.output
    completion = attrs.get("gen_ai.completion") or attrs.get("gen_ai.output")
    if isinstance(completion, str):
        try:
            parsed = json.loads(completion)
            if isinstance(parsed, dict) and "content" in parsed:
                return parsed["content"], []
            if isinstance(parsed, dict) and "role" in parsed:
                return parsed.get("content"), []
            if isinstance(parsed, list):
                for entry in parsed:
                    if isinstance(entry, dict) and "content" in entry:
                        return entry["content"], []
        except (json.JSONDecodeError, TypeError):
            return completion, []

    # Langfuse instrumentation: langfuse.observation.output
    # Stored as {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
    raw = attrs.get("langfuse.observation.output")
    if raw:
        parsed = raw if isinstance(raw, dict) else _try_json(raw)
        if isinstance(parsed, dict):
            content = parsed.get("content")
            if isinstance(content, str):
                return content, []
            if isinstance(content, list):
                text_parts: list[str] = []
                tcs: list[ToolCall] = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        t = part.get("text") or part.get("content") or ""
                        if t:
                            text_parts.append(t)
                    elif part.get("type") == "tool_use":
                        tcs.append(
                            ToolCall(
                                name=part.get("name", ""),
                                input=part.get("input"),
                            )
                        )
                return "\n".join(text_parts) or None, tcs

    return None, []


def _extract_tool_from_span(span: dict[str, Any]) -> ToolCall:
    """Extract tool call info from an execute_tool span."""
    attrs = span.get("attributes") or {}
    name = (span.get("name") or span.get("span_name") or "").lower()

    # Tool name from attributes or span name
    tool_name = attrs.get("gen_ai.tool.name") or attrs.get("tool.name") or ""
    if not tool_name and name.startswith("execute_tool "):
        tool_name = (span.get("name") or span.get("span_name") or "")[len("execute_tool ") :]

    # Arguments from attributes
    tool_input = _parse_json_attr(attrs.get("gen_ai.tool.call.arguments") or attrs.get("tool.input"))

    # Result from attributes
    tool_output = _parse_json_or_str(attrs.get("gen_ai.tool.call.result") or attrs.get("tool.output"))

    # Fallback: extract from gen_ai.input.messages / gen_ai.output.messages
    if not tool_input:
        for key in ("gen_ai.input.messages", "gen_ai.input_messages"):
            raw = attrs.get(key)
            if not raw:
                continue
            parsed = raw if isinstance(raw, list) else _try_json(raw)
            if isinstance(parsed, list):
                for msg in parsed:
                    if not isinstance(msg, dict):
                        continue
                    for part in msg.get("parts", []):
                        if isinstance(part, dict) and part.get("type") == "tool_call":
                            if not tool_name:
                                tool_name = part.get("name", "")
                            tool_input = part.get("arguments")
                            break

    if not tool_output:
        for key in ("gen_ai.output.messages", "gen_ai.output_messages"):
            raw = attrs.get(key)
            if not raw:
                continue
            parsed = raw if isinstance(raw, list) else _try_json(raw)
            if isinstance(parsed, list):
                for msg in parsed:
                    if not isinstance(msg, dict):
                        continue
                    for part in msg.get("parts", []):
                        if isinstance(part, dict) and part.get("type") == "tool_call_response":
                            # spec field is "response"; fall back for older exporters
                            tool_output = part.get("response") or part.get("result") or part.get("content", "")
                            break

    # Langfuse instrumentation: langfuse.observation.input/output
    # input: {"name": "ToolName", "arguments": {...}}
    # output: {"tool_use_id": "...", "tool_name": "...", "is_error": false, "content": "..."}
    if not tool_input:
        raw = attrs.get("langfuse.observation.input")
        if raw:
            parsed = raw if isinstance(raw, dict) else _try_json(raw)
            if isinstance(parsed, dict):
                if not tool_name:
                    tool_name = parsed.get("name", "")
                tool_input = parsed.get("arguments") or parsed.get("input")

    if not tool_output:
        raw = attrs.get("langfuse.observation.output")
        if raw:
            parsed = raw if isinstance(raw, (dict, list, str, int, float, bool)) else _try_json(raw)
            if isinstance(parsed, dict):
                content = parsed.get("content")
                if content is not None:
                    tool_output = content if isinstance(content, str) else json.dumps(content)
                else:
                    # Full observation payload (list/get responses often omit nested content).
                    tool_output = parsed
            elif parsed is not None:
                tool_output = parsed

    return ToolCall(name=tool_name, input=tool_input, output=tool_output)


# ------------------------------------------------------------------
# Message parts helpers
# ------------------------------------------------------------------


def _text_from_parts(parts: list) -> str | None:
    """Extract concatenated text from a parts array.

    Supports both ``{"type": "text", "content": "..."}`` (semconv)
    and ``{"type": "text", "text": "..."}`` (some exporters).
    """
    if not parts:
        return None
    texts = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type", "text")
        if ptype == "text":
            text = part.get("content") or part.get("text") or ""
            if text:
                texts.append(text)
        elif ptype == "tool_call_response":
            # Include tool results as text in the trajectory
            # spec field is "response"; fall back to "result"/"content" for older exporters
            result = part.get("response") or part.get("content") or part.get("result") or ""
            if result:
                texts.append(result if isinstance(result, str) else json.dumps(result))
    return "\n".join(texts) if texts else None


def _tool_calls_from_parts(parts: list) -> list[ToolCall]:
    """Extract tool calls from a parts array."""
    tool_calls = []
    for part in parts:
        if not isinstance(part, dict) or part.get("type") != "tool_call":
            continue
        # Inline structured format
        if "name" in part:
            tool_calls.append(
                ToolCall(
                    name=part.get("name", ""),
                    input=part.get("arguments"),
                )
            )
        # Content-wrapped format (semconv style)
        else:
            tc_data = part.get("content", "")
            if isinstance(tc_data, str):
                tc_data = _try_json(tc_data)
            if isinstance(tc_data, dict):
                tool_calls.append(
                    ToolCall(
                        name=tc_data.get("name", ""),
                        input=tc_data.get("arguments") or tc_data.get("input"),
                    )
                )
    return tool_calls


# ------------------------------------------------------------------
# SDK span conversion
# ------------------------------------------------------------------


def _convert_sdk_spans(spans: list[Any]) -> list[dict[str, Any]]:
    """Convert OTel SDK ReadableSpan objects to plain dicts."""
    converted: list[dict[str, Any]] = []
    for span in spans:
        d: dict[str, Any] = {
            "name": span.name,
            "attributes": dict(span.attributes) if span.attributes else {},
            "start_time_unix_nano": span.start_time,
            "end_time_unix_nano": span.end_time,
        }
        ctx = span.get_span_context()
        if ctx:
            d["span_id"] = format(ctx.span_id, "016x")
            d["trace_id"] = format(ctx.trace_id, "032x")
        parent = getattr(span, "parent", None)
        if parent:
            d["parent_span_id"] = format(parent.span_id, "016x")
        else:
            d["parent_span_id"] = None
        converted.append(d)
    return converted


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------


def _span_sort_key(span: dict[str, Any]) -> tuple:
    """Sort key: prefer start_time_unix_nano, fall back to start_timestamp."""
    nano = span.get("start_time_unix_nano")
    if nano is not None:
        return (0, nano)
    ts = span.get("start_timestamp", "")
    return (1, ts)


def _span_latency(span: dict[str, Any]) -> float | None:
    """Compute span latency in ms from nanosecond timestamps or duration_ms."""
    start = span.get("start_time_unix_nano")
    end = span.get("end_time_unix_nano")
    if start and end:
        return (end - start) / 1_000_000
    duration = span.get("duration_ms")
    if duration is not None:
        return float(duration)
    return None


def _compute_trace_latency(sorted_spans: list[dict[str, Any]]) -> float | None:
    """Compute total trace latency from earliest start to latest end."""
    if not sorted_spans:
        return None

    # Find min start and max end across all spans
    min_start: int | None = None
    max_end: int | None = None
    for span in sorted_spans:
        start = span.get("start_time_unix_nano")
        end = span.get("end_time_unix_nano")
        if start is not None and (min_start is None or start < min_start):
            min_start = start
        if end is not None and (max_end is None or end > max_end):
            max_end = end

    if min_start is not None and max_end is not None:
        return (max_end - min_start) / 1_000_000

    # Fallback: duration_ms on root/first span
    for span in sorted_spans:
        if classify_span(span) == SpanType.AGENT_ROOT:
            d = span.get("duration_ms")
            if d is not None:
                return float(d)

    return None


def _set_if(d: dict, key: str, val: Any) -> None:
    if val:
        d[key] = val


def _read_json_file(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def _load_traces_from_path(path: str) -> list[SpanTrace]:
    target = Path(path)
    if target.is_dir():
        traces: list[SpanTrace] = []
        for file_path in sorted(target.glob("*.json")):
            traces.extend(traces_from_json(_read_json_file(str(file_path))))
        return traces
    return traces_from_json(_read_json_file(path))


def _try_json(val: Any) -> Any:
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            pass
    return val


def _parse_json_attr(val: Any) -> dict | None:
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def _parse_json_or_str(val: Any) -> str | dict | None:
    if isinstance(val, (str, dict)):
        return val
    return None


def _safe_int(val: Any) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0
