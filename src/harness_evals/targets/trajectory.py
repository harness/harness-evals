"""Trajectory synthesis — build a best-effort ``messages`` trace for a target.

Targets grade a *shipped system* (HTTP) or a *prompt+model pair* (prompt) and
do not drive an agent loop themselves. When the system reports its own
trajectory (e.g. via ``messages_path``) that report is authoritative and is
used verbatim. Otherwise we assemble the richest honest trajectory we can from
the information actually observed — the user input, any captured tool calls,
and the final output — so agent/trajectory metrics have something to grade
instead of ``None``.

This is *assembly*, not *execution*: it never invents tool steps that were not
observed. A bare prompt call yields a truthful two-message ``[user, assistant]``
trace; a system that returned tool calls yields those too.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence

from harness_evals.core.types import Message, ToolCall
from harness_evals.utils.path import extract_path

logger = logging.getLogger(__name__)


def _as_content(value: object) -> str:
    """Render an input/output value as message content (JSON-encode non-strings)."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def synthesize_messages(
    input_value: object,
    output_value: object,
    tool_calls: list[ToolCall] | None = None,
) -> list[Message] | None:
    """Assemble a best-effort trajectory from observed target data.

    Builds ``[user, (assistant tool_calls?), assistant output?]`` — the user
    turn from ``input_value``, an assistant turn carrying any observed
    ``tool_calls``, and an assistant turn carrying the final output. Turns with
    no content are omitted; returns ``None`` only if nothing at all is available.

    Callers should apply this *only* when the target did not report its own
    ``messages`` — a reported trajectory always takes precedence.
    """
    messages: list[Message] = [Message(role="user", content=_as_content(input_value))]

    if tool_calls:
        messages.append(Message(role="assistant", content=None, tool_calls=list(tool_calls)))

    out_content = _as_content(output_value)
    if out_content:
        messages.append(Message(role="assistant", content=out_content))

    return messages or None


def coerce_messages(value: object) -> list[Message] | None:
    """Coerce an extracted ``messages`` value into ``Message`` objects.

    A trajectory reported by the target arrives as raw dicts (JSONPath
    extraction). Metrics read ``msg.role``/``msg.content`` as attributes, so
    normalize dicts into ``Message`` instances. Already-``Message`` items pass
    through; non-list or unparseable values yield ``None``.
    """
    if not isinstance(value, list):
        return None
    if not value:
        # An explicitly-empty reported trajectory is valid (the agent reported
        # no turns), not malformed — distinguish it from unparseable content.
        return []
    coerced: list[Message] = []
    for item in value:
        if isinstance(item, Message):
            coerced.append(item)
        elif isinstance(item, dict) and "role" in item:
            coerced.append(Message.from_dict(item))
    return coerced or None


def coerce_tool_calls(value: object) -> list[ToolCall] | None:
    """Coerce an extracted ``tool_calls`` value into ``ToolCall`` objects.

    Mirrors :func:`coerce_messages`: metrics call ``tc.to_dict()``, so raw
    dicts from JSONPath extraction must become ``ToolCall`` instances.
    """
    if not isinstance(value, list):
        return None
    coerced: list[ToolCall] = []
    for item in value:
        if isinstance(item, ToolCall):
            coerced.append(item)
        elif isinstance(item, dict) and "name" in item:
            coerced.append(ToolCall.from_dict(item))
    return coerced or None


def normalize_trajectory_fields(kwargs: dict, messages_path: str | None) -> None:
    """Coerce extracted ``messages``/``tool_calls`` in ``kwargs`` into objects in place.

    Metrics read ``messages``/``tool_calls`` as ``Message``/``ToolCall`` objects,
    but JSONPath extraction yields raw dicts. A reported trajectory that fails to
    coerce (``coerce_messages`` returns ``None``) is an instrumentation failure:
    it becomes an empty list — never absent — so the caller does NOT silently
    synthesize a clean trace over it, and metrics surface "no messages" instead.
    """
    if "messages" in kwargs:
        coerced = coerce_messages(kwargs["messages"])
        if coerced is None:
            logger.warning(
                "messages_path %r extracted a value that could not be coerced into a "
                "trajectory (%s); leaving messages empty rather than synthesizing over "
                "reported data",
                messages_path,
                type(kwargs["messages"]).__name__,
            )
            kwargs["messages"] = []
        else:
            kwargs["messages"] = coerced
    if "tool_calls" in kwargs:
        kwargs["tool_calls"] = coerce_tool_calls(kwargs["tool_calls"])


def _is_tool_result(entry: dict) -> bool:
    """A stream entry is a tool *result* (not a call) when it carries an output
    but no input — e.g. an agent that emits the call, then later the result
    against the same ``tool_calls_path``. Ambiguous entries are treated as calls.
    """
    has_input = entry.get("arguments") is not None or entry.get("input") is not None
    has_output = entry.get("output") is not None or entry.get("result") is not None
    return has_output and not has_input


def reconstruct_stream_messages(
    decoded: Sequence[tuple[str, object]],
    input_value: object,
    *,
    output_path: str,
    tool_calls_path: str | None,
    tool_calls_event: str | Sequence[str] | None = None,
    tool_results_event: str | Sequence[str] | None = None,
    tool_results_path: str | None = None,
) -> list[Message] | None:
    """Rebuild an ordered trajectory from decoded SSE events, in stream order.

    Applies ``output_path`` and tool extraction paths to JSON chunks as they
    arrive, interleaving assistant text with tool calls and tool results the way
    the agent emitted them:

    - text resolved via ``output_path`` is buffered and flushed as an
      ``assistant`` message at each tool boundary and at end of stream;
    - a chunk whose ``tool_calls_path`` resolves to tool *calls* becomes an
      ``assistant`` message carrying those ``ToolCall`` objects. When
      ``tool_calls_event`` is set, only matching SSE events are considered;
    - a chunk carrying tool *results* (output but no input) becomes ``tool``
      messages so metrics can see what each call returned. Results may either
      be inferred from ``tool_calls_path`` entries, or extracted explicitly via
      ``tool_results_path`` scoped by ``tool_results_event``.

    Returns ``None`` when the stream carried no structure this can assemble
    (no tool calls and no text deltas) — the caller then falls back to the
    plain synthesized envelope. The leading ``user`` turn is always included
    when anything is reconstructed. This never invents steps: every message
    comes from an observed chunk.
    """
    messages: list[Message] = [Message(role="user", content=_as_content(input_value))]
    text_buffer: list[str] = []
    produced = False

    def flush_text() -> None:
        if text_buffer:
            messages.append(Message(role="assistant", content="".join(text_buffer)))
            text_buffer.clear()

    for name, payload in decoded:
        if not isinstance(payload, (dict, list)):
            continue

        if tool_calls_path is not None and _event_matches(name, tool_calls_event):
            raw_tools = extract_path(payload, tool_calls_path)
            if isinstance(raw_tools, list) and raw_tools:
                calls: list[ToolCall] = []
                results: list[ToolCall] = []
                for entry in raw_tools:
                    if not isinstance(entry, dict) or "name" not in entry:
                        continue
                    if _is_tool_result(entry):
                        results.append(ToolCall.from_dict(entry))
                    else:
                        calls.append(ToolCall.from_dict(entry))
                if calls:
                    flush_text()
                    messages.append(Message(role="assistant", content=None, tool_calls=calls))
                    produced = True
                for result in results:
                    # Preserve falsey outputs (0, False, []) — only a missing
                    # (None) output becomes empty content.
                    content = "" if result.output is None else _as_content(result.output)
                    messages.append(Message(role="tool", content=content, tool_calls=[result]))
                    produced = True

        if tool_results_path is not None and _event_matches(name, tool_results_event):
            raw_results = extract_path(payload, tool_results_path)
            if isinstance(raw_results, list) and raw_results:
                for entry in raw_results:
                    if not isinstance(entry, dict) or "name" not in entry:
                        continue
                    result = ToolCall.from_dict(entry)
                    content = "" if result.output is None else _as_content(result.output)
                    messages.append(Message(role="tool", content=content, tool_calls=[result]))
                    produced = True

        text_val = extract_path(payload, output_path)
        if isinstance(text_val, str) and text_val:
            text_buffer.append(text_val)
            produced = True

    flush_text()

    if not produced:
        return None
    return messages


def _event_matches(event_name: str, selector: str | Sequence[str] | None) -> bool:
    if selector is None:
        return True
    if isinstance(selector, str):
        return event_name == selector
    return event_name in selector
