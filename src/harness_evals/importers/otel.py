"""OTEL eval-case importer — hydrate EvalCases from OpenTelemetry span data.

Requires: pip install harness-evals[otlp]
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.types import Message, ToolCall
from harness_evals.importers.base import BaseEvalCaseSource
from harness_evals.plugins import register_eval_case_source
from harness_evals.refs import ResourceRef


@register_eval_case_source("otel")
class OTELEvalCaseSource(BaseEvalCaseSource):
    """Fetch EvalCases from OpenTelemetry span data.

    Provides a uniform ``fetch(ref)`` entry point that reads span JSON from a
    local file path, plus static ``from_spans()`` and ``from_span_json()``
    convenience methods for programmatic use.

    Follows the `OpenTelemetry Semantic Conventions for Generative AI
    <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_.

    **Uniform entry point** — ``fetch(ref)``::

        source = OTELEvalCaseSource()

        # ref.id is a path to a JSON file containing a list of span dicts
        cases = await source.fetch(resolve("otel://./trace_spans.json"))

    **Convenience methods** (also available)::

        # From SDK ReadableSpan objects
        ec = OTELEvalCaseSource.from_spans(sdk_spans)

        # From exported JSON span data
        with open("trace_spans.json") as f:
            ec = OTELEvalCaseSource.from_span_json(json.load(f))
    """

    name = "otel"

    async def fetch(self, ref: ResourceRef) -> list[EvalCase]:
        """Load span JSON from the file path in ``ref.id`` and convert to EvalCase.

        ``ref.id`` must be a path to a JSON file containing a list of span dicts
        (OTLP JSON export format). Returns a single-element list.
        """
        path = ref.id
        raw = await asyncio.to_thread(_read_json_file, path)
        return [self.from_span_json(raw)]

    @staticmethod
    def from_spans(spans: list[Any]) -> EvalCase:
        """Convert a list of OTel ``ReadableSpan`` objects to an EvalCase.

        Requires ``opentelemetry-sdk`` to be installed.

        Maps:
          - gen_ai.* attributes -> messages, token_count
          - tool.* spans -> tool_calls
          - root span duration -> latency_ms
          - root span attributes -> input/output
        """
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

        return OTELEvalCaseSource._build_eval_case(converted)

    @staticmethod
    def from_span_json(data: list[dict[str, Any]]) -> EvalCase:
        """Convert exported OTEL JSON span data to an EvalCase.

        Useful when reading from files or OTLP JSON export. Each dict
        should have at minimum ``name``, ``attributes``, and optionally
        ``start_time_unix_nano``/``end_time_unix_nano``.
        """
        return OTELEvalCaseSource._build_eval_case(data)

    @staticmethod
    def _build_eval_case(spans: list[dict[str, Any]]) -> EvalCase:
        root = _find_root(spans)
        root_attrs = root.get("attributes", {}) if root else {}

        input_val = root_attrs.get("gen_ai.input") or root_attrs.get("input") or ""
        output_val = root_attrs.get("gen_ai.output") or root_attrs.get("output") or ""

        if isinstance(input_val, str):
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                input_val = json.loads(input_val)
        if isinstance(output_val, str):
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                output_val = json.loads(output_val)

        messages: list[Message] = []
        tool_calls: list[ToolCall] = []
        total_input_tokens = 0
        total_output_tokens = 0

        for span in spans:
            attrs = span.get("attributes", {})
            name = span.get("name", "")

            if _is_llm_span(name, attrs):
                _extract_messages(attrs, messages)
                total_input_tokens += _safe_int(attrs.get("gen_ai.usage.input_tokens", 0))
                total_output_tokens += _safe_int(attrs.get("gen_ai.usage.output_tokens", 0))

            elif _is_tool_span(name, attrs):
                tc = ToolCall(
                    name=attrs.get("tool.name", name),
                    input=_parse_json_attr(attrs.get("tool.input")),
                    output=_parse_json_or_str(attrs.get("tool.output")),
                )
                tool_calls.append(tc)

        latency_ms: float | None = None
        if root:
            start = root.get("start_time_unix_nano")
            end = root.get("end_time_unix_nano")
            if start and end:
                latency_ms = (end - start) / 1_000_000

        total_tokens = total_input_tokens + total_output_tokens

        return EvalCase(
            input=input_val,
            output=output_val,
            messages=messages or None,
            tool_calls=tool_calls or None,
            latency_ms=latency_ms,
            token_count=total_tokens if total_tokens > 0 else None,
        )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _read_json_file(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        return json.load(f)  # type: ignore[no-any-return]


def _find_root(spans: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Find the root span (no parent)."""
    for span in spans:
        if not span.get("parent_span_id"):
            return span
    return spans[0] if spans else None


def _is_llm_span(name: str, attrs: dict) -> bool:
    return (
        "gen_ai" in name.lower()
        or "llm" in name.lower()
        or "gen_ai.system" in attrs
        or "gen_ai.request.model" in attrs
    )


def _is_tool_span(name: str, attrs: dict) -> bool:
    return "tool" in name.lower() or "tool.name" in attrs


def _extract_messages(attrs: dict, messages: list[Message]) -> None:
    """Extract messages from gen_ai semantic convention attributes."""
    prompt = attrs.get("gen_ai.prompt") or attrs.get("gen_ai.input")
    if isinstance(prompt, str):
        try:
            parsed = json.loads(prompt)
            if isinstance(parsed, list):
                for entry in parsed:
                    if isinstance(entry, dict) and "role" in entry:
                        messages.append(
                            Message(
                                role=entry.get("role", "unknown"),
                                content=entry.get("content"),
                            )
                        )
        except (json.JSONDecodeError, TypeError):
            pass

    completion = attrs.get("gen_ai.completion") or attrs.get("gen_ai.output")
    if isinstance(completion, str):
        try:
            parsed = json.loads(completion)
            if isinstance(parsed, dict) and "role" in parsed:
                messages.append(
                    Message(
                        role=parsed.get("role", "assistant"),
                        content=parsed.get("content"),
                    )
                )
            elif isinstance(parsed, list):
                for entry in parsed:
                    if isinstance(entry, dict) and "role" in entry:
                        messages.append(
                            Message(
                                role=entry.get("role", "assistant"),
                                content=entry.get("content"),
                            )
                        )
            else:
                messages.append(Message(role="assistant", content=completion))
        except (json.JSONDecodeError, TypeError):
            messages.append(Message(role="assistant", content=completion))


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
