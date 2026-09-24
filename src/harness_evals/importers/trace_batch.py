"""Generic multi-trace batching for conversation eval-case importers.

Time windows and session grouping are backend-agnostic: a ``TraceCatalog`` lists
and hydrates traces; grouping keys come from OTel ``gen_ai.conversation.id`` or
an explicit ``session_id``. Langfuse is one catalog implementation, not a
requirement of this module.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol


# Well-known conversation-grouping attributes (OTel GenAI + common vendors).
SESSION_ATTR_KEYS: tuple[str, ...] = (
    "gen_ai.conversation.id",
    "session.id",
    "langfuse.session.id",
)


@dataclass
class SpanTrace:
    """One telemetry trace as a span list plus grouping metadata."""

    spans: list[dict[str, Any]]
    trace_id: str | None = None
    session_id: str | None = None
    start_time: datetime | None = None
    # Optional discovery metadata (module/env/tags/prompt) from the list API —
    # used for stratified sampling before span hydration.
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class TimeWindow:
    from_timestamp: datetime | None = None
    to_timestamp: datetime | None = None


class TraceCatalog(Protocol):
    """List and hydrate traces from any session store (Langfuse, Tempo, files, …)."""

    def list_traces(
        self,
        *,
        from_timestamp: datetime | None = None,
        to_timestamp: datetime | None = None,
        session_id: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
        limit: int = 100,
    ) -> Sequence[SpanTrace]:
        """Return trace stubs. ``spans`` may be empty; ``load_spans`` hydrates them."""
        ...

    def load_spans(self, trace_id: str) -> list[dict[str, Any]]:
        """Return span dicts for one trace, in the OTELEvalCaseSource JSON shape."""
        ...


def session_id_from_spans(spans: Sequence[dict[str, Any]]) -> str | None:
    """Return the first conversation/session id found on a span or its attributes."""
    for span in spans:
        direct = span.get("session_id")
        if direct:
            return str(direct)
        attrs = span.get("attributes") or {}
        for key in SESSION_ATTR_KEYS:
            val = attrs.get(key)
            if val:
                return str(val)
    return None


def trace_id_from_spans(spans: Sequence[dict[str, Any]]) -> str | None:
    for span in spans:
        tid = span.get("trace_id")
        if tid:
            return str(tid)
    return None


def infer_start_time(spans: Sequence[dict[str, Any]]) -> datetime | None:
    """Earliest span start, from unix nano or ISO ``start_timestamp``."""
    earliest: datetime | None = None
    for span in spans:
        dt = _span_start_datetime(span)
        if dt is not None and (earliest is None or dt < earliest):
            earliest = dt
    return earliest


def complete_trace(trace: SpanTrace) -> SpanTrace:
    """Fill session_id / trace_id / start_time from spans when omitted."""
    session_id = trace.session_id or session_id_from_spans(trace.spans)
    trace_id = trace.trace_id or trace_id_from_spans(trace.spans)
    start_time = trace.start_time or infer_start_time(trace.spans)
    return SpanTrace(
        spans=trace.spans,
        trace_id=trace_id,
        session_id=session_id,
        start_time=start_time,
        metadata=trace.metadata,
    )


def traces_from_json(raw: Any) -> list[SpanTrace]:
    """Normalize exported JSON into traces.

    Accepted shapes:
    - list of span dicts (one trace)
    - list of ``{spans, trace_id?, session_id?}`` objects
    - ``{traces: [...]}`` wrapping either of the above
    """
    if isinstance(raw, dict) and "traces" in raw:
        raw = raw["traces"]
    if not isinstance(raw, list):
        raise ValueError("OTEL span JSON must be a list of spans, a list of traces, or {traces: [...]}")
    if not raw:
        return []
    first = raw[0]
    if isinstance(first, dict) and "spans" in first:
        return [complete_trace(_trace_from_obj(obj)) for obj in raw if isinstance(obj, dict)]
    spans = [s for s in raw if isinstance(s, dict)]
    return [complete_trace(SpanTrace(spans=spans))]


def filter_traces(traces: Sequence[SpanTrace], window: TimeWindow) -> list[SpanTrace]:
    """Drop traces that start outside the window. Missing start_time is kept."""
    if window.from_timestamp is None and window.to_timestamp is None:
        return list(traces)
    kept: list[SpanTrace] = []
    for trace in traces:
        start = trace.start_time
        if start is None:
            kept.append(trace)
            continue
        if window.from_timestamp is not None and start < window.from_timestamp:
            continue
        if window.to_timestamp is not None and start >= window.to_timestamp:
            continue
        kept.append(trace)
    return kept


def group_traces(traces: Sequence[SpanTrace], *, group_by: str | None) -> list[list[SpanTrace]]:
    """Group traces for EvalCase construction.

    ``group_by="session_id"`` merges traces that share a session/conversation id.
    Traces with no session id stay one-per-trace so empty keys are not collapsed.
    Any other / empty ``group_by`` is one EvalCase per trace.
    """
    if group_by not in {"session_id", "conversation_id"}:
        return [[t] for t in traces]

    buckets: dict[str, list[SpanTrace]] = {}
    order: list[str] = []
    for trace in traces:
        if trace.session_id:
            key = f"session:{trace.session_id}"
        else:
            key = f"trace:{trace.trace_id or id(trace)}"
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(trace)
    return [buckets[k] for k in order]


def merge_span_groups(group: Sequence[SpanTrace]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Concatenate spans in time order and return grouping metadata."""
    traces = list(group)
    traces.sort(key=lambda t: t.start_time or datetime.min.replace(tzinfo=timezone.utc))
    spans: list[dict[str, Any]] = []
    trace_ids: list[str] = []
    session_id: str | None = None
    for trace in traces:
        session_id = session_id or trace.session_id
        if trace.trace_id:
            trace_ids.append(trace.trace_id)
        annotated = _annotate_session(trace.spans, session_id)
        spans.extend(annotated)
    meta: dict[str, Any] = {}
    if session_id:
        meta["session_id"] = session_id
    if trace_ids:
        meta["trace_ids"] = trace_ids
    return spans, meta


def time_window_from_extra(
    extra: dict[str, Any],
    *,
    now: Callable[[], datetime] | None = None,
) -> TimeWindow:
    """Build a window from ``lookback_days`` / ``from_timestamp`` / ``to_timestamp``."""
    clock = now or (lambda: datetime.now(timezone.utc))
    lookback = _coerce_int(extra.get("lookback_days"))
    from_ts = _coerce_datetime(extra.get("from_timestamp"))
    to_ts = _coerce_datetime(extra.get("to_timestamp"))
    if lookback is not None and lookback < 0:
        raise ValueError("lookback_days must be >= 0")
    if lookback is not None and from_ts is None:
        from_ts = clock() - timedelta(days=lookback)
    return TimeWindow(from_timestamp=from_ts, to_timestamp=to_ts)


def parse_group_by(value: Any) -> str | None:
    if value in (None, "", "trace", "trace_id", "none"):
        return None
    text = str(value)
    if text in {"session_id", "conversation_id"}:
        return text
    raise ValueError("group_by must be session_id, conversation_id, or trace_id")


def parse_tags(value: Any) -> list[str] | None:
    if value is None or value == "":
        return None
    if isinstance(value, list):
        return [str(v) for v in value]
    return [part for part in str(value).split(",") if part]


def parse_limit(value: Any, *, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    parsed = _coerce_int(value)
    if parsed is None:
        return default
    if parsed < 1:
        raise ValueError("limit must be >= 1")
    return parsed


def parse_bool(value: Any, *, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _trace_from_obj(obj: dict[str, Any]) -> SpanTrace:
    spans = obj.get("spans")
    if not isinstance(spans, list):
        raise ValueError("each trace object must include a 'spans' list")
    start = obj.get("start_time")
    start_dt = start if isinstance(start, datetime) else _coerce_datetime(start)
    return SpanTrace(
        spans=[s for s in spans if isinstance(s, dict)],
        trace_id=str(obj["trace_id"]) if obj.get("trace_id") else None,
        session_id=str(obj["session_id"]) if obj.get("session_id") else None,
        start_time=start_dt,
    )


def _annotate_session(spans: list[dict[str, Any]], session_id: str | None) -> list[dict[str, Any]]:
    if not session_id:
        return spans
    annotated: list[dict[str, Any]] = []
    for span in spans:
        clone = dict(span)
        attrs = dict(clone.get("attributes") or {})
        attrs.setdefault("gen_ai.conversation.id", session_id)
        clone["attributes"] = attrs
        annotated.append(clone)
    return annotated


def _span_start_datetime(span: dict[str, Any]) -> datetime | None:
    nano = span.get("start_time_unix_nano")
    if nano is not None:
        try:
            return datetime.fromtimestamp(int(nano) / 1_000_000_000, tz=timezone.utc)
        except (TypeError, ValueError, OSError, OverflowError):
            return None
    return _coerce_datetime(span.get("start_timestamp"))


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"expected an integer, got {value!r}") from None


def _coerce_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"invalid timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed
