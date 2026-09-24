"""Langfuse eval-case importer — hydrate EvalCases from Langfuse traces.

Requires: pip install harness-evals[langfuse]
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime

try:
    from langfuse import Langfuse
except ImportError as _err:
    raise ImportError(
        "LangfuseEvalCaseSource requires the langfuse package. Install with: pip install harness-evals[langfuse]"
    ) from _err

from harness_evals._langfuse_compat import flush_langfuse_client
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.types import Message, ToolCall
from harness_evals.importers.base import BaseEvalCaseSource
from harness_evals.importers.trace_batch import SpanTrace
from harness_evals.plugins import register_eval_case_source
from harness_evals.refs import ResourceRef

# Filter keys that, when present in ref.extra, trigger from_traces() dispatch.
_FILTER_KEYS = {"name", "tags", "user_id", "session_id", "from_timestamp", "to_timestamp", "limit"}


class LangfuseTraceCatalog:
    """``TraceCatalog`` adapter: list Langfuse traces as OTel-shaped span batches.

    Used with :class:`~harness_evals.importers.otel.OTELEvalCaseSource` so
    time-window listing and session merge stay generic on the OTEL importer.
    """

    def __init__(self, client: Langfuse) -> None:
        self._client = client

    def list_traces(
        self,
        *,
        from_timestamp: datetime | None = None,
        to_timestamp: datetime | None = None,
        session_id: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
        environment: str | list[str] | None = None,
        limit: int = 100,
    ) -> list[SpanTrace]:
        kwargs: dict[str, object] = {}
        if name is not None:
            kwargs["name"] = name
        if tags is not None:
            kwargs["tags"] = tags
        if user_id is not None:
            kwargs["user_id"] = user_id
        if session_id is not None:
            kwargs["session_id"] = session_id
        if environment is not None:
            kwargs["environment"] = environment
        if from_timestamp is not None:
            kwargs["from_timestamp"] = from_timestamp
        if to_timestamp is not None:
            kwargs["to_timestamp"] = to_timestamp

        collected: list[SpanTrace] = []
        for page_data in _iter_trace_pages(self._client, limit=limit, **kwargs):
            for trace in page_data:
                tid = getattr(trace, "id", None)
                if not tid:
                    continue
                collected.append(
                    SpanTrace(
                        spans=[],
                        trace_id=str(tid),
                        session_id=_trace_session_id(trace),
                        start_time=getattr(trace, "timestamp", None) or getattr(trace, "start_time", None),
                        metadata=_list_trace_metadata(trace),
                    )
                )
                if len(collected) >= limit:
                    return collected[:limit]
        return collected

    def load_spans(self, trace_id: str) -> list[dict[str, object]]:
        trace = self._client.api.trace.get(trace_id)
        session_id = _trace_session_id(trace)
        observations = self._client.api.observations.get_many(
            trace_id=trace_id,
        )
        obs_list = observations.data if hasattr(observations, "data") else []
        spans = [
            _observation_to_span(obs, trace_id=trace_id, session_id=session_id) for obs in obs_list
        ]
        # Stamp Langfuse trace-level I/O onto every span so the OTEL conversation
        # builder can recover the user prompt when generation observations omit
        # input/output (common for UA runner_v3 traces).
        trace_input = getattr(trace, "input", None)
        trace_output = getattr(trace, "output", None)
        trace_cost = getattr(trace, "total_cost", None)
        if trace_cost is None:
            trace_cost = getattr(trace, "totalCost", None)
        if trace_input is not None or trace_output is not None or trace_cost is not None:
            for i, span in enumerate(spans):
                attrs = span.setdefault("attributes", {})
                if not isinstance(attrs, dict):
                    attrs = {}
                    span["attributes"] = attrs
                if trace_input is not None:
                    attrs.setdefault("langfuse.trace.input", trace_input)
                if trace_output is not None:
                    attrs.setdefault("langfuse.trace.output", trace_output)
                # Stamp trace cost once (first span only) to avoid N× inflation
                # when the OTEL builder sums gen_ai.usage.cost across spans.
                if i == 0 and trace_cost is not None:
                    try:
                        cost_val = float(trace_cost)
                    except (TypeError, ValueError):
                        cost_val = None
                    if cost_val is not None and cost_val > 0:
                        attrs.setdefault("langfuse.trace.total_cost", cost_val)
        if not spans and (trace_input is not None or trace_output is not None):
            # No observations — still emit a root span so prompt/output survive.
            attrs: dict[str, object] = {"langfuse.observation.type": "agent"}
            if session_id:
                attrs["gen_ai.conversation.id"] = session_id
            if trace_input is not None:
                attrs["langfuse.trace.input"] = trace_input
            if trace_output is not None:
                attrs["langfuse.trace.output"] = trace_output
            spans = [
                {
                    "trace_id": trace_id,
                    "span_id": f"{trace_id}:trace-root",
                    "name": "langfuse.trace",
                    "attributes": attrs,
                }
            ]
        return spans

    def list_session_trace_ids(self, session_id: str) -> list[str] | None:
        sessions_api = getattr(self._client.api, "sessions", None)
        getter = getattr(sessions_api, "get", None) if sessions_api is not None else None
        if getter is None:
            return None
        session = getter(session_id)
        traces = getattr(session, "traces", None) or []
        ids: list[str] = []
        for item in traces:
            tid = item if isinstance(item, str) else getattr(item, "id", None)
            if tid:
                ids.append(str(tid))
        return ids


@register_eval_case_source("langfuse")
class LangfuseEvalCaseSource(BaseEvalCaseSource):
    """Fetch EvalCases from Langfuse traces.

    Uses the Langfuse SDK to retrieve trace + observation data and maps each
    trace to an :class:`~harness_evals.core.eval_case.EvalCase` with typed
    ``messages``, ``tool_calls``, and operational fields.

    Sets ``metadata["langfuse_trace_id"]`` so that a ``LangfuseSink`` can
    write scores back to the same trace.

    **Uniform entry point** — ``fetch(ref)``::

        source = LangfuseEvalCaseSource(Langfuse())

        # Single trace:  langfuse://trace-abc-123
        cases = await source.fetch(resolve("langfuse://trace-abc-123"))

        # Multi-trace with filters:
        ref = ResourceRef(source="langfuse", id="", extra={"tags": ["prod"], "limit": 50})
        cases = await source.fetch(ref)

    **Convenience methods** (also available)::

        ec = source.from_trace("trace-abc-123")
        cases = source.from_traces(tags=["prod"], limit=50)
    """

    name = "langfuse"

    def __init__(self, client: Langfuse, *, concurrency: int = 10) -> None:
        self._client = client
        self._CONCURRENCY = concurrency

    async def close(self) -> None:
        """Flush the Langfuse client to prevent data loss."""
        await flush_langfuse_client(self._client)

    # ------------------------------------------------------------------
    # BaseEvalCaseSource ABC
    # ------------------------------------------------------------------

    async def fetch(self, ref: ResourceRef) -> list[EvalCase]:
        """Dispatch to single-trace or multi-trace fetch based on ``ref``.

        - If ``ref.id`` is non-empty and ``ref.extra`` contains no filter
          keys beyond ``limit``, treats ``ref.id`` as a trace ID and calls
          :meth:`from_trace`, returning a single-element list. Note:
          ``limit`` is silently ignored in this case since a trace ID
          is deterministic.
        - If ``ref.extra`` contains any multi-trace filter key (name, tags,
          user_id, session_id, from_timestamp, to_timestamp), calls
          :meth:`from_traces` with those kwargs (including ``limit``).
        - If ``ref.id`` is empty and no filter keys are set, raises
          ``ValueError``.

        The synchronous Langfuse SDK calls are offloaded to a thread so
        they don't block the event loop.
        """
        filter_kwargs = {k: v for k, v in ref.extra.items() if k in _FILTER_KEYS}
        has_multi_trace_filters = bool(filter_kwargs.keys() - {"limit"})
        if ref.id and not has_multi_trace_filters:
            return [await asyncio.to_thread(self.from_trace, ref.id)]
        if filter_kwargs:
            return await self._fetch_traces_concurrent(**filter_kwargs)  # type: ignore[arg-type]
        if not ref.id:
            raise ValueError(
                "LangfuseEvalCaseSource.fetch() requires either a trace ID in ref.id "
                "or filter keys (name, tags, user_id, session_id, from_timestamp, to_timestamp) in ref.extra"
            )
        return [await asyncio.to_thread(self.from_trace, ref.id)]

    # ------------------------------------------------------------------
    # Concurrent fetch
    # ------------------------------------------------------------------

    async def _fetch_traces_concurrent(self, **kwargs: object) -> list[EvalCase]:
        """List traces with filters, then hydrate each concurrently via threads."""
        trace_ids = await asyncio.to_thread(self._list_trace_ids, **kwargs)  # type: ignore[arg-type]
        sem = asyncio.Semaphore(self._CONCURRENCY)

        async def _hydrate(trace_id: str) -> EvalCase:
            async with sem:
                return await asyncio.to_thread(self.from_trace, trace_id)

        return list(await asyncio.gather(*[_hydrate(tid) for tid in trace_ids]))

    def _list_trace_ids(
        self,
        *,
        name: str | None = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        from_timestamp: datetime | None = None,
        to_timestamp: datetime | None = None,
        limit: int = 100,
    ) -> list[str]:
        """Return trace IDs matching filters (paginated)."""
        kwargs: dict[str, object] = {}
        if name is not None:
            kwargs["name"] = name
        if tags is not None:
            kwargs["tags"] = tags
        if user_id is not None:
            kwargs["user_id"] = user_id
        if session_id is not None:
            kwargs["session_id"] = session_id
        if from_timestamp is not None:
            kwargs["from_timestamp"] = from_timestamp
        if to_timestamp is not None:
            kwargs["to_timestamp"] = to_timestamp

        collected: list[str] = []
        for page_data in _iter_trace_pages(self._client, limit=limit, **kwargs):
            for trace in page_data:
                tid = getattr(trace, "id", None)
                if tid:
                    collected.append(str(tid))
                if len(collected) >= limit:
                    return collected[:limit]
        return collected

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def from_trace(self, trace_id: str) -> EvalCase:
        """Fetch a single Langfuse trace and convert it to an EvalCase.

        Maps:
          - trace.input / trace.output -> input / output
          - generation observations -> messages (with tool_calls embedded)
          - tool observations -> tool_calls
          - aggregated usage -> token_count, cost_usd
          - trace duration -> latency_ms
          - trace.tags -> tags
          - trace.metadata -> metadata (includes langfuse_trace_id)
        """
        trace = self._client.api.trace.get(trace_id)

        observations = self._client.api.observations.get_many(
            trace_id=trace_id,
        )
        obs_list = observations.data if hasattr(observations, "data") else []

        messages: list[Message] = []
        tool_calls: list[ToolCall] = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0

        for obs in obs_list:
            obs_type = getattr(obs, "type", None)
            if isinstance(obs_type, str):
                obs_type = obs_type.lower()

            if obs_type == "generation":
                self._process_generation(obs, messages, tool_calls)
                usage = getattr(obs, "usage_details", None) or {}
                total_input_tokens += usage.get("input", 0) or 0
                total_output_tokens += usage.get("output", 0) or 0
                total_cost += getattr(obs, "total_cost", 0) or 0

            elif obs_type == "tool":
                tc = ToolCall(
                    name=getattr(obs, "name", None) or "unknown_tool",
                    input=_to_dict_or_none(getattr(obs, "input", None)),
                    output=_to_str_or_dict_or_none(getattr(obs, "output", None)),
                )
                tool_calls.append(tc)

        trace_input = getattr(trace, "input", None) or ""
        trace_output = getattr(trace, "output", None) or ""

        latency_ms: float | None = None
        start_time = getattr(trace, "start_time", None)
        end_time = getattr(trace, "end_time", None)
        if start_time and end_time:
            latency_ms = (end_time - start_time).total_seconds() * 1000

        total_tokens = total_input_tokens + total_output_tokens

        trace_tags = getattr(trace, "tags", None)
        tags: dict[str, str] | None = None
        if trace_tags and isinstance(trace_tags, list):
            tags = {t: "true" for t in trace_tags}

        trace_meta = getattr(trace, "metadata", None) or {}
        if isinstance(trace_meta, dict):
            metadata = {**trace_meta, "langfuse_trace_id": trace_id}
        else:
            metadata = {"langfuse_trace_id": trace_id}

        if start_time is not None:
            metadata["langfuse_trace_start_time"] = start_time.isoformat()

        return EvalCase(
            input=trace_input,
            output=trace_output,
            messages=messages or None,
            tool_calls=tool_calls or None,
            latency_ms=latency_ms,
            token_count=total_tokens if total_tokens > 0 else None,
            cost_usd=total_cost if total_cost > 0 else None,
            tags=tags,
            metadata=metadata,
        )

    def from_traces(
        self,
        *,
        name: str | None = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        from_timestamp: datetime | None = None,
        to_timestamp: datetime | None = None,
        limit: int = 100,
    ) -> list[EvalCase]:
        """Fetch multiple Langfuse traces matching filters and convert each to an EvalCase.

        Uses page-based pagination to collect up to ``limit`` traces.

        Args:
            name: Filter by trace name.
            tags: Filter by tags (traces must have all listed tags).
            user_id: Filter by user ID.
            session_id: Filter by session ID.
            from_timestamp: Only include traces started at or after this time.
            to_timestamp: Only include traces started before this time.
            limit: Maximum number of traces to fetch.

        Returns:
            List of EvalCases, one per trace, ordered as returned by the API.
        """
        kwargs: dict[str, object] = {}
        if name is not None:
            kwargs["name"] = name
        if tags is not None:
            kwargs["tags"] = tags
        if user_id is not None:
            kwargs["user_id"] = user_id
        if session_id is not None:
            kwargs["session_id"] = session_id
        if from_timestamp is not None:
            kwargs["from_timestamp"] = from_timestamp
        if to_timestamp is not None:
            kwargs["to_timestamp"] = to_timestamp

        results: list[EvalCase] = []
        for page_data in _iter_trace_pages(self._client, limit=limit, **kwargs):
            for trace in page_data:
                trace_id = getattr(trace, "id", None)
                if trace_id:
                    results.append(self.from_trace(str(trace_id)))
                if len(results) >= limit:
                    return results
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _process_generation(
        obs: object,
        messages: list[Message],
        tool_calls: list[ToolCall],
    ) -> None:
        """Extract messages and tool calls from a generation observation."""
        obs_input = getattr(obs, "input", None)
        obs_output = getattr(obs, "output", None)

        if isinstance(obs_input, list):
            for entry in obs_input:
                if isinstance(entry, dict) and "role" in entry:
                    msg_tool_calls = None
                    if "tool_calls" in entry and entry["tool_calls"]:
                        msg_tool_calls = [
                            ToolCall(
                                name=tc.get("function", {}).get("name", tc.get("name", "")),
                                input=_tool_call_input(tc),
                            )
                            for tc in entry["tool_calls"]
                            if isinstance(tc, dict)
                        ]
                    messages.append(
                        Message(
                            role=entry.get("role", "unknown"),
                            content=entry.get("content"),
                            tool_calls=msg_tool_calls,
                        )
                    )

        if isinstance(obs_output, dict):
            role = obs_output.get("role", "assistant")
            content = obs_output.get("content")
            msg_tool_calls = None
            if "tool_calls" in obs_output and obs_output["tool_calls"]:
                msg_tool_calls = []
                for tc in obs_output["tool_calls"]:
                    if isinstance(tc, dict):
                        tc_obj = ToolCall(
                            name=tc.get("function", {}).get("name", tc.get("name", "")),
                            input=_tool_call_input(tc),
                        )
                        msg_tool_calls.append(tc_obj)
                        tool_calls.append(tc_obj)
            messages.append(Message(role=role, content=content, tool_calls=msg_tool_calls or None))
        elif isinstance(obs_output, str):
            messages.append(Message(role="assistant", content=obs_output))


def _tool_call_input(tc: dict) -> dict | None:
    """Normalize a tool call's arguments to a dict.

    OpenAI-style traces put the arguments under ``function.arguments`` as a
    JSON *string*; other shapes use a dict under ``input``. Returning a parsed
    dict for both keeps ``ToolCall.input`` a consistent type so downstream
    tool-argument metrics can compare it against expected dicts.
    """
    raw = tc.get("function", {}).get("arguments") if "function" in tc else tc.get("input")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return None
    return raw if isinstance(raw, dict) else None


def _to_dict_or_none(val: object) -> dict | None:
    if isinstance(val, dict):
        return val
    return None


def _to_str_or_dict_or_none(val: object) -> str | dict | None:
    if isinstance(val, (str, dict)):
        return val
    return None


def _iter_trace_pages(client: Langfuse, *, limit: int, **filters: object):
    """Yield ``trace.list`` pages (1-indexed ``page``) until ``limit`` or exhaustion.

    The Langfuse SDK paginates with ``page`` / ``limit``, not cursors. Stop when a
    page is empty, shorter than the requested page size, or ``meta.page >=
    meta.total_pages`` when that metadata is present.
    """
    page_size = min(max(limit, 1), 100)
    page_num = 1
    seen = 0
    while seen < limit:
        page = client.api.trace.list(limit=min(page_size, limit - seen), page=page_num, **filters)
        page_data = list(page.data) if hasattr(page, "data") else []
        if not page_data:
            break
        yield page_data
        seen += len(page_data)
        page_meta = getattr(page, "meta", None)
        total_pages = getattr(page_meta, "total_pages", None) if page_meta else None
        current = getattr(page_meta, "page", None) if page_meta else None
        if total_pages is not None and current is not None:
            if current >= total_pages:
                break
        elif len(page_data) < page_size:
            break
        page_num += 1


def _trace_session_id(trace: object) -> str | None:
    sid = getattr(trace, "session_id", None) or getattr(trace, "sessionId", None)
    return str(sid) if sid else None


def _list_trace_metadata(trace: object) -> dict[str, object]:
    """Capture list-API fields needed for module/env stratified sampling."""
    meta: dict[str, object] = {}
    tags = getattr(trace, "tags", None)
    if tags is not None:
        meta["tags"] = list(tags) if isinstance(tags, (list, tuple)) else tags
    raw_md = getattr(trace, "metadata", None)
    if isinstance(raw_md, dict):
        meta["metadata"] = raw_md
    env = getattr(trace, "environment", None)
    if env:
        meta["environment"] = str(env)
    for key in ("input", "output", "name", "total_cost", "totalCost"):
        val = getattr(trace, key, None)
        if val is not None:
            meta[key if key != "totalCost" else "total_cost"] = val
    return meta


def _dt_to_nano(value: object) -> int | None:
    if not isinstance(value, datetime):
        return None
    return int(value.timestamp() * 1_000_000_000)


def _observation_to_span(obs: object, *, trace_id: str, session_id: str | None) -> dict[str, object]:
    """Map a Langfuse observation onto the OTEL span dict the conversation builder expects."""
    obs_type = getattr(obs, "type", None)
    if isinstance(obs_type, str):
        obs_type = obs_type.lower()
    name = getattr(obs, "name", None) or "observation"
    start = _dt_to_nano(getattr(obs, "start_time", None))
    end = _dt_to_nano(getattr(obs, "end_time", None))
    attrs: dict[str, object] = {}
    if session_id:
        attrs["gen_ai.conversation.id"] = session_id

    obs_input = getattr(obs, "input", None)
    obs_output = getattr(obs, "output", None)

    if obs_type == "generation":
        attrs["langfuse.observation.type"] = "generation"
        if obs_input is not None:
            attrs["gen_ai.input_messages"] = obs_input
        if obs_output is not None:
            if isinstance(obs_output, list):
                attrs["gen_ai.output_messages"] = obs_output
            elif isinstance(obs_output, dict):
                attrs["gen_ai.output_messages"] = [obs_output]
            else:
                attrs["gen_ai.output_messages"] = [{"role": "assistant", "content": obs_output}]
    elif obs_type == "tool" or (isinstance(name, str) and name.lower().startswith("tool:")):
        # Only real tool observations — do NOT map generic Langfuse SPAN here.
        # UA traces emit dozens of SPAN (mcp/rest/provider_call); treating them as
        # execute_tool drops AGENT/GENERATION recovery of langfuse.trace.input.
        attrs["gen_ai.operation.name"] = "execute_tool"
        tool_name = name[5:] if isinstance(name, str) and name.lower().startswith("tool:") else name
        attrs["gen_ai.tool.name"] = tool_name
        if obs_input is not None:
            attrs["gen_ai.tool.call.arguments"] = json.dumps(obs_input) if not isinstance(obs_input, str) else obs_input
        if obs_output is not None:
            attrs["gen_ai.tool.call.result"] = (
                json.dumps(obs_output) if isinstance(obs_output, dict) else obs_output
            )
    elif obs_type == "agent":
        attrs["langfuse.observation.type"] = "agent"
        attrs["gen_ai.operation.name"] = "invoke_agent"
        # Agent I/O is often {prompt, user_message} — keep on langfuse keys so
        # _extract_user_input_from_span can read it; do not force input_messages.
        if obs_input is not None:
            if isinstance(obs_input, list):
                attrs["gen_ai.input_messages"] = obs_input
            else:
                attrs["langfuse.observation.input"] = obs_input
        if obs_output is not None:
            if isinstance(obs_output, list):
                attrs["gen_ai.output_messages"] = obs_output
            else:
                attrs["langfuse.observation.output"] = obs_output
    else:
        # Generic SPAN / unknown — preserve I/O without forcing tool classification.
        attrs["langfuse.observation.type"] = obs_type or "span"
        if isinstance(name, str) and name.lower().startswith("llm_turn"):
            attrs["langfuse.observation.type"] = "generation"
        if obs_input is not None:
            attrs["langfuse.observation.input"] = obs_input
        if obs_output is not None:
            attrs["langfuse.observation.output"] = obs_output

    span: dict[str, object] = {
        "name": name,
        "span_id": getattr(obs, "id", None),
        "trace_id": trace_id,
        "parent_span_id": getattr(obs, "parent_observation_id", None),
        "attributes": attrs,
    }
    if start is not None:
        span["start_time_unix_nano"] = start
    if end is not None:
        span["end_time_unix_nano"] = end

    # Usage / cost — let the OTEL EvalCase builder aggregate session totals.
    usage = getattr(obs, "usage_details", None) or {}
    if isinstance(usage, dict):
        inp = usage.get("input")
        out = usage.get("output")
        total = usage.get("total")
        if inp is not None:
            attrs["gen_ai.usage.input_tokens"] = inp
        if out is not None:
            attrs["gen_ai.usage.output_tokens"] = out
        if total is not None:
            attrs["gen_ai.usage.total_tokens"] = total
    for cost_attr in ("total_cost", "calculated_total_cost"):
        raw_cost = getattr(obs, cost_attr, None)
        if raw_cost is None:
            continue
        try:
            cost_val = float(raw_cost)
        except (TypeError, ValueError):
            continue
        if cost_val > 0:
            attrs["gen_ai.usage.cost"] = cost_val
            break

    return span
