"""StreamingHttpTarget — POST to a streaming (SSE) endpoint and extract structured output.

Generic, vendor-neutral sibling of :class:`~harness_evals.targets.http.HttpTarget`.
It speaks Server-Sent Events (``text/event-stream``): it parses named events,
captures them into ``EvalCase.metadata["sse_events"]`` (all events by default,
or a configured subset), and selects a final output from the stream. Non-streaming
responses fall back to buffered JSON/text parsing, matching ``HttpTarget`` semantics.

This target makes no assumptions about any specific product's request or response
shape — request bodies are built from ``body_template`` with ``{{input}}`` /
``{{input.foo}}`` placeholders, and all output extraction uses the shared
``extract_path`` (JSONPath) utility.
"""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
import time
from dataclasses import dataclass, field
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.golden import Golden
from harness_evals.core.types import ToolCall
from harness_evals.errors import TargetInvocationError
from harness_evals.plugins import register_target
from harness_evals.targets.auth import AuthConfig, NoAuth
from harness_evals.targets.base import BaseTarget
from harness_evals.targets.templating import render_headers, render_request_body
from harness_evals.targets.trajectory import (
    normalize_trajectory_fields,
    reconstruct_stream_messages,
    synthesize_messages,
)
from harness_evals.utils.path import extract_path

logger = logging.getLogger(__name__)


@register_target("streaming_http")
@dataclass
class StreamingHttpTarget(BaseTarget):
    """POST to a streaming (SSE) endpoint and map the response to an EvalCase.

    Mirrors :class:`HttpTarget`'s configuration surface and adds generic
    Server-Sent Events handling. The agent's internals are opaque — this grades
    the *shipped system* end-to-end. All JSONPath extraction uses the shared
    ``extract_path`` utility (backed by ``jsonpath-ng``).

    Streaming-specific fields:
        stream: When ``True`` (default), ``text/event-stream`` responses are
            parsed as SSE. When ``False``, always use buffered parsing.
        capture_events: SSE event names to capture into
            ``metadata["sse_events"]`` as ``{event_name: [payloads...]}`` so
            metrics can evaluate across multiple events. Default (unset/``None``)
            captures *all* events. Provide an explicit list to capture only those
            events; an explicit empty list (``[]``) captures nothing.
        output_event: Which event carries the primary ``EvalCase.output``. When
            set, the last payload of that event is used (then ``output_path``
            within it if it is a dict/list). When unset, output is auto-selected:
            ``output_path`` is applied to the *last JSON ``data`` payload from
            which it resolves*, scanning backward so trailing envelope/telemetry
            events (e.g. ``model_usage``, ``done``, ``stream_metadata``) are
            skipped rather than mistaken for the answer. If no payload resolves,
            output falls back to the accumulated text of ``data`` lines (token
            streams); if there are structured payloads but none match, output is
            empty and a warning is logged — set ``output_event`` for such streams.
            Independent of ``capture_events``: all other events remain available
            to metrics via ``sse_events``.
        tool_calls_event: Optional event name (or list of names) that narrows
            where ``tool_calls_path`` is applied. When unset, trajectory
            reconstruction keeps the existing behavior and applies
            ``tool_calls_path`` to every JSON event.
        tool_results_event/tool_results_path: Optional explicit result selector
            for streams that emit tool calls and tool results in different
            event shapes. Results become ``tool`` messages in the reconstructed
            trajectory.

    Trajectory: unless the agent reports a consolidated trajectory via
    ``messages_path`` (which stays authoritative), ``EvalCase.messages`` is
    rebuilt from the stream in event order — interleaving assistant text
    (``output_path``), tool calls (``tool_calls_path``), and tool results
    (``tool_results_path`` or result-shaped ``tool_calls_path`` entries) as they
    were emitted. Streams that carry no such structure fall back to a plain
    ``[user, assistant]`` envelope. Raw events are still captured to
    ``sse_events`` regardless.
    """

    url: str
    method: str = "POST"
    auth: AuthConfig = field(default_factory=NoAuth)
    headers: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 60.0
    verify_tls: bool = True

    body_template: dict | None = None

    retries: int = 2
    backoff_s: float = 0.5

    output_path: str = "$.output"
    tool_calls_event: str | list[str] | None = None
    tool_calls_path: str | None = None
    tool_results_event: str | list[str] | None = None
    tool_results_path: str | None = None
    context_path: str | None = None
    messages_path: str | None = None
    token_count_path: str | None = None
    input_tokens_path: str | None = None
    output_tokens_path: str | None = None
    cost_usd_path: str | None = None
    retry_count_path: str | None = None
    confidence_path: str | None = None
    latency_ms_path: str | None = None

    stream: bool = True
    capture_events: list[str] | None = None
    output_event: str | None = None

    _async_client: object | None = field(default=None, init=False, repr=False)

    async def __aenter__(self) -> StreamingHttpTarget:
        if httpx is None:
            raise ImportError("httpx is required for async context manager usage: pip install harness-evals[harness]")
        verify: bool | ssl.SSLContext = self.verify_tls
        if not self.verify_tls:
            logger.warning("TLS verification disabled for %s — do not use in production", self.url)
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            verify = ctx
        self._async_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout_s),
            verify=verify,
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._async_client is not None:
            await self._async_client.aclose()  # type: ignore[union-attr]
            self._async_client = None

    async def ainvoke(self, golden: Golden) -> EvalCase:
        body = self._build_request_body(golden)
        headers = render_headers(self.headers, golden)
        if self._async_client is not None:
            raw_body, content_type, latency_ms, error = await self._execute_async(body, headers)
        else:
            logger.debug("StreamingHttpTarget used without context manager — falling back to sync thread pool")
            raw_body, content_type, latency_ms, error = await asyncio.to_thread(
                self._execute_with_retries, body, headers
            )

        if error is not None:
            raise TargetInvocationError(
                f"StreamingHttpTarget invocation failed: {error}",
                latency_ms=latency_ms,
            )

        output, kwargs, metadata_extra, extract_source = self._process_response(raw_body, content_type, golden.input)
        extracted_context = kwargs.pop("context", None)
        # A reported trajectory (messages_path) is authoritative; otherwise
        # assemble a best-effort trace from the input, any captured tool calls,
        # and the output so agent/trajectory metrics have something to grade.
        if kwargs.get("messages") is None:
            kwargs["messages"] = synthesize_messages(golden.input, output, kwargs.get("tool_calls"))
        if self.latency_ms_path is None:
            kwargs["latency_ms"] = latency_ms
        else:
            extracted_latency = (
                extract_path(extract_source, self.latency_ms_path) if extract_source is not None else None
            )
            kwargs["latency_ms"] = float(extracted_latency) if extracted_latency is not None else latency_ms

        eval_case = EvalCase.from_golden(golden, output=output, metadata_extra=metadata_extra, **kwargs)
        if extracted_context is not None:
            eval_case.context = extracted_context
        return eval_case

    async def _execute_async(self, body: bytes, user_headers: dict[str, str]) -> tuple[object, str, float, str | None]:
        """Async streaming HTTP call using httpx with retries. Returns (raw_text, content_type, latency_ms, error)."""
        client = self._async_client
        assert client is not None

        headers = {"Content-Type": "application/json", "Accept": "text/event-stream", **user_headers}
        params: dict[str, str] = {}
        self.auth.apply(headers, params)

        url = self.url
        if params:
            sep = "&" if "?" in url else "?"
            url = url + sep + urlencode(params)

        last_error: str | None = None
        last_latency_ms = 0.0
        attempts = 1 + self.retries

        for attempt in range(attempts):
            if attempt > 0:
                await asyncio.sleep(self.backoff_s * (2 ** (attempt - 1)))

            t0 = time.perf_counter()
            try:
                async with client.stream(  # type: ignore[union-attr]
                    self.method,
                    url,
                    content=body,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").lower()
                    if self.stream and "text/event-stream" in content_type:
                        lines: list[str] = []
                        async for line in response.aiter_lines():
                            lines.append(line)
                        raw = "\n".join(lines)
                    else:
                        chunks: list[str] = []
                        async for chunk in response.aiter_text():
                            chunks.append(chunk)
                        raw = "".join(chunks)
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    return raw, content_type, elapsed_ms, None
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                last_latency_ms = elapsed_ms
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("StreamingHttpTarget attempt %d/%d failed: %s", attempt + 1, attempts, last_error)

        return None, "", last_latency_ms, last_error

    def _execute_with_retries(self, body: bytes, user_headers: dict[str, str]) -> tuple[object, str, float, str | None]:
        """Synchronous HTTP call with retry logic. Returns (raw_text, content_type, latency_ms, error)."""
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream", **user_headers}
        params: dict[str, str] = {}
        self.auth.apply(headers, params)

        url = self.url
        if params:
            sep = "&" if "?" in url else "?"
            url = url + sep + urlencode(params)

        ssl_ctx: ssl.SSLContext | None = None
        if not self.verify_tls:
            logger.warning("TLS verification disabled for %s — do not use in production", self.url)
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE

        last_error: str | None = None
        last_attempt_latency_ms = 0.0
        attempts = 1 + self.retries

        for attempt in range(attempts):
            if attempt > 0:
                time.sleep(self.backoff_s * (2 ** (attempt - 1)))

            request = Request(url, data=body, headers=headers, method=self.method)
            t0 = time.perf_counter()
            try:
                with urlopen(request, timeout=self.timeout_s, context=ssl_ctx) as response:
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    raw = response.read().decode("utf-8")
                    content_type = _get_content_type(response)
                    return raw, content_type, elapsed_ms, None
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                last_attempt_latency_ms = elapsed_ms
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("StreamingHttpTarget attempt %d/%d failed: %s", attempt + 1, attempts, last_error)

        return None, "", last_attempt_latency_ms, last_error

    def _build_request_body(self, golden: Golden) -> bytes:
        payload = render_request_body(self.body_template, golden)
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def _process_response(
        self, raw: object, content_type: str, input_value: object
    ) -> tuple[object, dict, dict | None, object]:
        """Turn a raw response body into (output, optional_field_kwargs, metadata_extra, extract_source).

        ``extract_source`` is the object that ``latency_ms_path`` (and, for SSE,
        the optional-field paths) are evaluated against. ``input_value`` seeds
        the leading user turn when a trajectory is reconstructed from the stream.
        """
        if raw is None:
            return "", {}, None, None

        is_sse = self.stream and "text/event-stream" in content_type
        if not is_sse:
            parsed = _parse_response(str(raw), content_type)
            output = self._extract_output(parsed, content_type)
            kwargs = self._extract_optional_fields(parsed)
            return output, kwargs, None, parsed

        events = _parse_sse(str(raw))
        decoded = [(name, _decode(data)) for name, data in events]
        metadata_extra = self._capture(decoded)
        final_payload = self._final_payload(decoded)
        output = self._final_output(decoded, final_payload)
        kwargs = self._extract_optional_fields(final_payload) if final_payload is not None else {}
        stream_tool_calls = self._extract_stream_tool_calls(decoded)
        if stream_tool_calls:
            kwargs["tool_calls"] = stream_tool_calls
        # The answer payload rarely carries usage/telemetry: streaming agents
        # emit token counts in a separate trailing event (``model_usage``,
        # ``done``) that ``_final_payload`` deliberately skips when selecting the
        # answer. Recover any numeric field the answer payload didn't provide by
        # scanning the whole stream (latest event wins).
        self._recover_stream_fields(kwargs, decoded, final_payload)
        # Rebuild an ordered trajectory from the stream (interleaved assistant
        # text / tool calls / tool results) unless the agent already reported a
        # consolidated trajectory via messages_path — that stays authoritative.
        if kwargs.get("messages") is None:
            reconstructed = reconstruct_stream_messages(
                decoded,
                input_value,
                output_path=self.output_path,
                tool_calls_event=self.tool_calls_event,
                tool_calls_path=self.tool_calls_path,
                tool_results_event=self.tool_results_event,
                tool_results_path=self.tool_results_path,
            )
            if reconstructed is not None:
                kwargs["messages"] = reconstructed
        return output, kwargs, metadata_extra, final_payload

    def _capture(self, decoded: list[tuple[str, object]]) -> dict | None:
        """Capture SSE events into ``{"sse_events": {event_name: [payloads...]}}``.

        Default (``capture_events`` unset/``None``) captures *all* events so
        metrics can evaluate across the full stream. An explicit list captures
        only those event names; an explicit empty list captures nothing.
        """
        if self.capture_events is None:
            wanted: set[str] | None = None  # capture everything
        else:
            wanted = set(self.capture_events)
        captured: dict[str, list] = {}
        for name, payload in decoded:
            if wanted is None or name in wanted:
                captured.setdefault(name, []).append(payload)
        if not captured:
            return None
        return {"sse_events": captured}

    def _final_payload(self, decoded: list[tuple[str, object]]) -> object | None:
        """The JSON payload used for output + optional-field extraction.

        With ``output_event`` set, this is the last payload of that event.
        Otherwise it is the last JSON payload from which ``output_path`` resolves
        — scanning backward so trailing envelope/telemetry events (``model_usage``,
        ``done``, ``stream_metadata``, …) don't shadow the real answer. If nothing
        resolves, falls back to the last JSON payload overall (so optional-field
        paths still have a source), or ``None`` if there are no JSON payloads.
        """
        if self.output_event is not None:
            payloads = [p for name, p in decoded if name == self.output_event]
            if payloads and isinstance(payloads[-1], (dict, list)):
                return payloads[-1]
            return None
        json_payloads = [p for _name, p in decoded if isinstance(p, (dict, list))]
        if not json_payloads:
            return None
        for payload in reversed(json_payloads):
            if extract_path(payload, self.output_path) is not None:
                return payload
        return json_payloads[-1]

    def _final_output(self, decoded: list[tuple[str, object]], final_payload: object | None) -> str | dict | list:
        if self.output_event is not None:
            payloads = [p for name, p in decoded if name == self.output_event]
            if not payloads:
                return ""
            last = payloads[-1]
            if isinstance(last, (dict, list)):
                result = extract_path(last, self.output_path)
                return result if result is not None else ""
            return last

        if isinstance(final_payload, (dict, list)):
            result = extract_path(final_payload, self.output_path)
            if result is not None:
                return result
            # There were structured payloads but none carried output_path — the
            # answer isn't auto-locatable. Fail visibly (empty) rather than
            # grading against a trailing envelope/telemetry blob.
            logger.warning(
                "StreamingHttpTarget: output_event is not set and no SSE event "
                "matched output_path %r; EvalCase.output is empty. Set "
                "output_event to select the event that carries the answer.",
                self.output_path,
            )
            return ""

        # No structured payloads at all → token/text stream: accumulate text.
        texts = [p for _name, p in decoded if isinstance(p, str)]
        return "".join(texts)

    def _extract_output(self, response_body: object, content_type: str) -> str | dict | list:
        if self.output_path == "$" and "text/" in content_type:
            return str(response_body) if not isinstance(response_body, str) else response_body

        if response_body is None:
            return ""

        result = extract_path(response_body, self.output_path)
        return result if result is not None else ""

    def _extract_optional_fields(self, response_body: object) -> dict:
        kwargs: dict = {}
        if response_body is None:
            return kwargs

        _extract_field(kwargs, response_body, "tool_calls", self.tool_calls_path)
        _extract_field(kwargs, response_body, "context", self.context_path)
        _extract_field(kwargs, response_body, "messages", self.messages_path)
        _extract_float(kwargs, response_body, "token_count", self.token_count_path, int)
        _extract_float(kwargs, response_body, "input_tokens", self.input_tokens_path, int)
        _extract_float(kwargs, response_body, "output_tokens", self.output_tokens_path, int)
        _extract_float(kwargs, response_body, "cost_usd", self.cost_usd_path, float)
        _extract_float(kwargs, response_body, "retry_count", self.retry_count_path, int)
        _extract_float(kwargs, response_body, "confidence", self.confidence_path, float)

        normalize_trajectory_fields(kwargs, self.messages_path)
        return kwargs

    def _recover_stream_fields(self, kwargs: dict, decoded: list[tuple[str, object]], final_payload: object) -> None:
        """Fill numeric fields from any stream event when the answer payload lacked them.

        Token counts, cost, and similar telemetry usually arrive in a trailing
        envelope event (e.g. ``model_usage``) separate from the answer event, so
        extracting them only from ``final_payload`` misses them. For each numeric
        path that didn't already resolve, scan all decoded payloads *except*
        ``final_payload`` (already handled) in reverse so the most recent event
        wins. Casting failures are handled/logged by ``_extract_float``.
        """
        numeric_fields = (
            ("token_count", self.token_count_path, int),
            ("input_tokens", self.input_tokens_path, int),
            ("output_tokens", self.output_tokens_path, int),
            ("cost_usd", self.cost_usd_path, float),
            ("retry_count", self.retry_count_path, int),
            ("confidence", self.confidence_path, float),
        )
        pending = [(key, path, cast) for key, path, cast in numeric_fields if path is not None and key not in kwargs]
        if not pending:
            return
        for _name, payload in reversed(decoded):
            if not pending:
                break
            if payload is final_payload or not isinstance(payload, (dict, list)):
                continue
            for key, path, cast in list(pending):
                _extract_float(kwargs, payload, key, path, cast)
                if key in kwargs:
                    pending.remove((key, path, cast))

    def _extract_stream_tool_calls(self, decoded: list[tuple[str, object]]) -> list[ToolCall] | None:
        """Extract top-level tool calls from explicitly configured tool events.

        Historically ``EvalCase.tool_calls`` came from the selected final payload
        while stream-wide tool extraction was used for reconstructed messages.
        Preserve that behavior unless ``tool_calls_event`` is set; then the user
        has explicitly identified which SSE event(s) carry tool calls.
        """
        if self.tool_calls_path is None or self.tool_calls_event is None:
            return None

        tool_calls: list[ToolCall] = []
        for name, payload in decoded:
            if not _event_matches(name, self.tool_calls_event) or not isinstance(payload, (dict, list)):
                continue
            raw_tools = extract_path(payload, self.tool_calls_path)
            if not isinstance(raw_tools, list):
                continue
            for entry in raw_tools:
                if not isinstance(entry, dict) or "name" not in entry or _is_tool_result(entry):
                    continue
                tool_calls.append(ToolCall.from_dict(entry))
        return tool_calls or None


def _parse_sse(raw: str) -> list[tuple[str, str]]:
    """Parse an SSE stream into a list of ``(event_name, data)`` tuples.

    Implements the practical subset of the SSE spec: blank lines dispatch an
    event, ``event:`` sets the type (default ``message``), ``data:`` lines are
    concatenated with newlines, a single leading space after the colon is
    stripped, and lines starting with ``:`` are comments. A trailing event with
    data but no final blank line is still dispatched (servers often omit it).
    """
    events: list[tuple[str, str]] = []
    event_type: str | None = None
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal event_type, data_lines
        if data_lines:
            events.append((event_type or "message", "\n".join(data_lines)))
        event_type = None
        data_lines = []

    for raw_line in raw.splitlines():
        line = raw_line.rstrip("\r")
        if line == "":
            flush()
            continue
        if line.startswith(":"):
            continue
        field_name, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field_name == "event":
            event_type = value
        elif field_name == "data":
            data_lines.append(value)
        # Other fields (id, retry, unknown) are ignored.

    flush()
    return events


def _decode(data: str) -> object:
    """JSON-decode an SSE data payload, or return the raw string as-is."""
    try:
        return json.loads(data)
    except (json.JSONDecodeError, ValueError):
        return data


def _event_matches(event_name: str, selector: str | list[str] | None) -> bool:
    if selector is None:
        return True
    if isinstance(selector, str):
        return event_name == selector
    return event_name in selector


def _is_tool_result(entry: dict) -> bool:
    has_input = entry.get("arguments") is not None or entry.get("input") is not None
    has_output = entry.get("output") is not None or entry.get("result") is not None
    return has_output and not has_input


def _get_content_type(response: object) -> str:
    headers = getattr(response, "headers", None)
    if headers is None:
        return ""
    return str(headers.get("Content-Type", "")).lower()


def _parse_response(raw: str, content_type: str) -> object:
    """Parse response as JSON if possible, otherwise return raw text."""
    if "json" in content_type:
        return json.loads(raw)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def _extract_field(kwargs: dict, body: object, key: str, path: str | None) -> None:
    if path is None:
        return
    val = extract_path(body, path)
    if val is not None:
        kwargs[key] = val


def _extract_float(kwargs: dict, body: object, key: str, path: str | None, cast: type) -> None:
    if path is None:
        return
    val = extract_path(body, path)
    if val is None:
        return
    # The path *was* configured and resolved to a value, so a failed cast means
    # the value is the wrong type (non-numeric string, dict) or out of range
    # (json.loads accepts Infinity/NaN → int(inf) raises OverflowError). Warn so
    # a misconfigured path is distinguishable from "the endpoint reported nothing".
    try:
        kwargs[key] = cast(val)
    except (TypeError, ValueError, OverflowError):
        logger.warning(
            "%s path %r resolved to %r, which is not a valid %s; dropping it",
            key,
            path,
            val,
            cast.__name__,
        )
