"""Tests for the generic StreamingHttpTarget (SSE) adapter."""

from __future__ import annotations

import logging
import time
from urllib.error import URLError
from urllib.request import Request

import pytest

from harness_evals import plugins
from harness_evals.core.golden import Golden
from harness_evals.targets import BearerAuth, StreamingHttpTarget

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeHTTPResponse:
    def __init__(self, body: str, content_type: str = "text/event-stream", status: int = 200) -> None:
        self._body = body.encode("utf-8")
        self.headers = {"Content-Type": content_type}
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def _sse(*blocks: str) -> str:
    """Join SSE event blocks with the required blank-line separators."""
    return "\n\n".join(blocks) + "\n\n"


def _patch_response(monkeypatch: pytest.MonkeyPatch, body: str, content_type: str = "text/event-stream") -> list[Request]:
    from harness_evals.targets import streaming_http as target_mod

    captured: list[Request] = []

    def fake_urlopen(request: Request, timeout: float, context=None) -> FakeHTTPResponse:
        captured.append(request)
        return FakeHTTPResponse(body, content_type=content_type)

    monkeypatch.setattr(target_mod, "urlopen", fake_urlopen)
    return captured


# ---------------------------------------------------------------------------
# Buffered (non-SSE) fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_buffered_json_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_response(
        monkeypatch,
        '{"output": "agent response", "tokens": 50}',
        content_type="application/json",
    )

    target = StreamingHttpTarget(url="http://localhost:8080/run", token_count_path="$.tokens")
    result = await target.ainvoke(Golden(input="hello", expected="world"))

    assert result.output == "agent response"
    assert result.token_count == 50
    assert result.expected == "world"
    assert result.meta("sse_events") is None

    body = captured[0].data.decode()
    assert body == '{"input": "hello"}'


@pytest.mark.unit
async def test_stream_disabled_uses_buffered_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    # Even with an event-stream content-type, stream=False forces buffered parsing.
    _patch_response(monkeypatch, '{"output": "buffered"}', content_type="text/event-stream")

    target = StreamingHttpTarget(url="http://localhost:8080/run", stream=False)
    result = await target.ainvoke(Golden(input="hi"))

    assert result.output == "buffered"
    assert result.meta("sse_events") is None


# ---------------------------------------------------------------------------
# SSE parsing
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_sse_output_from_last_json_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _sse(
        "event: token\ndata: {\"chunk\": \"partial\"}",
        "event: message\ndata: {\"output\": \"final answer\"}",
    )
    _patch_response(monkeypatch, body)

    target = StreamingHttpTarget(url="http://localhost:8080/run")
    result = await target.ainvoke(Golden(input="q"))

    assert result.output == "final answer"


@pytest.mark.unit
async def test_sse_output_via_output_event(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _sse(
        "event: message\ndata: {\"output\": \"ignored\"}",
        "event: final\ndata: {\"output\": \"chosen\"}",
    )
    _patch_response(monkeypatch, body)

    target = StreamingHttpTarget(url="http://localhost:8080/run", output_event="final")
    result = await target.ainvoke(Golden(input="q"))

    assert result.output == "chosen"


@pytest.mark.unit
async def test_sse_text_token_stream_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _sse(
        "event: message\ndata: Hello",
        "event: message\ndata:  world",
    )
    _patch_response(monkeypatch, body)

    target = StreamingHttpTarget(url="http://localhost:8080/run")
    result = await target.ainvoke(Golden(input="q"))

    # "data: Hello" -> "Hello"; "data:  world" -> " world" (one leading space stripped)
    assert result.output == "Hello world"


@pytest.mark.unit
async def test_sse_multiline_data_is_joined(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _sse("event: note\ndata: line1\ndata: line2")
    _patch_response(monkeypatch, body)

    target = StreamingHttpTarget(url="http://localhost:8080/run", capture_events=["note"])
    result = await target.ainvoke(Golden(input="q"))

    assert result.meta("sse_events") == {"note": ["line1\nline2"]}


@pytest.mark.unit
async def test_sse_default_event_name_is_message(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _sse("data: {\"output\": \"hi\"}")
    _patch_response(monkeypatch, body)

    target = StreamingHttpTarget(url="http://localhost:8080/run", capture_events=["message"])
    result = await target.ainvoke(Golden(input="q"))

    assert result.output == "hi"
    assert result.meta("sse_events") == {"message": [{"output": "hi"}]}


@pytest.mark.unit
async def test_sse_output_skips_trailing_envelope_events(monkeypatch: pytest.MonkeyPatch) -> None:
    # Real-world shape: the answer is followed by telemetry/terminator events
    # (model_usage, done) whose payloads don't carry output_path. The backward
    # scan must pick the message, not the trailing envelopes.
    body = _sse(
        "event: message\ndata: {\"output\": \"real answer\"}",
        "event: model_usage\ndata: {\"input_tokens\": 17, \"output_tokens\": 4299}",
        "event: done\ndata: {}",
    )
    _patch_response(monkeypatch, body)

    target = StreamingHttpTarget(url="http://localhost:8080/run")
    result = await target.ainvoke(Golden(input="q"))

    assert result.output == "real answer"


@pytest.mark.unit
async def test_sse_output_empty_when_no_event_matches_output_path(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # No event carries $.output (only envelope/telemetry payloads). Rather than
    # grading against a usage/terminator blob, output is empty and a warning is
    # logged so the misconfiguration is visible.
    body = _sse(
        "event: model_usage\ndata: {\"input_tokens\": 17}",
        "event: done\ndata: {}",
    )
    _patch_response(monkeypatch, body)

    target = StreamingHttpTarget(url="http://localhost:8080/run")
    with caplog.at_level(logging.WARNING, logger="harness_evals.targets.streaming_http"):
        result = await target.ainvoke(Golden(input="q"))

    assert result.output == ""
    assert any("output_event is not set" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Event capture
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_capture_events_stores_only_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _sse(
        "event: progress\ndata: {\"pct\": 10}",
        "event: tool_call\ndata: {\"name\": \"search\"}",
        "event: progress\ndata: {\"pct\": 90}",
        "event: message\ndata: {\"output\": \"done\"}",
    )
    _patch_response(monkeypatch, body)

    target = StreamingHttpTarget(url="http://localhost:8080/run", capture_events=["progress", "tool_call"])
    result = await target.ainvoke(Golden(input="q"))

    sse_events = result.meta("sse_events")
    assert sse_events == {
        "progress": [{"pct": 10}, {"pct": 90}],
        "tool_call": [{"name": "search"}],
    }
    # message was not requested, so it is not captured
    assert "message" not in sse_events
    assert result.output == "done"


@pytest.mark.unit
async def test_default_captures_all_events(monkeypatch: pytest.MonkeyPatch) -> None:
    # Unset capture_events -> every event is captured so metrics can evaluate
    # across the whole stream.
    body = _sse(
        "event: progress\ndata: {\"pct\": 10}",
        "event: tool_call\ndata: {\"name\": \"search\"}",
        "event: message\ndata: {\"output\": \"done\"}",
    )
    _patch_response(monkeypatch, body)

    target = StreamingHttpTarget(url="http://localhost:8080/run")
    result = await target.ainvoke(Golden(input="q"))

    assert result.output == "done"
    assert result.meta("sse_events") == {
        "progress": [{"pct": 10}],
        "tool_call": [{"name": "search"}],
        "message": [{"output": "done"}],
    }


@pytest.mark.unit
async def test_empty_capture_events_stores_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Explicit empty list is the opt-out: capture nothing.
    body = _sse("event: message\ndata: {\"output\": \"done\"}")
    _patch_response(monkeypatch, body)

    target = StreamingHttpTarget(url="http://localhost:8080/run", capture_events=[])
    result = await target.ainvoke(Golden(input="q"))

    assert result.output == "done"
    assert result.meta("sse_events") is None


@pytest.mark.unit
async def test_metadata_preserves_golden_metadata_plus_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _sse("event: progress\ndata: {\"pct\": 50}", "event: message\ndata: {\"output\": \"ok\"}")
    _patch_response(monkeypatch, body)

    target = StreamingHttpTarget(url="http://localhost:8080/run", capture_events=["progress"])
    golden = Golden(input="q", metadata={"suite": "smoke"})
    result = await target.ainvoke(golden)

    assert result.metadata is not None
    assert result.metadata["suite"] == "smoke"
    assert result.metadata["sse_events"] == {"progress": [{"pct": 50}]}


# ---------------------------------------------------------------------------
# Optional field extraction from the final payload
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_optional_fields_from_final_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _sse(
        "event: token\ndata: {\"chunk\": \"x\"}",
        "event: message\ndata: {\"output\": \"answer\", \"cost\": 0.01, \"tokens\": 12}",
    )
    _patch_response(monkeypatch, body)

    target = StreamingHttpTarget(
        url="http://localhost:8080/run",
        cost_usd_path="$.cost",
        token_count_path="$.tokens",
    )
    result = await target.ainvoke(Golden(input="q"))

    assert result.output == "answer"
    assert result.cost_usd == 0.01
    assert result.token_count == 12


# ---------------------------------------------------------------------------
# Request shape / auth
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_accepts_event_stream_and_applies_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_response(monkeypatch, _sse("event: message\ndata: {\"output\": \"ok\"}"))

    target = StreamingHttpTarget(url="http://localhost:8080/run", auth=BearerAuth(token="tok123"))
    await target.ainvoke(Golden(input="q"))

    headers = dict(captured[0].headers)
    # urllib title-cases header keys
    assert headers.get("Accept") == "text/event-stream"
    assert headers.get("Authorization") == "Bearer tok123"


@pytest.mark.unit
async def test_body_template_and_input_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_response(monkeypatch, _sse("event: message\ndata: {\"output\": \"ok\"}"))

    target = StreamingHttpTarget(
        url="http://localhost:8080/api",
        body_template={"data": {"query": None, "mode": "eval"}},
        input_path="$.data.query",
    )
    result = await target.ainvoke(Golden(input="test query"))

    assert result.output == "ok"
    import json

    body = json.loads(captured[0].data.decode())
    assert body == {"data": {"query": "test query", "mode": "eval"}}


@pytest.mark.unit
def test_rejects_bracket_input_path() -> None:
    with pytest.raises(ValueError, match="dot notation"):
        StreamingHttpTarget(url="http://localhost:8080/run", input_path="$.data[0].query")


# ---------------------------------------------------------------------------
# Failure semantics
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_transport_failure_returns_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_evals.targets import streaming_http as target_mod

    def fake_urlopen(request: Request, timeout: float, context=None) -> FakeHTTPResponse:
        raise URLError("Connection refused")

    monkeypatch.setattr(target_mod, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    target = StreamingHttpTarget(url="http://localhost:8080/run", retries=1, backoff_s=0.01)
    result = await target.ainvoke(Golden(input="test"))

    assert result.output == ""
    assert result.metadata is not None
    assert "http_error" in result.metadata
    assert "Connection refused" in result.metadata["http_error"]


@pytest.mark.unit
async def test_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_evals.targets import streaming_http as target_mod

    call_count = 0

    def fake_urlopen(request: Request, timeout: float, context=None) -> FakeHTTPResponse:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise URLError("Connection refused")
        return FakeHTTPResponse(_sse("event: message\ndata: {\"output\": \"recovered\"}"))

    monkeypatch.setattr(target_mod, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    target = StreamingHttpTarget(url="http://localhost:8080/run", retries=2, backoff_s=0.01)
    result = await target.ainvoke(Golden(input="test"))

    assert result.output == "recovered"
    assert call_count == 3


# ---------------------------------------------------------------------------
# Async production path (httpx.AsyncClient.stream)
#
# The config runner enters targets via ``async with target:`` (config/runner.py),
# which sets ``_async_client`` and routes ``ainvoke`` through ``_execute_async``.
# That path — line reassembly (``aiter_lines`` -> ``"\n".join``), content-type
# detection, and ``raise_for_status`` — is unique to the async client and is not
# exercised by the sync ``urlopen`` fallback tests above. These tests inject a
# fake async client so the real ``_execute_async`` code runs without httpx.
# ---------------------------------------------------------------------------


class _FakeStreamResponse:
    """Mimics the subset of ``httpx.Response`` used by ``_execute_async``."""

    def __init__(self, body: str, content_type: str = "text/event-stream", status_code: int = 200) -> None:
        self._body = body
        self.headers = {"content-type": content_type}
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    async def aiter_lines(self):
        # httpx yields lines without their trailing newline; the target rejoins
        # them with "\n". Splitting on "\n" reproduces that exactly.
        for line in self._body.split("\n"):
            yield line

    async def aiter_text(self):
        yield self._body


class _FakeStreamCtx:
    def __init__(self, response: _FakeStreamResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeStreamResponse:
        return self._response

    async def __aexit__(self, *_: object) -> None:
        return None


class _FakeAsyncClient:
    """Records each ``stream()`` call and returns a canned response.

    ``responses`` is consumed one per attempt; a ``BaseException`` value is
    raised from ``stream()`` (transport failure) while a ``_FakeStreamResponse``
    is returned normally.
    """

    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def stream(self, method: str, url: str, *, content: bytes | None = None, headers: dict | None = None):
        self.calls.append({"method": method, "url": url, "content": content, "headers": headers})
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return _FakeStreamCtx(item)  # type: ignore[arg-type]

    async def aclose(self) -> None:
        return None


@pytest.mark.unit
async def test_async_sse_line_reassembly_and_output() -> None:
    body = _sse(
        "event: token\ndata: {\"chunk\": \"partial\"}",
        "event: message\ndata: {\"output\": \"final answer\"}",
    )
    client = _FakeAsyncClient([_FakeStreamResponse(body)])

    target = StreamingHttpTarget(url="http://localhost:8080/run", auth=BearerAuth(token="tok123"))
    target._async_client = client  # inject fake, bypassing __aenter__

    result = await target.ainvoke(Golden(input="q"))

    # Output survives the aiter_lines -> "\n".join -> _parse_sse round-trip.
    assert result.output == "final answer"

    # Request shape: method, streaming Accept header, and auth all applied.
    call = client.calls[0]
    assert call["method"] == "POST"
    assert call["headers"]["Accept"] == "text/event-stream"
    assert call["headers"]["Authorization"] == "Bearer tok123"
    assert call["content"] == b'{"input": "q"}'


@pytest.mark.unit
async def test_async_captures_events_and_selects_output_event() -> None:
    body = _sse(
        "event: progress\ndata: {\"pct\": 10}",
        "event: message\ndata: {\"output\": \"ignored\"}",
        "event: final\ndata: {\"output\": \"chosen\"}",
    )
    client = _FakeAsyncClient([_FakeStreamResponse(body)])

    target = StreamingHttpTarget(
        url="http://localhost:8080/run",
        output_event="final",
        capture_events=["progress"],
    )
    target._async_client = client

    result = await target.ainvoke(Golden(input="q"))

    assert result.output == "chosen"
    assert result.meta("sse_events") == {"progress": [{"pct": 10}]}


@pytest.mark.unit
async def test_async_non_event_stream_uses_buffered_text_path() -> None:
    client = _FakeAsyncClient([_FakeStreamResponse('{"output": "buffered"}', content_type="application/json")])

    target = StreamingHttpTarget(url="http://localhost:8080/run")
    target._async_client = client

    result = await target.ainvoke(Golden(input="q"))

    # content-type is not text/event-stream -> aiter_text path -> buffered parse.
    assert result.output == "buffered"
    assert result.meta("sse_events") is None


@pytest.mark.unit
async def test_async_raise_for_status_retries_then_fails() -> None:
    # Every attempt returns a 500 whose raise_for_status raises; after retries
    # are exhausted the error is surfaced on the EvalCase.
    responses = [
        _FakeStreamResponse("", content_type="text/event-stream", status_code=500),
        _FakeStreamResponse("", content_type="text/event-stream", status_code=500),
    ]
    client = _FakeAsyncClient(responses)

    target = StreamingHttpTarget(url="http://localhost:8080/run", retries=1, backoff_s=0.0)
    target._async_client = client

    result = await target.ainvoke(Golden(input="q"))

    assert result.output == ""
    assert result.metadata is not None
    assert "http_error" in result.metadata
    assert "HTTP 500" in result.metadata["http_error"]
    assert len(client.calls) == 2  # initial attempt + one retry


@pytest.mark.unit
async def test_async_retries_then_succeeds() -> None:
    ok = _FakeStreamResponse(_sse("event: message\ndata: {\"output\": \"recovered\"}"))
    client = _FakeAsyncClient([URLError("boom"), ok])

    target = StreamingHttpTarget(url="http://localhost:8080/run", retries=2, backoff_s=0.0)
    target._async_client = client

    result = await target.ainvoke(Golden(input="q"))

    assert result.output == "recovered"
    assert len(client.calls) == 2


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_target_is_registered() -> None:
    assert plugins.target("streaming_http") is StreamingHttpTarget
