"""Tests for the generic StreamingHttpTarget (SSE) adapter."""

from __future__ import annotations

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
async def test_no_capture_events_stores_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _sse("event: message\ndata: {\"output\": \"done\"}")
    _patch_response(monkeypatch, body)

    target = StreamingHttpTarget(url="http://localhost:8080/run")
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
# Plugin registration
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_target_is_registered() -> None:
    assert plugins.target("streaming_http") is StreamingHttpTarget
