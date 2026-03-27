"""Tests for LangfuseSource adapter.

Uses mock objects to avoid requiring the langfuse package at test time.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest


@dataclass
class _FakeTrace:
    input: Any = "user question"
    output: Any = "assistant answer"
    tags: list[str] | None = field(default_factory=lambda: ["prod"])
    metadata: dict | None = field(default_factory=lambda: {"env": "test"})
    start_time: datetime | None = field(default_factory=lambda: datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc))
    end_time: datetime | None = field(default_factory=lambda: datetime(2025, 1, 1, 0, 0, 1, tzinfo=timezone.utc))


@dataclass
class _FakeObservation:
    type: str = "GENERATION"
    name: str | None = None
    input: Any = None
    output: Any = None
    usage_details: dict | None = None
    total_cost: float | None = None


@dataclass
class _FakeObservationList:
    data: list[_FakeObservation] = field(default_factory=list)


@pytest.fixture()
def _langfuse_module():
    """Inject a fake langfuse module so LangfuseSource can be imported without the real package."""
    fake_mod = ModuleType("langfuse")
    fake_mod.Langfuse = MagicMock
    already = "langfuse" in sys.modules
    old = sys.modules.get("langfuse")
    sys.modules["langfuse"] = fake_mod
    yield
    if already:
        sys.modules["langfuse"] = old
    else:
        del sys.modules["langfuse"]


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
class TestLangfuseSource:
    def _make_source(self, trace: _FakeTrace, observations: list[_FakeObservation]):
        from harness_evals.sources.langfuse import LangfuseSource

        client = MagicMock()
        client.api.trace.get.return_value = trace
        client.api.observations.get_many.return_value = _FakeObservationList(data=observations)
        return LangfuseSource(client)

    def test_basic_trace(self):
        source = self._make_source(_FakeTrace(), [])
        ec = source.from_trace("trace-123")
        assert ec.input == "user question"
        assert ec.output == "assistant answer"
        assert ec.latency_ms == pytest.approx(1000.0)
        assert ec.metadata["langfuse_trace_id"] == "trace-123"
        assert ec.tags == {"prod": "true"}

    def test_generation_observation_creates_messages(self):
        obs = _FakeObservation(
            type="GENERATION",
            input=[
                {"role": "user", "content": "What is 2+2?"},
            ],
            output={"role": "assistant", "content": "4"},
            usage_details={"input": 10, "output": 5},
            total_cost=0.001,
        )
        source = self._make_source(_FakeTrace(), [obs])
        ec = source.from_trace("t1")
        assert ec.messages is not None
        assert len(ec.messages) == 2
        assert ec.messages[0].role == "user"
        assert ec.messages[0].content == "What is 2+2?"
        assert ec.messages[1].role == "assistant"
        assert ec.messages[1].content == "4"
        assert ec.token_count == 15
        assert ec.cost_usd == pytest.approx(0.001)

    def test_tool_observation(self):
        obs = _FakeObservation(
            type="TOOL",
            name="search",
            input={"q": "hello"},
            output="results",
        )
        source = self._make_source(_FakeTrace(), [obs])
        ec = source.from_trace("t1")
        assert ec.tool_calls is not None
        assert len(ec.tool_calls) == 1
        assert ec.tool_calls[0].name == "search"
        assert ec.tool_calls[0].input == {"q": "hello"}
        assert ec.tool_calls[0].output == "results"

    def test_generation_with_tool_calls_in_output(self):
        obs = _FakeObservation(
            type="GENERATION",
            input=[{"role": "user", "content": "search for cats"}],
            output={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "function": {"name": "web_search", "arguments": {"q": "cats"}},
                    }
                ],
            },
            usage_details={"input": 5, "output": 10},
            total_cost=0.002,
        )
        source = self._make_source(_FakeTrace(), [obs])
        ec = source.from_trace("t1")
        assert ec.messages is not None
        assert len(ec.messages) == 2
        assert ec.messages[1].tool_calls is not None
        assert ec.messages[1].tool_calls[0].name == "web_search"
        assert ec.tool_calls is not None
        assert ec.tool_calls[0].name == "web_search"

    def test_no_timestamps_yields_no_latency(self):
        trace = _FakeTrace(start_time=None, end_time=None)
        source = self._make_source(trace, [])
        ec = source.from_trace("t1")
        assert ec.latency_ms is None

    def test_string_output_generation(self):
        obs = _FakeObservation(
            type="GENERATION",
            output="Just a string response",
        )
        source = self._make_source(_FakeTrace(), [obs])
        ec = source.from_trace("t1")
        assert ec.messages is not None
        assert ec.messages[0].role == "assistant"
        assert ec.messages[0].content == "Just a string response"

    def test_metadata_preserved_and_extended(self):
        trace = _FakeTrace(metadata={"custom_key": "custom_value"})
        source = self._make_source(trace, [])
        ec = source.from_trace("t1")
        assert ec.metadata["custom_key"] == "custom_value"
        assert ec.metadata["langfuse_trace_id"] == "t1"

    def test_no_tags(self):
        trace = _FakeTrace(tags=None)
        source = self._make_source(trace, [])
        ec = source.from_trace("t1")
        assert ec.tags is None

    def test_start_time_stored_in_metadata(self):
        source = self._make_source(_FakeTrace(), [])
        ec = source.from_trace("t1")
        assert "langfuse_trace_start_time" in ec.metadata


@dataclass
class _FakeTraceListItem:
    """Minimal trace object returned by api.trace.list()."""

    id: str = "trace-1"


@dataclass
class _FakePageMeta:
    next_cursor: str | None = None


@dataclass
class _FakeTracePage:
    data: list[_FakeTraceListItem] = field(default_factory=list)
    meta: _FakePageMeta = field(default_factory=_FakePageMeta)


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
class TestLangfuseSourceFromTraces:
    def _make_source(
        self,
        pages: list[_FakeTracePage],
        trace_map: dict[str, _FakeTrace] | None = None,
        obs_map: dict[str, list[_FakeObservation]] | None = None,
    ):
        from harness_evals.sources.langfuse import LangfuseSource

        client = MagicMock()
        client.api.trace.list.side_effect = pages

        if trace_map is None:
            trace_map = {}
        if obs_map is None:
            obs_map = {}

        def get_trace(tid):
            return trace_map.get(tid, _FakeTrace())

        def get_obs(trace_id, **_kwargs):
            return _FakeObservationList(data=obs_map.get(trace_id, []))

        client.api.trace.get.side_effect = get_trace
        client.api.observations.get_many.side_effect = get_obs
        return LangfuseSource(client), client

    def test_single_page(self):
        page = _FakeTracePage(
            data=[_FakeTraceListItem(id="t1"), _FakeTraceListItem(id="t2")],
        )
        source, client = self._make_source([page])
        cases = source.from_traces(tags=["prod"], limit=10)
        assert len(cases) == 2
        assert cases[0].metadata["langfuse_trace_id"] == "t1"
        assert cases[1].metadata["langfuse_trace_id"] == "t2"
        client.api.trace.list.assert_called_once()
        call_kwargs = client.api.trace.list.call_args
        assert call_kwargs.kwargs.get("tags") == ["prod"] or call_kwargs[1].get("tags") == ["prod"]

    def test_pagination(self):
        page1 = _FakeTracePage(
            data=[_FakeTraceListItem(id="t1")],
            meta=_FakePageMeta(next_cursor="cursor_2"),
        )
        page2 = _FakeTracePage(
            data=[_FakeTraceListItem(id="t2")],
        )
        source, client = self._make_source([page1, page2])
        cases = source.from_traces(limit=10)
        assert len(cases) == 2
        assert client.api.trace.list.call_count == 2

    def test_limit_truncates(self):
        page = _FakeTracePage(
            data=[_FakeTraceListItem(id=f"t{i}") for i in range(5)],
        )
        source, _ = self._make_source([page])
        cases = source.from_traces(limit=3)
        assert len(cases) == 3

    def test_empty_result(self):
        page = _FakeTracePage(data=[])
        source, _ = self._make_source([page])
        cases = source.from_traces()
        assert cases == []

    def test_filter_kwargs_forwarded(self):
        page = _FakeTracePage(data=[])
        source, client = self._make_source([page])
        source.from_traces(
            name="my-trace",
            user_id="u1",
            session_id="s1",
        )
        call_kwargs = client.api.trace.list.call_args[1]
        assert call_kwargs["name"] == "my-trace"
        assert call_kwargs["user_id"] == "u1"
        assert call_kwargs["session_id"] == "s1"
