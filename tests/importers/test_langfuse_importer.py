"""Tests for LangfuseEvalCaseSource.

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

from harness_evals.refs import ResourceRef


@dataclass
class _FakeTrace:
    input: Any = "user question"
    output: Any = "assistant answer"
    tags: list[str] | None = field(default_factory=lambda: ["prod"])
    metadata: dict | None = field(default_factory=lambda: {"env": "test"})
    start_time: datetime | None = field(default_factory=lambda: datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc))
    end_time: datetime | None = field(default_factory=lambda: datetime(2025, 1, 1, 0, 0, 1, tzinfo=timezone.utc))
    session_id: str | None = None


@dataclass
class _FakeObservation:
    type: str = "GENERATION"
    name: str | None = None
    input: Any = None
    output: Any = None
    usage_details: dict | None = None
    total_cost: float | None = None
    id: str | None = "obs-1"
    start_time: datetime | None = None
    end_time: datetime | None = None
    parent_observation_id: str | None = None


@dataclass
class _FakeObservationList:
    data: list[_FakeObservation] = field(default_factory=list)


@pytest.fixture()
def _langfuse_module():
    """Inject a fake langfuse module so LangfuseEvalCaseSource can be imported without the real package."""
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
class TestLangfuseEvalCaseSourceFromTrace:
    def _make_source(self, trace: _FakeTrace, observations: list[_FakeObservation]):
        from harness_evals.importers.langfuse import LangfuseEvalCaseSource

        client = MagicMock()
        client.api.trace.get.return_value = trace
        client.api.observations.get_many.return_value = _FakeObservationList(data=observations)
        return LangfuseEvalCaseSource(client)

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
            input=[{"role": "user", "content": "What is 2+2?"}],
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
                "tool_calls": [{"function": {"name": "web_search", "arguments": {"q": "cats"}}}],
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
        obs = _FakeObservation(type="GENERATION", output="Just a string response")
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
    id: str = "trace-1"
    session_id: str | None = None
    timestamp: datetime | None = None


@dataclass
class _FakePageMeta:
    page: int = 1
    total_pages: int = 1
    limit: int = 100
    total_items: int = 0


@dataclass
class _FakeTracePage:
    data: list[_FakeTraceListItem] = field(default_factory=list)
    meta: _FakePageMeta = field(default_factory=_FakePageMeta)


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
class TestLangfuseEvalCaseSourceFromTraces:
    def _make_source(
        self,
        pages: list[_FakeTracePage],
        trace_map: dict[str, _FakeTrace] | None = None,
        obs_map: dict[str, list[_FakeObservation]] | None = None,
    ):
        from harness_evals.importers.langfuse import LangfuseEvalCaseSource

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
        return LangfuseEvalCaseSource(client), client

    def test_single_page(self):
        page = _FakeTracePage(data=[_FakeTraceListItem(id="t1"), _FakeTraceListItem(id="t2")])
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
            meta=_FakePageMeta(page=1, total_pages=2, total_items=2),
        )
        page2 = _FakeTracePage(
            data=[_FakeTraceListItem(id="t2")],
            meta=_FakePageMeta(page=2, total_pages=2, total_items=2),
        )
        source, client = self._make_source([page1, page2])
        cases = source.from_traces(limit=10)
        assert len(cases) == 2
        assert client.api.trace.list.call_count == 2
        assert client.api.trace.list.call_args_list[0].kwargs.get("page") == 1
        assert client.api.trace.list.call_args_list[1].kwargs.get("page") == 2

    def test_limit_truncates(self):
        page = _FakeTracePage(data=[_FakeTraceListItem(id=f"t{i}") for i in range(5)])
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
        source.from_traces(name="my-trace", user_id="u1", session_id="s1")
        call_kwargs = client.api.trace.list.call_args[1]
        assert call_kwargs["name"] == "my-trace"
        assert call_kwargs["user_id"] == "u1"
        assert call_kwargs["session_id"] == "s1"


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
class TestLangfuseEvalCaseSourceFetch:
    """Tests for the uniform fetch(ref) entry point."""

    def _make_source(self, trace: _FakeTrace | None = None):
        from harness_evals.importers.langfuse import LangfuseEvalCaseSource

        client = MagicMock()
        client.api.trace.get.return_value = trace or _FakeTrace()
        client.api.observations.get_many.return_value = _FakeObservationList(data=[])
        return LangfuseEvalCaseSource(client), client

    @pytest.mark.asyncio
    async def test_fetch_single_trace_by_id(self):
        source, _ = self._make_source()
        ref = ResourceRef(source="langfuse", id="trace-xyz")
        cases = await source.fetch(ref)
        assert len(cases) == 1
        assert cases[0].metadata["langfuse_trace_id"] == "trace-xyz"

    @pytest.mark.asyncio
    async def test_fetch_dispatches_to_from_traces_with_filter_keys(self):
        page = _FakeTracePage(data=[_FakeTraceListItem(id="t1")])
        from harness_evals.importers.langfuse import LangfuseEvalCaseSource

        client = MagicMock()
        client.api.trace.list.return_value = page
        client.api.trace.get.return_value = _FakeTrace()
        client.api.observations.get_many.return_value = _FakeObservationList(data=[])
        source = LangfuseEvalCaseSource(client)

        ref = ResourceRef(source="langfuse", id="", extra={"tags": ["prod"], "limit": 5})
        cases = await source.fetch(ref)
        assert len(cases) == 1
        client.api.trace.list.assert_called_once()


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
class TestDeprecationWarnings:
    """Verify that importing from sources/ compat path emits DeprecationWarning."""

    def test_sources_langfuse_getattr_emits_warning(self):
        import harness_evals.sources.langfuse as compat_mod

        compat_mod.__dict__.pop("LangfuseSource", None)

        with pytest.warns(DeprecationWarning, match="LangfuseSource is deprecated"):
            _ = compat_mod.LangfuseSource

    def test_sources_init_getattr_emits_warning(self):
        import harness_evals.sources as sources_pkg

        sources_pkg.__dict__.pop("LangfuseSource", None)

        with pytest.warns(DeprecationWarning, match="LangfuseSource is deprecated"):
            _ = sources_pkg.LangfuseSource


@pytest.mark.unit
class TestInitSubclassEnforcement:
    """Verify that __init_subclass__ catches missing name attribute."""

    def test_missing_name_raises_type_error(self):
        from harness_evals.importers.base import BaseEvalCaseSource

        with pytest.raises(TypeError, match="must define a class-level 'name: str'"):

            class BadSource(BaseEvalCaseSource):
                async def fetch(self, ref):
                    return []


@pytest.mark.unit
@pytest.mark.usefixtures("_langfuse_module")
class TestLangfuseTraceCatalog:
    def test_list_and_load_spans_for_otel_builder(self):
        from harness_evals.importers.langfuse import LangfuseTraceCatalog
        from harness_evals.importers.otel import OTELEvalCaseSource
        from harness_evals.importers.trace_batch import SpanTrace

        client = MagicMock()
        client.api.trace.list.return_value = _FakeTracePage(
            data=[
                _FakeTraceListItem(id="t1", session_id="sess-a"),
                _FakeTraceListItem(id="t2", session_id="sess-a"),
            ]
        )
        client.api.trace.get.side_effect = lambda tid: _FakeTrace(
            session_id="sess-a",
            input={"prompt": "hello", "user_message": "hello"},
            output={"text": "ok"} if tid == "t2" else {"text": "hi"},
        )
        obs = {
            "t1": [
                _FakeObservation(
                    type="GENERATION",
                    name="chat",
                    input=[{"role": "user", "content": "hello"}],
                    output="hi",
                    start_time=datetime(2026, 9, 20, 1, 0, tzinfo=timezone.utc),
                    end_time=datetime(2026, 9, 20, 1, 0, 1, tzinfo=timezone.utc),
                )
            ],
            "t2": [
                _FakeObservation(
                    type="GENERATION",
                    name="chat",
                    input=[
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "hi"},
                        {"role": "user", "content": "more"},
                    ],
                    output="ok",
                    start_time=datetime(2026, 9, 20, 1, 1, tzinfo=timezone.utc),
                    end_time=datetime(2026, 9, 20, 1, 1, 1, tzinfo=timezone.utc),
                )
            ],
        }

        def get_obs(trace_id, **_kwargs):
            return _FakeObservationList(data=obs[trace_id])

        client.api.observations.get_many.side_effect = get_obs

        catalog = LangfuseTraceCatalog(client)
        listed = catalog.list_traces(limit=10)
        assert [t.trace_id for t in listed] == ["t1", "t2"]
        traces = [
            SpanTrace(
                spans=catalog.load_spans(stub.trace_id),
                trace_id=stub.trace_id,
                session_id=stub.session_id,
            )
            for stub in listed
        ]
        cases = OTELEvalCaseSource.from_span_traces(traces, group_by="session_id")
        assert len(cases) == 1
        assert cases[0].input == "hello"
        assert cases[0].messages[0].content == "hello"
        assert any(m.content == "more" for m in cases[0].messages if m.role == "user")
        assert cases[0].output == "ok"

    def test_trace_prompt_used_when_generation_input_missing(self):
        """UA traces often omit generation I/O; session input must come from trace.input."""
        from harness_evals.importers.langfuse import LangfuseTraceCatalog
        from harness_evals.importers.otel import OTELEvalCaseSource
        from harness_evals.importers.trace_batch import SpanTrace

        client = MagicMock()
        client.api.trace.list.return_value = _FakeTracePage(
            data=[
                _FakeTraceListItem(id="t1", session_id="sess-b"),
                _FakeTraceListItem(id="t2", session_id="sess-b"),
            ]
        )
        prompts = {
            "t1": {
                "input": {"prompt": "first question", "user_message": "first question"},
                "output": {"status": "completed", "text": "first answer"},
            },
            "t2": {
                "input": {"prompt": "follow up", "user_message": "follow up"},
                "output": {"status": "completed", "text": "final answer"},
            },
        }

        def get_trace(tid):
            payload = prompts[tid]
            return _FakeTrace(session_id="sess-b", input=payload["input"], output=payload["output"])

        client.api.trace.get.side_effect = get_trace

        def get_obs(trace_id, **_kwargs):
            # Generation observations with no input/output (prod UA shape).
            return _FakeObservationList(
                data=[
                    _FakeObservation(
                        type="GENERATION",
                        name="litellm_request",
                        input=None,
                        output=None,
                        start_time=datetime(2026, 9, 20, 1, 0 if trace_id == "t1" else 1, tzinfo=timezone.utc),
                        end_time=datetime(2026, 9, 20, 1, 0 if trace_id == "t1" else 1, 1, tzinfo=timezone.utc),
                    )
                ]
            )

        client.api.observations.get_many.side_effect = get_obs
        catalog = LangfuseTraceCatalog(client)
        traces = [
            SpanTrace(
                spans=catalog.load_spans(tid),
                trace_id=tid,
                session_id="sess-b",
                start_time=datetime(2026, 9, 20, 1, 0 if tid == "t1" else 1, tzinfo=timezone.utc),
            )
            for tid in ("t1", "t2")
        ]
        cases = OTELEvalCaseSource.from_span_traces(traces, group_by="session_id")
        assert len(cases) == 1
        assert cases[0].input == "first question"
        assert cases[0].output == "final answer"

    def test_generic_span_observations_are_not_tools(self):
        """Langfuse SPAN (mcp/rest) must not be mapped to execute_tool."""
        from harness_evals.importers.langfuse import LangfuseTraceCatalog, _observation_to_span

        client = MagicMock()
        client.api.trace.get.return_value = _FakeTrace(
            session_id="sess-c",
            input={"prompt": "Analyze the error", "user_message": "<current_context>\nAnalyze"},
            output={"status": "completed", "text": "## Analysis"},
        )
        client.api.observations.get_many.return_value = _FakeObservationList(
            data=[
                _FakeObservation(
                    type="SPAN",
                    name="rest.request",
                    input={"method": "GET"},
                    output={"status": 200},
                    parent_observation_id="agent-1",
                ),
                _FakeObservation(
                    type="TOOL",
                    name="harness_list",
                    input={"resource_type": "pipeline"},
                    output={"items": []},
                    parent_observation_id="agent-1",
                ),
                _FakeObservation(
                    type="AGENT",
                    name="chat_unified_agent",
                    input={"prompt": "Analyze the error"},
                    output=None,
                    parent_observation_id=None,
                ),
            ]
        )
        catalog = LangfuseTraceCatalog(client)
        spans = catalog.load_spans("t-span")
        by_name = {s["name"]: s["attributes"] for s in spans}
        assert by_name["rest.request"].get("gen_ai.operation.name") != "execute_tool"
        assert by_name["harness_list"].get("gen_ai.operation.name") == "execute_tool"
        assert by_name["chat_unified_agent"].get("gen_ai.operation.name") == "invoke_agent"

        from harness_evals.importers.otel import OTELEvalCaseSource
        from harness_evals.importers.trace_batch import SpanTrace

        cases = OTELEvalCaseSource.from_span_traces(
            [SpanTrace(spans=spans, trace_id="t-span", session_id="sess-c")]
        )
        assert len(cases) == 1
        assert cases[0].input == "Analyze the error"
        assert cases[0].output == "## Analysis"

