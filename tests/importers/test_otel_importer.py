"""Tests for OTELEvalCaseSource."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from harness_evals.importers.otel import OTELEvalCaseSource, SpanType, classify_span
from harness_evals.refs import ResourceRef


@pytest.mark.unit
class TestOTELEvalCaseSourceFromSpanJson:
    """Tests for legacy attribute format (backwards compat)."""

    def test_basic_llm_trace(self):
        spans = [
            {
                "name": "root",
                "attributes": {
                    "gen_ai.input": '{"query": "hello"}',
                    "gen_ai.output": '{"answer": "world"}',
                },
                "start_time_unix_nano": 1000000000,
                "end_time_unix_nano": 2000000000,
                "parent_span_id": None,
            },
            {
                "name": "gen_ai.chat",
                "attributes": {
                    "gen_ai.system": "openai",
                    "gen_ai.prompt": '[{"role": "user", "content": "hello"}]',
                    "gen_ai.completion": '{"role": "assistant", "content": "world"}',
                    "gen_ai.usage.input_tokens": 10,
                    "gen_ai.usage.output_tokens": 5,
                },
                "start_time_unix_nano": 1100000000,
                "end_time_unix_nano": 1900000000,
                "parent_span_id": "0001",
            },
        ]
        ec = OTELEvalCaseSource.from_span_json(spans)
        # New behavior: input/output are extracted from conversation messages
        assert ec.input == "hello"
        assert ec.output == "world"
        assert ec.latency_ms == pytest.approx(1000.0)
        assert ec.token_count == 15
        assert ec.messages is not None
        assert len(ec.messages) == 2
        assert ec.messages[0].role == "user"
        assert ec.messages[1].role == "assistant"

    def test_tool_spans(self):
        spans = [
            {
                "name": "root",
                "attributes": {"input": "query", "output": "result"},
                "start_time_unix_nano": 0,
                "end_time_unix_nano": 1000000000,
                "parent_span_id": None,
            },
            {
                "name": "tool.search",
                "attributes": {
                    "tool.name": "search",
                    "tool.input": '{"q": "foo"}',
                    "tool.output": "bar",
                },
                "start_time_unix_nano": 100000000,
                "end_time_unix_nano": 500000000,
                "parent_span_id": "0001",
            },
        ]
        ec = OTELEvalCaseSource.from_span_json(spans)
        assert ec.tool_calls is not None
        assert len(ec.tool_calls) == 1
        assert ec.tool_calls[0].name == "search"
        assert ec.tool_calls[0].input == {"q": "foo"}
        assert ec.tool_calls[0].output == "bar"

    def test_empty_spans(self):
        ec = OTELEvalCaseSource.from_span_json([])
        assert ec.input == ""
        assert ec.output == ""
        assert ec.messages is None
        assert ec.tool_calls is None

    def test_no_tokens(self):
        spans = [
            {
                "name": "root",
                "attributes": {"input": "q", "output": "a"},
                "parent_span_id": None,
            },
        ]
        ec = OTELEvalCaseSource.from_span_json(spans)
        assert ec.token_count is None

    def test_plain_string_completion(self):
        spans = [
            {
                "name": "llm.call",
                "attributes": {
                    "gen_ai.system": "openai",
                    "gen_ai.completion": "Hello there!",
                },
                "parent_span_id": None,
            },
        ]
        ec = OTELEvalCaseSource.from_span_json(spans)
        assert ec.messages is not None
        assert ec.messages[0].role == "assistant"
        assert ec.messages[0].content == "Hello there!"


@pytest.mark.unit
class TestOTELSemconvFormat:
    """Tests for the new OTel GenAI semantic conventions format."""

    def test_parentless_langfuse_tool_is_a_tool_call(self):
        span = {
            "name": "tool-observation",
            "attributes": {"langfuse.observation.type": "tool"},
            "parent_span_id": None,
        }

        assert classify_span(span) is SpanType.TOOL_CALL

    def test_new_semconv_attributes(self):
        """Test spans using gen_ai.operation.name, gen_ai.provider.name, etc."""
        spans = [
            {
                "name": "chat claude-sonnet-4-6-20250514",
                "span_id": "span_001",
                "attributes": {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.provider.name": "anthropic",
                    "gen_ai.request.model": "claude-sonnet-4-6-20250514",
                    "gen_ai.response.model": "claude-sonnet-4-6-20250514",
                    "gen_ai.usage.input_tokens": 100,
                    "gen_ai.usage.output_tokens": 50,
                    "gen_ai.input_messages": json.dumps(
                        [
                            {
                                "role": "user",
                                "parts": [{"type": "text", "content": "What is 2+2?"}],
                            }
                        ]
                    ),
                    "gen_ai.output_messages": json.dumps(
                        [
                            {
                                "role": "assistant",
                                "parts": [{"type": "text", "content": "4"}],
                            }
                        ]
                    ),
                },
                "start_time_unix_nano": 1000000000,
                "end_time_unix_nano": 1500000000,
                "parent_span_id": None,
            },
        ]
        ec = OTELEvalCaseSource.from_span_json(spans)
        assert ec.token_count == 150
        assert ec.messages is not None
        assert len(ec.messages) == 2
        assert ec.messages[0].role == "user"
        assert ec.messages[0].content == "What is 2+2?"
        assert ec.messages[1].role == "assistant"
        assert ec.messages[1].content == "4"
        assert ec.metadata is not None
        assert ec.metadata["provider"] == "anthropic"
        assert ec.metadata["model"] == "claude-sonnet-4-6-20250514"

    def test_new_semconv_tool_spans(self):
        """Test tool spans using gen_ai.tool.* attributes."""
        spans = [
            {
                "name": "chat gpt-4",
                "span_id": "span_001",
                "attributes": {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.provider.name": "openai",
                    "gen_ai.request.model": "gpt-4",
                    "gen_ai.usage.input_tokens": 200,
                    "gen_ai.usage.output_tokens": 80,
                    "gen_ai.input_messages": json.dumps(
                        [
                            {"role": "user", "parts": [{"type": "text", "content": "Look up order 123"}]},
                        ]
                    ),
                    "gen_ai.output_messages": json.dumps(
                        [
                            {
                                "role": "assistant",
                                "parts": [
                                    {"type": "text", "content": "Let me look that up."},
                                    {
                                        "type": "tool_call",
                                        "content": json.dumps({"name": "lookup_order", "arguments": {"id": "123"}}),
                                    },
                                ],
                            }
                        ]
                    ),
                },
                "start_time_unix_nano": 1000000000,
                "end_time_unix_nano": 2000000000,
                "parent_span_id": None,
            },
            {
                "name": "execute_tool lookup_order",
                "span_id": "span_002",
                "attributes": {
                    "gen_ai.tool.name": "lookup_order",
                    "gen_ai.tool.type": "function",
                    "gen_ai.tool.call.arguments": '{"id": "123"}',
                    "gen_ai.tool.call.result": '{"status": "shipped"}',
                },
                "start_time_unix_nano": 2000000000,
                "end_time_unix_nano": 2500000000,
                "parent_span_id": "span_001",
            },
        ]
        ec = OTELEvalCaseSource.from_span_json(spans)
        assert ec.tool_calls is not None
        assert len(ec.tool_calls) == 1
        assert ec.tool_calls[0].name == "lookup_order"
        assert ec.tool_calls[0].input == {"id": "123"}
        assert ec.tool_calls[0].output == '{"status": "shipped"}'

    def test_tool_call_in_messages(self):
        """Tool calls embedded in output_messages are parsed into Message.tool_calls."""
        spans = [
            {
                "name": "chat gpt-4",
                "span_id": "span_001",
                "attributes": {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.provider.name": "openai",
                    "gen_ai.request.model": "gpt-4",
                    "gen_ai.input_messages": json.dumps(
                        [
                            {"role": "user", "parts": [{"type": "text", "content": "hello"}]},
                        ]
                    ),
                    "gen_ai.output_messages": json.dumps(
                        [
                            {
                                "role": "assistant",
                                "parts": [
                                    {
                                        "type": "tool_call",
                                        "content": json.dumps({"name": "get_weather", "arguments": {"city": "NYC"}}),
                                    },
                                ],
                            }
                        ]
                    ),
                },
                "start_time_unix_nano": 1000000000,
                "end_time_unix_nano": 2000000000,
                "parent_span_id": None,
            },
        ]
        ec = OTELEvalCaseSource.from_span_json(spans)
        assert ec.messages is not None
        assistant_msg = ec.messages[1]
        assert assistant_msg.role == "assistant"
        assert assistant_msg.tool_calls is not None
        assert assistant_msg.tool_calls[0].name == "get_weather"
        assert assistant_msg.tool_calls[0].input == {"city": "NYC"}

    def test_system_instructions_in_metadata(self):
        """gen_ai.system_instructions is available via span attributes."""
        spans = [
            {
                "name": "chat gpt-4",
                "span_id": "span_001",
                "attributes": {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.provider.name": "openai",
                    "gen_ai.request.model": "gpt-4",
                    "gen_ai.system_instructions": "You are helpful.",
                    "gen_ai.input_messages": json.dumps(
                        [
                            {"role": "user", "parts": [{"type": "text", "content": "hi"}]},
                        ]
                    ),
                    "gen_ai.output_messages": json.dumps(
                        [
                            {"role": "assistant", "parts": [{"type": "text", "content": "hello!"}]},
                        ]
                    ),
                    "gen_ai.usage.input_tokens": 20,
                    "gen_ai.usage.output_tokens": 5,
                },
                "start_time_unix_nano": 1000000000,
                "end_time_unix_nano": 1200000000,
                "parent_span_id": None,
            },
        ]
        ec = OTELEvalCaseSource.from_span_json(spans)
        assert ec.messages is not None
        assert ec.messages[0].role == "user"
        assert ec.messages[1].role == "assistant"
        assert ec.messages[1].content == "hello!"

    def test_langfuse_agent_generation_and_tool_observations(self):
        spans = [
            {
                "name": "review-agent",
                "span_id": "root",
                "attributes": {
                    "langfuse.observation.type": "agent",
                    "langfuse.observation.input": json.dumps(
                        [{"role": "user", "content": [{"type": "text", "text": "Review this pull request"}]}]
                    ),
                },
                "parent_span_id": None,
            },
            {
                "name": "review-turn",
                "span_id": "llm",
                "attributes": {
                    "langfuse.observation.type": "generation",
                    "langfuse.observation.output": json.dumps(
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "I will inspect the change."},
                                {"type": "tool_use", "name": "Read", "input": {"path": "app.py"}},
                            ],
                        }
                    ),
                },
                "parent_span_id": "root",
            },
            {
                "name": "tool-observation",
                "span_id": "tool",
                "attributes": {
                    "langfuse.observation.type": "tool",
                    "langfuse.observation.name": "Read",
                    "langfuse.observation.input": json.dumps({"arguments": {"path": "app.py"}}),
                    "langfuse.observation.output": json.dumps({"content": "print('hello')"}),
                },
                "parent_span_id": "root",
            },
        ]

        ec = OTELEvalCaseSource.from_span_json(spans)

        assert ec.input == "Review this pull request"
        assert ec.output == "I will inspect the change."
        assert ec.tool_calls is not None
        assert ec.tool_calls[0].name == "Read"
        assert ec.tool_calls[0].input == {"path": "app.py"}
        assert ec.tool_calls[0].output == "print('hello')"

    def test_langfuse_plain_text_and_query_fields(self):
        spans = [
            {
                "name": "agent",
                "span_id": "root",
                "attributes": {
                    "langfuse.observation.type": "agent",
                    "langfuse.observation.input": json.dumps({"query": "Review this pull request"}),
                },
                "parent_span_id": None,
            },
            {
                "name": "generation",
                "span_id": "llm",
                "attributes": {
                    "langfuse.observation.type": "generation",
                    "langfuse.observation.output": "No issues found.",
                },
                "parent_span_id": "root",
            },
        ]

        ec = OTELEvalCaseSource.from_span_json(spans)

        assert ec.input == "Review this pull request"
        assert ec.output == "No issues found."


@pytest.mark.unit
class TestOTELEvalCaseSourceFetch:
    """Tests for the uniform fetch(ref) entry point."""

    @pytest.mark.asyncio
    async def test_fetch_reads_file_and_returns_single_case(self):
        spans = [
            {
                "name": "chat gpt-4",
                "span_id": "span_001",
                "attributes": {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.provider.name": "openai",
                    "gen_ai.request.model": "gpt-4",
                    "gen_ai.input_messages": json.dumps(
                        [
                            {"role": "user", "parts": [{"type": "text", "content": "what is 2+2?"}]},
                        ]
                    ),
                    "gen_ai.output_messages": json.dumps(
                        [
                            {"role": "assistant", "parts": [{"type": "text", "content": "4"}]},
                        ]
                    ),
                },
                "start_time_unix_nano": 1000000000,
                "end_time_unix_nano": 2000000000,
                "parent_span_id": None,
            }
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(spans, f)
            tmp_path = f.name

        source = OTELEvalCaseSource()
        ref = ResourceRef(source="otel", id=tmp_path)
        cases = await source.fetch(ref)

        assert len(cases) == 1
        assert cases[0].input == "what is 2+2?"
        assert cases[0].output == "4"
        Path(tmp_path).unlink()

    @pytest.mark.asyncio
    async def test_fetch_missing_file_raises(self):
        source = OTELEvalCaseSource()
        ref = ResourceRef(source="otel", id="/nonexistent/path/spans.json")
        with pytest.raises(FileNotFoundError):
            await source.fetch(ref)


@pytest.mark.unit
class TestAgentRootOnlyOutput:
    """Regression: single invoke_agent span with output must not be lost."""

    def test_single_agent_root_span_extracts_output(self):
        spans = [
            {
                "name": "invoke_agent",
                "attributes": {
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.input_messages": json.dumps(
                        [{"role": "user", "parts": [{"type": "text", "content": "hello"}]}]
                    ),
                    "gen_ai.output_messages": json.dumps(
                        [{"role": "assistant", "parts": [{"type": "text", "content": "hi there"}]}]
                    ),
                },
                "start_time_unix_nano": 1000000000,
                "end_time_unix_nano": 2000000000,
                "parent_span_id": None,
            }
        ]
        ec = OTELEvalCaseSource.from_span_json(spans)
        assert ec.input == "hello"
        assert ec.output == "hi there"
        assert any(m.role == "assistant" and m.content == "hi there" for m in ec.messages)

    def test_agent_root_with_child_llm_turn_does_not_duplicate(self):
        spans = [
            {
                "name": "invoke_agent",
                "attributes": {
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.input_messages": json.dumps(
                        [{"role": "user", "parts": [{"type": "text", "content": "hello"}]}]
                    ),
                    "gen_ai.output_messages": json.dumps(
                        [{"role": "assistant", "parts": [{"type": "text", "content": "hi there"}]}]
                    ),
                },
                "start_time_unix_nano": 1000000000,
                "end_time_unix_nano": 3000000000,
                "parent_span_id": None,
            },
            {
                "name": "gen_ai.chat",
                "attributes": {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.system": "openai",
                    "gen_ai.input_messages": json.dumps(
                        [{"role": "user", "parts": [{"type": "text", "content": "hello"}]}]
                    ),
                    "gen_ai.output_messages": json.dumps(
                        [{"role": "assistant", "parts": [{"type": "text", "content": "hi there"}]}]
                    ),
                    "gen_ai.usage.input_tokens": 5,
                    "gen_ai.usage.output_tokens": 3,
                },
                "start_time_unix_nano": 1100000000,
                "end_time_unix_nano": 2000000000,
                "parent_span_id": "root",
            },
        ]
        ec = OTELEvalCaseSource.from_span_json(spans)
        assistant_msgs = [m for m in ec.messages if m.role == "assistant"]
        assert len(assistant_msgs) == 1


@pytest.mark.unit
class TestMultiTurnUserRecovery:
    """Regression: intermediate user messages must appear in the trajectory."""

    def test_two_turn_conversation_recovers_second_user_message(self):
        spans = [
            {
                "name": "invoke_agent",
                "attributes": {
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.input_messages": json.dumps(
                        [{"role": "user", "parts": [{"type": "text", "content": "first question"}]}]
                    ),
                },
                "start_time_unix_nano": 1000000000,
                "end_time_unix_nano": 5000000000,
                "parent_span_id": None,
            },
            {
                "name": "gen_ai.chat",
                "attributes": {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.system": "openai",
                    "gen_ai.input_messages": json.dumps(
                        [{"role": "user", "parts": [{"type": "text", "content": "first question"}]}]
                    ),
                    "gen_ai.output_messages": json.dumps(
                        [{"role": "assistant", "parts": [{"type": "text", "content": "first answer"}]}]
                    ),
                    "gen_ai.usage.input_tokens": 5,
                    "gen_ai.usage.output_tokens": 3,
                },
                "start_time_unix_nano": 1100000000,
                "end_time_unix_nano": 2000000000,
                "parent_span_id": "root",
            },
            {
                "name": "gen_ai.chat",
                "attributes": {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.system": "openai",
                    "gen_ai.input_messages": json.dumps(
                        [
                            {"role": "user", "parts": [{"type": "text", "content": "first question"}]},
                            {"role": "assistant", "parts": [{"type": "text", "content": "first answer"}]},
                            {"role": "user", "parts": [{"type": "text", "content": "second question"}]},
                        ]
                    ),
                    "gen_ai.output_messages": json.dumps(
                        [{"role": "assistant", "parts": [{"type": "text", "content": "second answer"}]}]
                    ),
                    "gen_ai.usage.input_tokens": 10,
                    "gen_ai.usage.output_tokens": 5,
                },
                "start_time_unix_nano": 3000000000,
                "end_time_unix_nano": 4000000000,
                "parent_span_id": "root",
            },
        ]
        ec = OTELEvalCaseSource.from_span_json(spans)
        roles = [m.role for m in ec.messages]
        assert roles == ["user", "assistant", "user", "assistant"]
        assert ec.messages[2].content == "second question"
        assert ec.output == "second answer"


def _chat_spans(user: str, assistant: str, *, session: str, trace_id: str, start_nano: int) -> list[dict]:
    return [
        {
            "name": "chat",
            "span_id": f"{trace_id}-llm",
            "trace_id": trace_id,
            "parent_span_id": None,
            "start_time_unix_nano": start_nano,
            "end_time_unix_nano": start_nano + 1_000_000_000,
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.conversation.id": session,
                "gen_ai.input_messages": json.dumps([{"role": "user", "parts": [{"type": "text", "content": user}]}]),
                "gen_ai.output_messages": json.dumps(
                    [{"role": "assistant", "parts": [{"type": "text", "content": assistant}]}]
                ),
            },
        }
    ]


@pytest.mark.unit
class TestOTELSessionGrouping:
    def test_merges_traces_that_share_session_id(self):
        from harness_evals.importers.trace_batch import SpanTrace

        traces = [
            SpanTrace(
                spans=_chat_spans("first", "ack", session="sess-1", trace_id="t1", start_nano=1_000_000_000),
                trace_id="t1",
                session_id="sess-1",
            ),
            SpanTrace(
                spans=_chat_spans("second", "done", session="sess-1", trace_id="t2", start_nano=3_000_000_000),
                trace_id="t2",
                session_id="sess-1",
            ),
        ]
        cases = OTELEvalCaseSource.from_span_traces(traces, group_by="session_id")
        assert len(cases) == 1
        roles = [m.role for m in cases[0].messages]
        assert roles == ["user", "assistant", "user", "assistant"]
        assert cases[0].messages[2].content == "second"
        assert cases[0].output == "done"
        assert cases[0].metadata["session_id"] == "sess-1"
        assert cases[0].metadata["trace_ids"] == ["t1", "t2"]

    def test_session_merge_sums_per_trace_total_cost(self):
        """Trace-level cost stamps (one per source trace) must sum on merge."""
        from harness_evals.importers.trace_batch import SpanTrace

        t1 = _chat_spans("first", "ack", session="sess-1", trace_id="t1", start_nano=1_000_000_000)
        t2 = _chat_spans("second", "done", session="sess-1", trace_id="t2", start_nano=3_000_000_000)
        # Mirror Langfuse load_spans: stamp total_cost once on the first span
        # of each trace (no gen_ai.usage.cost → fallback path).
        t1[0]["attributes"]["langfuse.trace.total_cost"] = 0.01
        t2[0]["attributes"]["langfuse.trace.total_cost"] = 0.02
        traces = [
            SpanTrace(spans=t1, trace_id="t1", session_id="sess-1"),
            SpanTrace(spans=t2, trace_id="t2", session_id="sess-1"),
        ]
        cases = OTELEvalCaseSource.from_span_traces(traces, group_by="session_id")
        assert len(cases) == 1
        assert cases[0].cost_usd == pytest.approx(0.03)

    def test_prefers_child_usage_cost_over_rolled_up_root(self):
        """Skip agent-root usage cost when child spans also carry cost."""
        from harness_evals.importers.otel import _build_conversation_eval_case

        spans = [
            {
                "name": "agent",
                "span_id": "root",
                "trace_id": "t1",
                "parent_span_id": None,
                "attributes": {
                    "gen_ai.operation.name": "invoke_agent",
                    "langfuse.observation.type": "agent",
                    "gen_ai.usage.cost": 0.03,
                    "gen_ai.input_messages": json.dumps(
                        [{"role": "user", "parts": [{"type": "text", "content": "hi"}]}]
                    ),
                },
                "start_time_unix_nano": 1,
                "end_time_unix_nano": 4,
            },
            {
                "name": "llm_turn_1",
                "span_id": "llm1",
                "trace_id": "t1",
                "parent_span_id": "root",
                "attributes": {
                    "langfuse.observation.type": "generation",
                    "gen_ai.usage.cost": 0.01,
                    "gen_ai.output_messages": json.dumps(
                        [{"role": "assistant", "parts": [{"type": "text", "content": "a"}]}]
                    ),
                },
                "start_time_unix_nano": 2,
                "end_time_unix_nano": 3,
            },
            {
                "name": "llm_turn_2",
                "span_id": "llm2",
                "trace_id": "t1",
                "parent_span_id": "root",
                "attributes": {
                    "langfuse.observation.type": "generation",
                    "gen_ai.usage.cost": 0.02,
                    "gen_ai.output_messages": json.dumps(
                        [{"role": "assistant", "parts": [{"type": "text", "content": "b"}]}]
                    ),
                },
                "start_time_unix_nano": 3,
                "end_time_unix_nano": 4,
            },
        ]
        ec = _build_conversation_eval_case(spans)
        assert ec.cost_usd == pytest.approx(0.03)

    def test_root_only_usage_cost_is_kept(self):
        from harness_evals.importers.otel import _build_conversation_eval_case

        spans = [
            {
                "name": "agent",
                "span_id": "root",
                "trace_id": "t1",
                "parent_span_id": None,
                "attributes": {
                    "gen_ai.operation.name": "invoke_agent",
                    "langfuse.observation.type": "agent",
                    "gen_ai.usage.cost": 0.05,
                    "gen_ai.input_messages": json.dumps(
                        [{"role": "user", "parts": [{"type": "text", "content": "hi"}]}]
                    ),
                    "gen_ai.output_messages": json.dumps(
                        [{"role": "assistant", "parts": [{"type": "text", "content": "ok"}]}]
                    ),
                },
                "start_time_unix_nano": 1,
                "end_time_unix_nano": 2,
            }
        ]
        ec = _build_conversation_eval_case(spans)
        assert ec.cost_usd == pytest.approx(0.05)

    def test_prefers_harness_agent_run_cost_over_child_and_trace(self):
        """SDK cost on harness_agent_run beats litellm children + trace total."""
        from harness_evals.importers.otel import _build_conversation_eval_case

        spans = [
            {
                "name": "unified_v2_chat",
                "span_id": "root",
                "trace_id": "t1",
                "parent_span_id": None,
                "attributes": {
                    "langfuse.observation.type": "span",
                    "langfuse.trace.total_cost": 0.99,
                    "langfuse.trace.input": {"prompt": "hi"},
                    "langfuse.trace.output": {"text": "ok"},
                },
                "start_time_unix_nano": 1,
                "end_time_unix_nano": 10,
            },
            {
                "name": "harness_agent_run",
                "span_id": "agent",
                "trace_id": "t1",
                "parent_span_id": "root",
                "attributes": {
                    "langfuse.observation.type": "agent",
                    "gen_ai.operation.name": "invoke_agent",
                    "agent.total_cost_usd": "0.1456975",
                    "gen_ai.usage.cost": "0.1456975",
                },
                "start_time_unix_nano": 2,
                "end_time_unix_nano": 9,
            },
            {
                "name": "litellm_request",
                "span_id": "llm",
                "trace_id": "t1",
                "parent_span_id": "agent",
                "attributes": {
                    "langfuse.observation.type": "generation",
                    "gen_ai.usage.cost": 0.04,
                },
                "start_time_unix_nano": 3,
                "end_time_unix_nano": 4,
            },
        ]
        ec = _build_conversation_eval_case(spans)
        assert ec.cost_usd == pytest.approx(0.1456975)

    def test_session_merge_sums_harness_agent_run_costs(self):
        from harness_evals.importers.trace_batch import SpanTrace

        def _turn(prompt: str, answer: str, *, tid: str, cost: float, start: int) -> list[dict]:
            return [
                {
                    "name": "harness_agent_run",
                    "span_id": f"{tid}-agent",
                    "trace_id": tid,
                    "parent_span_id": None,
                    "attributes": {
                        "gen_ai.conversation.id": "sess-1",
                        "langfuse.observation.type": "agent",
                        "gen_ai.operation.name": "invoke_agent",
                        "agent.total_cost_usd": str(cost),
                        "langfuse.trace.input": {"prompt": prompt},
                        "langfuse.trace.output": {"text": answer},
                        "langfuse.trace.total_cost": 9.99,
                    },
                    "start_time_unix_nano": start,
                    "end_time_unix_nano": start + 1,
                }
            ]

        traces = [
            SpanTrace(
                spans=_turn("first", "ack", tid="t1", cost=0.01, start=1_000_000_000),
                trace_id="t1",
                session_id="sess-1",
            ),
            SpanTrace(
                spans=_turn("second", "done", tid="t2", cost=0.02, start=3_000_000_000),
                trace_id="t2",
                session_id="sess-1",
            ),
        ]
        cases = OTELEvalCaseSource.from_span_traces(traces, group_by="session_id")
        assert len(cases) == 1
        assert cases[0].cost_usd == pytest.approx(0.03)

    def test_missing_session_id_stays_one_case_per_trace(self):
        from harness_evals.importers.trace_batch import SpanTrace

        traces = [
            SpanTrace(spans=_chat_spans("a", "b", session="", trace_id="t1", start_nano=1), trace_id="t1"),
            SpanTrace(spans=_chat_spans("c", "d", session="", trace_id="t2", start_nano=2), trace_id="t2"),
        ]
        # strip conversation ids so grouping cannot collapse them
        for trace in traces:
            for span in trace.spans:
                span["attributes"].pop("gen_ai.conversation.id", None)
        cases = OTELEvalCaseSource.from_span_traces(traces, group_by="session_id")
        assert len(cases) == 2

    def test_distinct_sessions_stay_separate(self):
        from harness_evals.importers.trace_batch import SpanTrace

        traces = [
            SpanTrace(
                spans=_chat_spans("a", "b", session="s1", trace_id="t1", start_nano=1),
                trace_id="t1",
                session_id="s1",
            ),
            SpanTrace(
                spans=_chat_spans("c", "d", session="s2", trace_id="t2", start_nano=2),
                trace_id="t2",
                session_id="s2",
            ),
        ]
        cases = OTELEvalCaseSource.from_span_traces(traces, group_by="session_id")
        assert len(cases) == 2


@pytest.mark.unit
class TestOTELTimeWindowFetch:
    @pytest.mark.asyncio
    async def test_lookback_days_drops_old_file_traces(self, tmp_path):
        from datetime import datetime, timezone

        now = datetime(2026, 9, 21, tzinfo=timezone.utc)
        old_nano = int(datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp() * 1e9)
        new_nano = int(datetime(2026, 9, 20, tzinfo=timezone.utc).timestamp() * 1e9)
        payload = {
            "traces": [
                {
                    "trace_id": "old",
                    "session_id": "s-old",
                    "spans": _chat_spans("old", "old-out", session="s-old", trace_id="old", start_nano=old_nano),
                },
                {
                    "trace_id": "new",
                    "session_id": "s-new",
                    "spans": _chat_spans("new", "new-out", session="s-new", trace_id="new", start_nano=new_nano),
                },
            ]
        }
        path = tmp_path / "traces.json"
        path.write_text(json.dumps(payload))

        source = OTELEvalCaseSource(now=lambda: now)
        cases = await source.fetch(
            ResourceRef(source="otel", id=str(path), extra={"lookback_days": 7, "group_by": "session_id"})
        )
        assert len(cases) == 1
        assert cases[0].input == "new"

    @pytest.mark.asyncio
    async def test_catalog_lists_and_merges_sessions(self):
        from datetime import datetime, timezone

        from harness_evals.importers.trace_batch import SpanTrace

        class _Catalog:
            def list_traces(self, **kwargs):
                assert kwargs["from_timestamp"] is not None
                return [
                    SpanTrace(spans=[], trace_id="t1", session_id="sess"),
                    SpanTrace(spans=[], trace_id="t2", session_id="sess"),
                ]

            def load_spans(self, trace_id: str):
                start = 1_000_000_000 if trace_id == "t1" else 3_000_000_000
                user = "first" if trace_id == "t1" else "second"
                return _chat_spans(user, user + "-out", session="sess", trace_id=trace_id, start_nano=start)

        source = OTELEvalCaseSource(
            catalog=_Catalog(),
            now=lambda: datetime(2026, 9, 21, tzinfo=timezone.utc),
        )
        cases = await source.fetch(
            ResourceRef(source="otel", id="", extra={"lookback_days": 7, "group_by": "session_id"})
        )
        assert len(cases) == 1
        assert [m.content for m in cases[0].messages if m.role == "user"] == ["first", "second"]

    @pytest.mark.asyncio
    async def test_hydrate_skips_traces_that_fail_load_spans(self):
        from datetime import datetime, timezone

        from harness_evals.importers.trace_batch import SpanTrace

        class _Catalog:
            def list_traces(self, **kwargs):
                return [
                    SpanTrace(spans=[], trace_id="ok", session_id="s-ok"),
                    SpanTrace(spans=[], trace_id="huge", session_id="s-huge"),
                ]

            def load_spans(self, trace_id: str):
                if trace_id == "huge":
                    raise RuntimeError("Observations in trace are too large: 80.11MB exceeds limit of 80.00MB")
                return _chat_spans(
                    "ok prompt",
                    "ok-out",
                    session="s-ok",
                    trace_id="ok",
                    start_nano=1_000_000_000,
                )

        source = OTELEvalCaseSource(
            catalog=_Catalog(),
            now=lambda: datetime(2026, 9, 21, tzinfo=timezone.utc),
        )
        cases = await source.fetch(
            ResourceRef(source="otel", id="", extra={"lookback_days": 7, "group_by": "session_id"})
        )
        assert len(cases) == 1
        assert cases[0].input == "ok prompt"

    @pytest.mark.asyncio
    async def test_hydrate_skips_sessions_that_fail_expand(self):
        from datetime import datetime, timezone

        from harness_evals.importers.trace_batch import SpanTrace

        class _Catalog:
            def list_traces(self, **kwargs):
                return [
                    SpanTrace(spans=[], trace_id="ok", session_id="s-ok"),
                    SpanTrace(spans=[], trace_id="partial", session_id="s-fail"),
                ]

            def list_session_trace_ids(self, session_id: str):
                if session_id == "s-fail":
                    raise RuntimeError("status_code: 502, body: Server Error")
                return ["ok"]

            def load_spans(self, trace_id: str):
                session = "s-ok" if trace_id == "ok" else "s-fail"
                return _chat_spans(
                    f"{trace_id} prompt",
                    f"{trace_id}-out",
                    session=session,
                    trace_id=trace_id,
                    start_nano=1_000_000_000,
                )

        source = OTELEvalCaseSource(
            catalog=_Catalog(),
            now=lambda: datetime(2026, 9, 21, tzinfo=timezone.utc),
        )
        listed = [
            SpanTrace(spans=[], trace_id="ok", session_id="s-ok"),
            SpanTrace(spans=[], trace_id="partial", session_id="s-fail"),
        ]
        traces = await source._hydrate_catalog_traces(listed, expand_sessions=True)
        cases = source.from_span_traces(traces, group_by="session_id")
        assert {c.input for c in cases} == {"ok prompt", "partial prompt"}

    @pytest.mark.asyncio
    async def test_time_filters_without_catalog_or_path_raise(self):
        source = OTELEvalCaseSource()
        with pytest.raises(ValueError, match="TraceCatalog"):
            await source.fetch(ResourceRef(source="otel", id="", extra={"lookback_days": 7}))


@pytest.mark.unit
class TestLangfuseTraceIoRecovery:
    """Recover session prompt from langfuse.trace.input when tool spans dominate."""

    def test_prefers_prompt_over_user_message_when_only_tools(self):
        from harness_evals.importers.otel import _build_conversation_eval_case

        spans = [
            {
                "name": "http.request",
                "span_id": "1",
                "trace_id": "t1",
                "parent_span_id": "root",
                "attributes": {
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "http.request",
                    "gen_ai.tool.call.result": '{"ok": true}',
                    "langfuse.trace.input": {
                        "prompt": "Summarize the failed job logs",
                        "user_message": "<extra_context>\nSummarize the failed job logs",
                    },
                    "langfuse.trace.output": {
                        "status": "completed",
                        "text": "## Summary: Missing configuration key",
                    },
                },
                "start_time_unix_nano": 1,
                "end_time_unix_nano": 2,
            }
        ]
        ec = _build_conversation_eval_case(spans)
        assert ec.input == "Summarize the failed job logs"
        assert ec.output == "## Summary: Missing configuration key"

    def test_prefers_trace_prompt_over_system_reminder_input_messages(self):
        from harness_evals.importers.otel import _build_conversation_eval_case

        spans = [
            {
                "name": "chat.completion",
                "span_id": "2",
                "trace_id": "t2",
                "parent_span_id": None,
                "attributes": {
                    "langfuse.observation.type": "generation",
                    "gen_ai.input_messages": [
                        {
                            "role": "user",
                            "content": "<system-reminder>\nThe following skills are available\n",
                        }
                    ],
                    "gen_ai.output_messages": [{"role": "assistant", "content": "done"}],
                    "langfuse.trace.input": {
                        "prompt": "Ask a support question",
                        "user_message": "<extra_context>\nAsk a support question",
                    },
                },
                "start_time_unix_nano": 1,
                "end_time_unix_nano": 2,
            }
        ]
        ec = _build_conversation_eval_case(spans)
        assert ec.input == "Ask a support question"
        assert ec.output == "done"

    def test_output_messages_list_content_does_not_crash(self):
        """Assistant content may be multipart list (Langfuse / GenAI exporters)."""
        from harness_evals.importers.otel import _build_conversation_eval_case

        spans = [
            {
                "name": "chat.completion",
                "span_id": "1",
                "trace_id": "t-list-content",
                "parent_span_id": None,
                "attributes": {
                    "langfuse.observation.type": "generation",
                    "gen_ai.input_messages": [{"role": "user", "content": "hi"}],
                    "gen_ai.output_messages": [
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "hello "},
                                {"type": "text", "text": "world"},
                            ],
                        }
                    ],
                },
                "start_time_unix_nano": 1,
                "end_time_unix_nano": 2,
            }
        ]
        ec = _build_conversation_eval_case(spans)
        assert "hello" in (ec.output or "")
        assert "world" in (ec.output or "")
