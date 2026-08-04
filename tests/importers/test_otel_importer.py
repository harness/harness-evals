"""Tests for OTELEvalCaseSource."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from harness_evals.importers.otel import OTELEvalCaseSource
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
