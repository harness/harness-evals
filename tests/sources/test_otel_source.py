"""Tests for OTELSource adapter."""

import pytest

from harness_evals.sources.otel import OTELSource


@pytest.mark.unit
class TestOTELSourceFromSpanJson:
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
        ec = OTELSource.from_span_json(spans)
        assert ec.input == {"query": "hello"}
        assert ec.output == {"answer": "world"}
        assert ec.latency_ms == pytest.approx(1000.0)
        assert ec.token_count == 15
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
        ec = OTELSource.from_span_json(spans)
        assert ec.tool_calls is not None
        assert len(ec.tool_calls) == 1
        assert ec.tool_calls[0].name == "search"
        assert ec.tool_calls[0].input == {"q": "foo"}
        assert ec.tool_calls[0].output == "bar"

    def test_empty_spans(self):
        ec = OTELSource.from_span_json([])
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
        ec = OTELSource.from_span_json(spans)
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
        ec = OTELSource.from_span_json(spans)
        assert ec.messages is not None
        assert ec.messages[0].role == "assistant"
        assert ec.messages[0].content == "Hello there!"
