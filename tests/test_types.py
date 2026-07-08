"""Tests for core types: ToolCall and Message."""

import pytest

from harness_evals.core.types import Message, ToolCall


@pytest.mark.unit
class TestToolCall:
    def test_minimal(self):
        tc = ToolCall(name="search")
        assert tc.name == "search"
        assert tc.input is None
        assert tc.output is None

    def test_full(self):
        tc = ToolCall(name="search", input={"q": "hello"}, output={"results": []})
        assert tc.input == {"q": "hello"}
        assert tc.output == {"results": []}

    def test_to_dict_omits_none(self):
        tc = ToolCall(name="search")
        d = tc.to_dict()
        assert d == {"name": "search"}
        assert "input" not in d

    def test_to_dict_full(self):
        tc = ToolCall(name="search", input={"q": "foo"}, output="result")
        d = tc.to_dict()
        assert d == {"name": "search", "input": {"q": "foo"}, "output": "result"}

    def test_from_dict_minimal(self):
        tc = ToolCall.from_dict({"name": "search"})
        assert tc.name == "search"
        assert tc.input is None

    def test_from_dict_full(self):
        tc = ToolCall.from_dict({"name": "search", "input": {"q": "foo"}, "output": "result"})
        assert tc.name == "search"
        assert tc.input == {"q": "foo"}
        assert tc.output == "result"

    def test_roundtrip(self):
        tc = ToolCall(name="read", input={"path": "/a"}, output={"content": "hello"})
        assert ToolCall.from_dict(tc.to_dict()) == tc


@pytest.mark.unit
class TestMessage:
    def test_minimal(self):
        msg = Message(role="user")
        assert msg.role == "user"
        assert msg.content is None
        assert msg.tool_calls is None

    def test_with_content(self):
        msg = Message(role="assistant", content="Hello!")
        assert msg.content == "Hello!"

    def test_with_tool_calls(self):
        msg = Message(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(name="search", input={"q": "test"})],
        )
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "search"

    def test_to_dict_minimal(self):
        msg = Message(role="user")
        d = msg.to_dict()
        assert d == {"role": "user"}

    def test_to_dict_full(self):
        msg = Message(
            role="assistant",
            content="hi",
            tool_calls=[ToolCall(name="search")],
        )
        d = msg.to_dict()
        assert d == {
            "role": "assistant",
            "content": "hi",
            "tool_calls": [{"name": "search"}],
        }

    def test_from_dict_minimal(self):
        msg = Message.from_dict({"role": "user"})
        assert msg.role == "user"
        assert msg.content is None

    def test_from_dict_with_tool_calls(self):
        msg = Message.from_dict(
            {
                "role": "assistant",
                "content": "I'll search for that.",
                "tool_calls": [{"name": "search", "input": {"q": "test"}}],
            }
        )
        assert msg.role == "assistant"
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "search"

    def test_from_dict_none_tool_calls(self):
        msg = Message.from_dict({"role": "user", "content": "hi", "tool_calls": None})
        assert msg.tool_calls is None

    def test_roundtrip(self):
        msg = Message(
            role="assistant",
            content="Here are results",
            tool_calls=[ToolCall(name="search", input={"q": "foo"}, output="bar")],
        )
        assert Message.from_dict(msg.to_dict()) == msg

    def test_message_operational_fields_default_none(self):
        msg = Message(role="user", content="hi")
        assert msg.latency_ms is None
        assert msg.token_count is None
        assert msg.cost_usd is None

    def test_message_operational_fields_set(self):
        msg = Message(role="assistant", content="done", latency_ms=120.5, token_count=42, cost_usd=0.0003)
        assert msg.latency_ms == 120.5
        assert msg.token_count == 42
        assert msg.cost_usd == 0.0003

    def test_message_to_dict_includes_operational_when_set(self):
        msg = Message(role="assistant", content="hi", latency_ms=100.0)
        d = msg.to_dict()
        assert d["latency_ms"] == 100.0
        assert "token_count" not in d  # None omitted

    def test_message_from_dict_roundtrip_with_operational(self):
        msg = Message(role="assistant", content="resp", latency_ms=55.0, token_count=10, cost_usd=0.001)
        assert Message.from_dict(msg.to_dict()) == msg

    def test_message_from_dict_backward_compat_no_operational(self):
        msg = Message.from_dict({"role": "user", "content": "hello"})
        assert msg.latency_ms is None
        assert msg.token_count is None

    def test_message_rag_fields_default_none(self):
        msg = Message(role="assistant", content="answer")
        assert msg.retrieval_context is None
        assert msg.expected is None

    def test_message_rag_fields_set(self):
        msg = Message(
            role="assistant",
            content="answer",
            retrieval_context=["chunk1", "chunk2"],
            expected="the expected answer",
        )
        assert msg.retrieval_context == ["chunk1", "chunk2"]
        assert msg.expected == "the expected answer"

    def test_message_to_dict_includes_rag_when_set(self):
        msg = Message(role="assistant", content="a", retrieval_context=["c1"], expected="e")
        d = msg.to_dict()
        assert d["retrieval_context"] == ["c1"]
        assert d["expected"] == "e"

    def test_message_to_dict_omits_rag_when_none(self):
        msg = Message(role="assistant", content="a")
        d = msg.to_dict()
        assert "retrieval_context" not in d
        assert "expected" not in d

    def test_message_from_dict_reads_rag_fields(self):
        msg = Message.from_dict({"role": "assistant", "content": "a", "retrieval_context": ["c1"], "expected": "e"})
        assert msg.retrieval_context == ["c1"]
        assert msg.expected == "e"

    def test_message_roundtrip_with_rag_fields(self):
        msg = Message(
            role="assistant",
            content="answer",
            retrieval_context=["chunk1", "chunk2"],
            expected="the expected answer",
        )
        assert Message.from_dict(msg.to_dict()) == msg

    def test_message_from_dict_backward_compat_no_rag(self):
        msg = Message.from_dict({"role": "user", "content": "hello"})
        assert msg.retrieval_context is None
        assert msg.expected is None
