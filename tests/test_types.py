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
