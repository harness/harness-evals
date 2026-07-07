"""Unit tests for target trajectory synthesis and coercion helpers."""

from __future__ import annotations

import json

import pytest

from harness_evals.core.types import Message, ToolCall
from harness_evals.targets.trajectory import (
    coerce_messages,
    coerce_tool_calls,
    reconstruct_stream_messages,
    synthesize_messages,
)


@pytest.mark.unit
def test_synthesize_two_message_envelope() -> None:
    messages = synthesize_messages("what is 6*7?", "42")
    assert messages is not None
    assert [(m.role, m.content) for m in messages] == [
        ("user", "what is 6*7?"),
        ("assistant", "42"),
    ]


@pytest.mark.unit
def test_synthesize_json_encodes_non_string_input_and_output() -> None:
    messages = synthesize_messages({"q": "hi"}, {"answer": 42})
    assert messages is not None
    assert json.loads(messages[0].content) == {"q": "hi"}
    assert json.loads(messages[1].content) == {"answer": 42}


@pytest.mark.unit
def test_synthesize_includes_tool_calls_between_user_and_output() -> None:
    tools = [ToolCall(name="search", input={"q": "cats"}, output="found")]
    messages = synthesize_messages("find cats", "done", tools)
    assert messages is not None
    assert [m.role for m in messages] == ["user", "assistant", "assistant"]
    assert messages[1].tool_calls == tools
    assert messages[1].content is None
    assert messages[2].content == "done"


@pytest.mark.unit
def test_synthesize_omits_empty_output_turn() -> None:
    # An empty output (e.g. an errored call) yields just the user turn — we
    # never fabricate an assistant turn that has no content.
    messages = synthesize_messages("hello", "")
    assert messages is not None
    assert [(m.role, m.content) for m in messages] == [("user", "hello")]


@pytest.mark.unit
def test_coerce_messages_from_dicts() -> None:
    coerced = coerce_messages([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}])
    assert coerced is not None
    assert all(isinstance(m, Message) for m in coerced)
    assert [m.content for m in coerced] == ["hi", "yo"]


@pytest.mark.unit
def test_coerce_messages_passes_through_objects_and_skips_malformed() -> None:
    existing = Message(role="user", content="hi")
    coerced = coerce_messages([existing, {"content": "no role"}, "garbage"])
    assert coerced == [existing]


@pytest.mark.unit
def test_coerce_messages_non_list_returns_none() -> None:
    assert coerce_messages("not a list") is None
    assert coerce_messages(None) is None


@pytest.mark.unit
def test_coerce_messages_empty_list_is_valid_not_malformed() -> None:
    # An explicitly-empty reported trajectory (messages: []) is a valid "no
    # turns" report, distinct from a malformed non-list/unparseable value which
    # yields None. Returning [] means the caller won't warn about coercion.
    assert coerce_messages([]) == []


@pytest.mark.unit
def test_coerce_tool_calls_from_dicts() -> None:
    coerced = coerce_tool_calls([{"name": "search", "input": {"q": "x"}, "output": "r"}])
    assert coerced is not None
    assert isinstance(coerced[0], ToolCall)
    assert coerced[0].name == "search"


@pytest.mark.unit
def test_coerce_tool_calls_skips_entries_without_name() -> None:
    assert coerce_tool_calls([{"input": {}}]) is None
    assert coerce_tool_calls("nope") is None


# ---------------------------------------------------------------------------
# reconstruct_stream_messages
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_reconstruct_interleaves_text_tool_call_result() -> None:
    decoded = [
        ("message", {"output": "let me search"}),
        ("message", {"tools": [{"name": "search", "input": {"q": "cats"}}]}),
        ("message", {"tools": [{"name": "search", "output": "found 3"}]}),
        ("message", {"output": "here are the results"}),
    ]
    messages = reconstruct_stream_messages(
        decoded, "find cats", output_path="$.output", tool_calls_path="$.tools"
    )
    assert messages is not None
    assert [(m.role, m.content) for m in messages] == [
        ("user", "find cats"),
        ("assistant", "let me search"),
        ("assistant", None),  # tool-call message
        ("tool", "found 3"),
        ("assistant", "here are the results"),
    ]
    # assistant tool-call message sits between the two text turns
    call_msg = messages[2]
    assert call_msg.tool_calls is not None
    assert call_msg.tool_calls[0].name == "search"
    assert call_msg.tool_calls[0].input == {"q": "cats"}


@pytest.mark.unit
def test_reconstruct_buffers_consecutive_text_deltas() -> None:
    decoded = [
        ("token", {"output": "Hel"}),
        ("token", {"output": "lo "}),
        ("token", {"output": "world"}),
    ]
    messages = reconstruct_stream_messages(
        decoded, "hi", output_path="$.output", tool_calls_path=None
    )
    assert messages is not None
    assert [(m.role, m.content) for m in messages] == [
        ("user", "hi"),
        ("assistant", "Hello world"),
    ]


@pytest.mark.unit
def test_reconstruct_returns_none_when_no_structure() -> None:
    # Trailing telemetry-only events carry neither output_path nor tool_calls.
    decoded = [("model_usage", {"tokens": 42}), ("done", {"status": "ok"})]
    assert reconstruct_stream_messages(
        decoded, "hi", output_path="$.output", tool_calls_path="$.tools"
    ) is None


@pytest.mark.unit
def test_reconstruct_coalesces_arguments_and_result_aliases() -> None:
    decoded = [
        ("message", {"tools": [{"name": "lookup", "arguments": {"id": 1}}]}),
        ("message", {"tools": [{"name": "lookup", "result": "ok"}]}),
    ]
    messages = reconstruct_stream_messages(
        decoded, "q", output_path="$.output", tool_calls_path="$.tools"
    )
    assert messages is not None
    call = messages[1]
    assert call.tool_calls is not None
    assert call.tool_calls[0].input == {"id": 1}
    result = messages[2]
    assert result.role == "tool"
    assert result.content == "ok"


@pytest.mark.unit
def test_reconstruct_preserves_falsey_tool_result() -> None:
    # A falsey output (0, False, [], "") must survive — only a missing (None)
    # result becomes empty content.
    for raw_output, expected_content in [(0, "0"), (False, "false"), ([], "[]"), ("", "")]:
        decoded = [("message", {"tools": [{"name": "calc", "result": raw_output}]})]
        messages = reconstruct_stream_messages(
            decoded, "q", output_path="$.output", tool_calls_path="$.tools"
        )
        assert messages is not None
        result = messages[1]
        assert result.role == "tool"
        assert result.content == expected_content
        assert result.tool_calls is not None
        assert result.tool_calls[0].output == raw_output


@pytest.mark.unit
def test_coerce_tool_calls_honors_field_aliases() -> None:
    # Extracted tool calls (messages_path / tool_calls_path) must not lose input
    # or output carried under the arguments/result aliases.
    coerced = coerce_tool_calls([{"name": "search", "arguments": {"q": "x"}, "result": "hit"}])
    assert coerced is not None
    assert coerced[0].input == {"q": "x"}
    assert coerced[0].output == "hit"


@pytest.mark.unit
def test_coerce_messages_honors_nested_tool_call_aliases() -> None:
    coerced = coerce_messages(
        [{"role": "assistant", "tool_calls": [{"name": "search", "arguments": {"q": "x"}, "result": "hit"}]}]
    )
    assert coerced is not None
    assert coerced[0].tool_calls is not None
    tc = coerced[0].tool_calls[0]
    assert tc.input == {"q": "x"}
    assert tc.output == "hit"


@pytest.mark.unit
def test_coerce_messages_returns_none_for_malformed_reported_trajectory() -> None:
    # Distinguishable-from-empty signal: callers treat None as "reported but
    # malformed" and must not synthesize over it.
    assert coerce_messages(["not", "messages"]) is None
    assert coerce_messages("nope") is None
    assert coerce_messages([{"no_role": 1}]) is None
