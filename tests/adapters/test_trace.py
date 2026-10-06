"""Trace adapter tests: spans_to_eval_case parity across semconv,
langfuse-marker, and span_type dialects, plus the unscorable-content signal."""

import json

import pytest

from harness_evals.adapters.trace import (
    SpanType,
    classify_span,
    extract_span_subtree,
    spans_to_eval_case,
    spans_to_eval_case_for_span,
)
from harness_evals.metrics.conversation.tool_use import ToolUseMetric
from tests.conftest import MockLLM

TS = "2026-08-03T10:00:00Z"


def _span(span_id, parent="", span_type="", attrs=None, name="", in_toks=None, out_toks=None, ts=TS):
    return {
        "span_id": span_id,
        "parent_span_id": parent,
        "start_timestamp": ts,
        "span_type": span_type,
        "service_name": "agent-service",
        "attributes": json.dumps(attrs or {}),
        "input_tokens": in_toks,
        "output_tokens": out_toks,
        "span_name": name,
    }


def _semconv_root():
    return _span(
        "root",
        attrs={
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.input.messages": json.dumps(
                [{"role": "user", "parts": [{"type": "text", "text": "summarize this"}]}]
            ),
            "gen_ai.agent.name": "qa-agent",
            "gen_ai.request.model": "gpt-4o",
        },
    )


def _semconv_llm(output_text, span_id="llm1", parent="root"):
    return _span(
        span_id,
        parent=parent,
        attrs={
            "gen_ai.operation.name": "chat",
            "gen_ai.output.messages": json.dumps(
                [{"role": "assistant", "parts": [{"type": "text", "text": output_text}]}]
            ),
            "gen_ai.usage.input_tokens": 10,
            "gen_ai.usage.output_tokens": 5,
        },
    )


def _semconv_tool(span_id="tool1", parent="root"):
    return _span(
        span_id,
        parent=parent,
        name="execute_tool search",
        attrs={
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.input.messages": json.dumps(
                [{"role": "assistant", "parts": [{"type": "tool_call", "name": "search", "arguments": {"q": "x"}}]}]
            ),
            "gen_ai.output.messages": json.dumps(
                [{"role": "tool", "parts": [{"type": "tool_call_response", "id": "c1", "result": "42"}]}]
            ),
        },
    )


def _langfuse_trace():
    return [
        _span(
            "root",
            attrs={
                "langfuse.observation.type": "agent",
                "langfuse.observation.input": json.dumps(
                    {"messages": [{"role": "user", "content": "Review this pull request"}]}
                ),
            },
        ),
        _span(
            "llm1",
            parent="root",
            attrs={
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
        ),
        _span(
            "tool1",
            parent="root",
            name="execute_tool Read",
            attrs={
                "langfuse.observation.type": "tool",
                "gen_ai.tool.name": "Read",
                "langfuse.observation.input": json.dumps({"path": "app.py"}),
                "langfuse.observation.output": json.dumps({"content": "print('hello')"}),
            },
        ),
        _span(
            "llm2",
            parent="root",
            attrs={
                "langfuse.observation.type": "generation",
                "langfuse.observation.output": json.dumps(
                    {"role": "assistant", "content": [{"type": "text", "text": "No issues found."}]}
                ),
            },
        ),
    ]


class TestClassifyDialects:
    def test_semconv_operations(self):
        assert classify_span(_semconv_root()) == SpanType.AGENT_ROOT
        assert classify_span(_semconv_llm("x")) == SpanType.LLM_TURN
        assert classify_span(_semconv_tool()) == SpanType.TOOL_CALL

    def test_langfuse_markers(self):
        root = _span("r", attrs={"langfuse.observation.type": "agent"})
        gen = _span("g", parent="r", attrs={"langfuse.observation.type": "generation"})
        tool = _span("t", parent="r", attrs={"langfuse.observation.type": "tool"})
        assert classify_span(root) == SpanType.AGENT_ROOT
        assert classify_span(gen) == SpanType.LLM_TURN
        assert classify_span(tool) == SpanType.TOOL_CALL

    def test_span_type_column_fallback(self):
        assert classify_span(_span("r", span_type="agent")) == SpanType.AGENT_ROOT
        assert classify_span(_span("t", parent="r", span_type="tool")) == SpanType.TOOL_CALL
        assert classify_span(_span("l", parent="r", span_type="llm")) == SpanType.LLM_TURN

    def test_root_fallback(self):
        assert classify_span(_span("r")) == SpanType.AGENT_ROOT
        assert classify_span(_span("c", parent="r")) == SpanType.OTHER

    def test_semconv_wins_over_langfuse(self):
        span = _span(
            "r",
            attrs={
                "gen_ai.operation.name": "chat",
                "langfuse.observation.type": "agent",
            },
        )
        assert classify_span(span) == SpanType.LLM_TURN

    def test_dict_attributes_accepted(self):
        # HQL-sourced spans carry parsed dict attributes (no JSON string)
        span = dict(_semconv_root())
        span["attributes"] = json.loads(span["attributes"])
        assert classify_span(span) == SpanType.AGENT_ROOT


class TestEvalCaseParity:
    def test_multi_turn_with_tool_calls(self):
        spans = [
            _semconv_root(),
            _semconv_llm("first answer"),
            _semconv_tool(),
            _semconv_llm("final answer", span_id="llm2"),
        ]
        case, warnings = spans_to_eval_case(spans)
        assert case.input == "summarize this"
        assert case.output == "final answer"
        assert case.tool_calls is not None and case.tool_calls[0].name == "search"
        assert case.tool_calls[0].output == "42"
        assert case.messages is not None
        assert case.metadata["source"] == "online_eval"
        assert case.metadata["agent_name"] == "qa-agent"
        assert case.metadata["model"] == "gpt-4o"
        assert case.token_count == 30
        assert not any("No LLM output" in w for w in warnings)

    def test_langfuse_dialect_trace_adapts(self):
        root = _span(
            "r",
            attrs={
                "langfuse.observation.type": "agent",
                "gen_ai.input.messages": json.dumps(
                    [{"role": "user", "parts": [{"type": "text", "text": "deploy it"}]}]
                ),
                "langfuse.observation.name": "harness_agent_run",
            },
        )
        gen = _span(
            "g",
            parent="r",
            attrs={
                "langfuse.observation.type": "generation",
                "gen_ai.output.messages": json.dumps(
                    [{"role": "assistant", "parts": [{"type": "text", "text": "done"}]}]
                ),
            },
        )
        case, _ = spans_to_eval_case([root, gen])
        assert case.input == "deploy it"
        assert case.output == "done"
        assert case.metadata["agent_name"] == "harness_agent_run"

    def test_token_columns_win_over_attrs(self):
        llm = _semconv_llm("x")
        llm["input_tokens"] = 99
        case, _ = spans_to_eval_case([_semconv_root(), llm])
        assert case.input_tokens == 99

    def test_zero_token_columns_win_over_attrs(self):
        first = _semconv_llm("first")
        first["input_tokens"] = 0
        first["output_tokens"] = 0
        second = _semconv_llm("second", span_id="llm2")
        second["input_tokens"] = 1
        second["output_tokens"] = 2

        case, _ = spans_to_eval_case([_semconv_root(), first, second])

        assert case.input_tokens == 1
        assert case.output_tokens == 2
        assert case.token_count == 3

    def test_empty_output_marks_unscorable(self):
        proxy = _span("p", attrs={"gen_ai.operation.name": "chat"})
        case, warnings = spans_to_eval_case([proxy])
        assert case.output == ""
        assert any("No LLM output" in w for w in warnings)

    def test_span_scope_subtree(self):
        spans = [_semconv_root(), _semconv_llm("a"), _semconv_tool(), _semconv_llm("b", span_id="llm2")]
        case, _ = spans_to_eval_case_for_span(spans, "llm1")
        assert case.output == "a"
        assert case.metadata["span_id"] == "llm1"

    def test_python_repr_messages_parsed(self):
        span = _span(
            "l",
            parent="root",
            attrs={
                "gen_ai.operation.name": "chat",
                "gen_ai.output.messages": "[{'role': 'assistant', 'parts': [{'type': 'text', 'text': 'hi'}]}]",
            },
        )
        case, _ = spans_to_eval_case([_semconv_root(), span])
        assert case.output == "hi"

    def test_typed_text_parts_with_content_key(self):
        root = _span(
            "root",
            attrs={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.input.messages": json.dumps(
                    [{"role": "user", "parts": [{"type": "text", "content": "create a pipeline"}]}]
                ),
            },
        )
        llm = _span(
            "llm",
            parent="root",
            attrs={
                "gen_ai.operation.name": "chat",
                "gen_ai.output.messages": json.dumps(
                    [{"role": "assistant", "parts": [{"type": "text", "content": "Pipeline created"}]}]
                ),
            },
        )
        case, warnings = spans_to_eval_case([root, llm])
        assert case.input == "create a pipeline"
        assert case.output == "Pipeline created"
        assert not any("No LLM output" in warning for warning in warnings)

    def test_typed_text_parts_ignore_non_string_content(self):
        span = _span(
            "llm",
            attrs={
                "gen_ai.operation.name": "chat",
                "gen_ai.output.messages": json.dumps(
                    [
                        {
                            "role": "assistant",
                            "parts": [
                                {"type": "text", "content": None},
                                {"type": "text", "content": {"unexpected": True}},
                                {"type": "text", "content": "valid"},
                            ],
                        }
                    ]
                ),
            },
        )
        case, _ = spans_to_eval_case([span])
        assert case.output == "valid"

    def test_configured_aggregate_tool_calls_are_adapted(self):
        root = _semconv_root()
        root["attributes"] = json.dumps(
            {
                **json.loads(root["attributes"]),
                "custom.tool_calls": {
                    "_type": "ArrayValue",
                    "values": [
                        {"_type": "StringValue", "value": "Read"},
                        {"_type": "StringValue", "value": "Write"},
                    ],
                },
            }
        )

        case, _ = spans_to_eval_case(
            [root, _semconv_llm("complete")],
            aggregate_tool_call_attribute_keys=("custom.tool_calls",),
        )

        assert [tool_call.name for tool_call in case.tool_calls or []] == ["Read", "Write"]
        assert case.messages is not None
        assert any(message.tool_calls for message in case.messages)

    def test_configured_aggregate_tool_call_decodes_otlp_key_value_list(self):
        root = _semconv_root()
        root["attributes"] = json.dumps(
            {
                **json.loads(root["attributes"]),
                "custom.tool_calls": {
                    "kvlistValue": {
                        "values": [
                            {"key": "name", "value": {"stringValue": "Read"}},
                            {
                                "key": "arguments",
                                "value": {
                                    "kvlistValue": {
                                        "values": [
                                            {
                                                "key": "path",
                                                "value": {"stringValue": "app.py"},
                                            }
                                        ]
                                    }
                                },
                            },
                        ]
                    }
                },
            }
        )

        case, _ = spans_to_eval_case(
            [root, _semconv_llm("complete")],
            aggregate_tool_call_attribute_keys=("custom.tool_calls",),
        )

        assert case.tool_calls is not None
        assert case.tool_calls[0].name == "Read"
        assert case.tool_calls[0].input == {"path": "app.py"}

    def test_aggregate_tool_calls_require_explicit_mapping(self):
        root = _semconv_root()
        root["attributes"] = json.dumps(
            {
                **json.loads(root["attributes"]),
                "custom.tool_calls": ["Read"],
            }
        )

        case, _ = spans_to_eval_case([root, _semconv_llm("complete")])

        assert case.tool_calls is None

    def test_aggregate_tool_calls_merge_without_duplicate_tool_spans(self):
        root = _semconv_root()
        root["attributes"] = json.dumps(
            {
                **json.loads(root["attributes"]),
                "custom.tool_calls": [{"name": "search"}, "Write"],
            }
        )

        case, _ = spans_to_eval_case(
            [root, _semconv_tool(), _semconv_llm("complete")],
            aggregate_tool_call_attribute_keys=("custom.tool_calls",),
        )

        assert [tool_call.name for tool_call in case.tool_calls or []] == ["search", "Write"]

    def test_langfuse_trace_preserves_tool_use_trajectory(self):
        case, warnings = spans_to_eval_case(_langfuse_trace())

        assert case.input == "Review this pull request"
        assert case.output == "No issues found."
        assert [message.role for message in case.messages or []] == ["user", "assistant", "tool", "assistant"]
        assert case.tool_calls is not None
        assert case.tool_calls[0].name == "Read"
        assert case.tool_calls[0].input == {"path": "app.py"}
        assert case.tool_calls[0].output == "print('hello')"
        assert not warnings

    def test_langfuse_plain_text_and_query_fields_are_preserved(self):
        spans = [
            _span(
                "root",
                attrs={
                    "langfuse.observation.type": "agent",
                    "langfuse.observation.input": json.dumps({"query": "Review this pull request"}),
                },
            ),
            _span(
                "generation",
                parent="root",
                attrs={
                    "langfuse.observation.type": "generation",
                    "langfuse.observation.output": "No issues found.",
                },
            ),
        ]

        case, warnings = spans_to_eval_case(spans)

        assert case.input == "Review this pull request"
        assert case.output == "No issues found."
        assert [message.role for message in case.messages or []] == ["user", "assistant"]
        assert not warnings

    def test_langfuse_marker_wins_over_name_heuristic(self):
        span = _span(
            "tool",
            parent="root",
            name="chat",
            attrs={"langfuse.observation.type": "tool"},
        )

        assert classify_span(span) == SpanType.TOOL_CALL

    def test_tool_name_column_populates_tool_call(self):
        span = _span("tool", parent="root")
        span["tool_name"] = "search"

        case, _ = spans_to_eval_case([span])

        assert case.tool_calls is not None
        assert case.tool_calls[0].name == "search"

    @pytest.mark.asyncio
    async def test_langfuse_trace_is_usable_by_tool_use_metric(self):
        case, _ = spans_to_eval_case(_langfuse_trace())
        llm = MockLLM(default={"reasoning": "Appropriate tool use", "score": 0.9})

        score = await ToolUseMetric(llm=llm).a_measure(case)

        assert score.value == 0.9
        assert score.metadata["n_tool_calls"] == 1
        assert 'Read args={"path": "app.py"} -> print(\'hello\')' in llm.prompts[0]

    def test_extract_span_subtree_is_public_complete_and_ordered(self):
        subtree = extract_span_subtree(_langfuse_trace(), "root")
        assert [span["span_id"] for span in subtree] == ["root", "llm1", "tool1", "llm2"]

    def test_extract_span_subtree_handles_cyclic_parent_ids(self):
        spans = [
            _span("root", parent="child"),
            _span("child", parent="root"),
        ]

        subtree = extract_span_subtree(spans, "root")

        assert [span["span_id"] for span in subtree] == ["root", "child"]
