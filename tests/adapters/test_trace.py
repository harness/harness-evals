"""Trace adapter tests: spans_to_eval_case parity across semconv,
langfuse-marker, and span_type dialects, plus the unscorable-content signal."""

import json

from harness_evals.adapters.trace import (
    SpanType,
    classify_span,
    spans_to_eval_case,
    spans_to_eval_case_for_span,
)

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
