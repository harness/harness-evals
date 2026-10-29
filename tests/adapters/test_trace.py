"""Trace adapter tests: spans_to_eval_case parity across semconv,
langfuse-marker, and span_type dialects, plus the unscorable-content signal."""

import json
from dataclasses import replace
from typing import Any

import pytest

import harness_evals.adapters.trace as trace_adapter
from harness_evals.adapters.trace import (
    SpanType,
    classify_span,
    extract_span_subtree,
    spans_to_eval_case,
    spans_to_eval_case_for_span,
)
from harness_evals.applicability import TRACE_INPUTS
from harness_evals.core.metric import BaseMetric, Dimension, MetricInput, MetricInputs
from harness_evals.core.runner import a_evaluate, evaluate
from harness_evals.core.score import Score
from harness_evals.core.types import ToolCall
from harness_evals.metrics.agent.plan_adherence import PlanAdherenceMetric
from harness_evals.metrics.conversation.tool_use import ToolUseMetric
from harness_evals.metrics.mcp.trace_completeness import MCPTraceCompletenessMetric
from harness_evals.metrics.operational.cost_efficiency import CostEfficiencyMetric
from harness_evals.metrics.operational.latency import LatencyMetric
from harness_evals.metrics.operational.token_cost import TokenCostMetric
from harness_evals.metrics.operational.turn_latency import TurnLatencyMetric
from harness_evals.metrics.operational.turn_token_cost import TurnTokenCostMetric
from harness_evals.metrics.structural.schema_validation import SchemaValidationMetric
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


class TestTelemetryValidation:
    @pytest.mark.parametrize("attributes", [[], [1], True, 5, "null", "[]", "[1]", '"text"', "true", "5"])
    def test_non_object_attributes_do_not_break_classification_or_adaptation(self, attributes: Any) -> None:
        span = {**_span("llm", parent="root", span_type="llm"), "attributes": attributes}

        assert classify_span(span) == SpanType.LLM_TURN
        case, _ = spans_to_eval_case([span])

        assert case.output == ""
        assert case.token_count is None
        assert case.cost_usd is None
        assert span["attributes"] == attributes

    @pytest.mark.parametrize("selective", [False, True], ids=["full", "selective"])
    @pytest.mark.parametrize(
        "invalid",
        [True, False, -1, -0.5, 2.5, float("nan"), float("inf"), -float("inf"), "NaN", "²", [], {}],
    )
    def test_invalid_token_dimension_does_not_discard_valid_usage(self, selective: bool, invalid: Any) -> None:
        turn = _semconv_llm("answer")
        turn["input_tokens"] = invalid
        metrics = [TokenCostMetric(), TurnTokenCostMetric()] if selective else None

        case, _ = spans_to_eval_case([turn], metrics=metrics)

        assert case.token_count == 5
        assert case.messages[-1].token_count == 5
        if not selective:
            assert case.input_tokens is None
            assert case.output_tokens == 5

    @pytest.mark.parametrize("selective", [False, True], ids=["full", "selective"])
    @pytest.mark.parametrize("tokens", [0, "0", 0.0, 5, "5", 5.0])
    def test_integral_token_values_and_observed_zero_are_preserved(
        self, selective: bool, tokens: int | float | str
    ) -> None:
        turn = _semconv_llm("answer")
        turn["input_tokens"] = tokens
        turn["output_tokens"] = tokens
        metrics = [TokenCostMetric(), TurnTokenCostMetric()] if selective else None

        case, _ = spans_to_eval_case([turn], metrics=metrics)

        assert case.token_count == int(tokens) * 2
        assert case.messages[-1].token_count == int(tokens) * 2

    @pytest.mark.parametrize("selective", [False, True], ids=["full", "selective"])
    @pytest.mark.parametrize("invalid", [True, False, float("nan"), float("inf"), -float("inf"), "NaN", "Infinity"])
    def test_invalid_cost_does_not_override_observed_root_cost(self, selective: bool, invalid: Any) -> None:
        root = _span("root", attrs={"gen_ai.usage.cost": 0.07})
        turn = _span("llm", parent="root", span_type="llm", attrs={"gen_ai.usage.cost": invalid})

        case, _ = spans_to_eval_case([root, turn], metrics=[CostEfficiencyMetric()] if selective else None)

        assert case.cost_usd == 0.07

    @pytest.mark.parametrize("selective", [False, True], ids=["full", "selective"])
    @pytest.mark.parametrize("invalid", [True, False, float("nan"), float("inf"), -1, "invalid"])
    def test_invalid_duration_remains_unobserved(self, selective: bool, invalid: Any) -> None:
        root = {**_semconv_root(), "duration_ms": invalid}
        turn = {**_semconv_llm("answer"), "duration_ms": invalid}
        metrics = [LatencyMetric(), TurnLatencyMetric()] if selective else None

        case, _ = spans_to_eval_case([root, turn], metrics=metrics)

        assert case.latency_ms is None
        assert case.messages[-1].latency_ms is None

    @pytest.mark.parametrize("selective", [False, True], ids=["full", "selective"])
    def test_zero_duration_is_observed(self, selective: bool) -> None:
        root = {**_semconv_root(), "duration_ms": 0}
        case, _ = spans_to_eval_case([root], metrics=[LatencyMetric()] if selective else None)

        assert case.latency_ms == 0

    @pytest.mark.parametrize("scope", ["trace", "span"])
    @pytest.mark.parametrize("selective", [False, True], ids=["full", "selective"])
    def test_latency_uses_latest_end_not_latest_start(self, scope: str, selective: bool) -> None:
        root = {**_semconv_root(), "duration_ms": 10000}
        turn = {
            **_semconv_llm("answer"),
            "start_timestamp": "2026-08-03T10:00:01Z",
            "duration_ms": 100,
        }
        metrics = [LatencyMetric()] if selective else None
        if scope == "span":
            case, _ = spans_to_eval_case_for_span([turn, root], "root", metrics=metrics)
        else:
            case, _ = spans_to_eval_case([turn, root], metrics=metrics)

        assert case.latency_ms == pytest.approx(10000)
        assert LatencyMetric(max_ms=5000).measure(case).value == 0

    @pytest.mark.parametrize("timestamp", [None, "invalid", True, float("nan"), float("inf")])
    def test_invalid_start_does_not_inflate_trace_latency(self, timestamp: Any) -> None:
        root = {**_semconv_root(), "duration_ms": 10000}
        turn = {**_semconv_llm("answer"), "start_timestamp": timestamp, "duration_ms": 100}

        case, _ = spans_to_eval_case([root, turn], metrics=[LatencyMetric()])

        assert case.latency_ms == pytest.approx(10000)

    @pytest.mark.parametrize("scope", ["trace", "span"])
    @pytest.mark.parametrize("selective", [False, True], ids=["full", "selective"])
    @pytest.mark.parametrize("timestamp", [None, "invalid", True, float("nan"), float("inf")])
    def test_untimed_root_duration_bounds_child_envelope(self, scope: str, selective: bool, timestamp: Any) -> None:
        root = {**_semconv_root(), "start_timestamp": timestamp, "duration_ms": 10000}
        turn = {**_semconv_llm("answer"), "duration_ms": 100}
        metrics = [LatencyMetric()] if selective else None
        if scope == "span":
            case, _ = spans_to_eval_case_for_span([root, turn], "root", metrics=metrics)
        else:
            case, _ = spans_to_eval_case([root, turn], metrics=metrics)

        assert case.latency_ms == pytest.approx(10000)
        assert LatencyMetric(max_ms=5000).measure(case).value == 0

    def test_unobserved_durations_do_not_become_zero(self) -> None:
        root = _semconv_root()
        turn = {**_semconv_llm("answer"), "start_timestamp": "2026-08-03T10:00:01Z"}

        case, _ = spans_to_eval_case([root, turn], metrics=[LatencyMetric()])

        assert case.latency_ms is None

    def test_unix_epoch_is_a_valid_start(self) -> None:
        root = {**_semconv_root(), "start_timestamp": 0, "duration_ms": 10000}
        turn = {**_semconv_llm("answer"), "start_timestamp": 1, "duration_ms": 100}

        case, _ = spans_to_eval_case([root, turn], metrics=[LatencyMetric()])

        assert case.latency_ms == pytest.approx(10000)


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

    def test_requested_output_warns_when_missing(self):
        proxy = _span("p", attrs={"gen_ai.operation.name": "chat"})
        case, warnings = spans_to_eval_case([proxy])
        assert case.output == ""
        assert any("No LLM output" in w for w in warnings)

    def test_selected_metrics_materialize_only_required_trace_data(self):
        spans = [_semconv_root(), _semconv_llm('{"status": "ok"}'), _semconv_tool()]

        case, warnings = spans_to_eval_case(
            spans,
            metrics=[SchemaValidationMetric(schema={"type": "object"})],
        )

        assert case.output == '{"status": "ok"}'
        assert case.input == ""
        assert case.messages is None
        assert case.tool_calls is None
        assert case.latency_ms is None
        assert case.token_count is None
        assert case.metadata is None
        assert warnings == []

    def test_selected_latency_metric_avoids_content_materialization(self, monkeypatch):
        def unexpected_extraction(*args, **kwargs):
            pytest.fail("latency-only adaptation must not extract span content")

        for name in ("_extract_input_from_span", "_extract_output_from_span", "_extract_tool_from_span"):
            monkeypatch.setattr(trace_adapter, name, unexpected_extraction)
        spans = [_semconv_root(), _semconv_llm("answer")]
        spans[1]["duration_ms"] = 15
        spans[1]["start_timestamp"] = "2026-08-03T10:00:01Z"

        case, warnings = spans_to_eval_case(spans, metrics=[LatencyMetric()])

        assert case.input == ""
        assert case.output == ""
        assert case.messages is None
        assert case.tool_calls is None
        assert case.latency_ms == 1015.0
        assert case.token_count is None
        assert case.metadata is None
        assert warnings == []

    async def test_latency_only_case_can_be_scored_without_output(self):
        root = _semconv_root()
        root["duration_ms"] = 15
        metric = LatencyMetric()
        case, warnings = spans_to_eval_case([root], metrics=[metric])

        assert case.output == ""
        assert warnings == []
        for scores in (evaluate(case, [metric]), await a_evaluate(case, [metric])):
            assert len(scores) == 1
            assert scores[0].passed
            assert scores[0].value > 0

    @pytest.mark.parametrize(
        "metric",
        [
            LatencyMetric(),
            TokenCostMetric(),
            CostEfficiencyMetric(),
            TurnLatencyMetric(),
            TurnTokenCostMetric(),
            SchemaValidationMetric(schema={"type": "object"}),
            MCPTraceCompletenessMetric(expected_trace=[ToolCall(name="search", input={"q": "x"})]),
        ],
        ids=lambda metric: metric.name,
    )
    async def test_selected_fields_and_scores_match_full_adaptation(self, metric):
        root = _semconv_root()
        root["duration_ms"] = 30
        turn = _semconv_llm('{"status": "ok"}')
        turn["duration_ms"] = 15
        turn["attributes"] = json.dumps({**json.loads(turn["attributes"]), "gen_ai.usage.cost": 0.01})
        spans = [root, turn, _semconv_tool()]
        full, _ = spans_to_eval_case(spans)
        selected, _ = spans_to_eval_case(spans, metrics=[metric])

        for field in metric.eval_inputs.fields:
            assert getattr(selected, field.value) == getattr(full, field.value)
        expected = metric.measure(full)
        for score in (metric.measure(selected), await metric.a_measure(selected)):
            assert replace(score, created_at=expected.created_at) == expected

    def test_selected_plan_adherence_metric_materializes_input(self):
        case, _ = spans_to_eval_case(
            [_semconv_root(), _semconv_llm("answer"), _semconv_tool()],
            metrics=[PlanAdherenceMetric(llm=MockLLM())],
        )

        assert case.input == "summarize this"
        assert case.messages is not None
        assert case.tool_calls is not None

    def test_selected_turn_metrics_materialize_per_turn_telemetry(self):
        turn = _semconv_llm("answer")
        turn["duration_ms"] = 15

        latency_case, _ = spans_to_eval_case([_semconv_root(), turn], metrics=[TurnLatencyMetric()])
        token_case, _ = spans_to_eval_case([_semconv_root(), turn], metrics=[TurnTokenCostMetric()])

        assert latency_case.messages is not None
        assert latency_case.messages[1].latency_ms == 15
        assert TurnLatencyMetric(max_ms_per_turn=15).measure(latency_case).passed
        assert token_case.messages is not None
        assert token_case.messages[1].token_count == 15
        assert TurnTokenCostMetric(max_tokens_per_turn=15).measure(token_case).value == 0.0

    def test_selected_token_metrics_keep_missing_usage_unobserved(self):
        unmetered_turn = _span(
            "llm",
            parent="root",
            attrs={
                "gen_ai.operation.name": "chat",
                "gen_ai.output.messages": json.dumps(
                    [{"role": "assistant", "parts": [{"type": "text", "text": "answer"}]}]
                ),
            },
        )

        total_case, _ = spans_to_eval_case([_semconv_root(), unmetered_turn], metrics=[TokenCostMetric()])
        turn_case, _ = spans_to_eval_case([_semconv_root(), unmetered_turn], metrics=[TurnTokenCostMetric()])

        assert total_case.token_count is None
        assert turn_case.messages is not None
        assert turn_case.messages[1].token_count is None
        assert TurnTokenCostMetric().measure(turn_case).value == 0.0

    def test_selected_token_metrics_reject_invalid_usage(self):
        invalid_turn = _semconv_llm("answer")
        invalid_turn["input_tokens"] = "not-a-number"
        invalid_turn["output_tokens"] = "also-not-a-number"

        total_case, _ = spans_to_eval_case([_semconv_root(), invalid_turn], metrics=[TokenCostMetric()])
        turn_case, _ = spans_to_eval_case([_semconv_root(), invalid_turn], metrics=[TurnTokenCostMetric()])

        assert total_case.token_count is None
        assert turn_case.messages is not None
        assert turn_case.messages[1].token_count is None

    def test_selected_token_metrics_exclude_invalid_dimensions(self):
        mixed_turn = _semconv_llm("answer")
        mixed_turn["input_tokens"] = 10
        mixed_turn["output_tokens"] = -9

        total_case, _ = spans_to_eval_case([_semconv_root(), mixed_turn], metrics=[TokenCostMetric()])
        turn_case, _ = spans_to_eval_case([_semconv_root(), mixed_turn], metrics=[TurnTokenCostMetric()])

        assert total_case.token_count == 10
        assert turn_case.messages is not None
        assert turn_case.messages[1].token_count == 10

    def test_selected_cost_metric_materializes_observed_span_cost(self):
        turn = _semconv_llm("answer")
        attrs = json.loads(turn["attributes"])
        attrs["gen_ai.usage.cost"] = "0.01"
        turn["attributes"] = json.dumps(attrs)

        case, _ = spans_to_eval_case(
            [_semconv_root(), turn],
            metrics=[CostEfficiencyMetric(max_cost_usd=0.10)],
        )

        assert case.cost_usd == 0.01
        assert CostEfficiencyMetric(max_cost_usd=0.10).measure(case).value == 0.9

    @pytest.mark.parametrize("selective", [False, True], ids=["full", "selective"])
    @pytest.mark.parametrize(
        "root_cost, child_costs, expected_cost",
        [
            pytest.param(0.07, [None], 0.07, id="root-fallback"),
            pytest.param(0.07, [0.01, 0.02], 0.03, id="sum-children-without-root"),
            pytest.param(0.07, [0.0], 0.0, id="zero-child-overrides-root"),
            pytest.param(0.07, [-1.0, 0.02], 0.02, id="skip-negative-child"),
            pytest.param(0.07, [-1.0], 0.07, id="negative-child-keeps-root-fallback"),
            pytest.param(-1.0, [0.02], 0.02, id="skip-negative-root"),
            pytest.param(-1.0, [-0.02], None, id="all-negative-is-unobserved"),
            pytest.param(None, [None], None, id="missing-is-unobserved"),
            pytest.param(0.0, [None], 0.0, id="observed-zero-root"),
            pytest.param("invalid", ["invalid"], None, id="invalid-is-unobserved"),
        ],
    )
    def test_trace_cost_aggregation(self, selective, root_cost, child_costs, expected_cost):
        spans = [_semconv_root()] + [_semconv_llm("answer", span_id=f"llm{index}") for index in range(len(child_costs))]
        for span, cost in zip(spans, [root_cost, *child_costs], strict=True):
            if cost is not None:
                attrs = json.loads(span["attributes"])
                attrs["gen_ai.usage.cost"] = cost
                span["attributes"] = json.dumps(attrs)

        case, _ = spans_to_eval_case(
            spans,
            metrics=[CostEfficiencyMetric()] if selective else None,
        )

        if expected_cost is None:
            assert case.cost_usd is None
        else:
            assert case.cost_usd == pytest.approx(expected_cost)

    def test_span_scope_subtree(self):
        spans = [_semconv_root(), _semconv_llm("a"), _semconv_tool(), _semconv_llm("b", span_id="llm2")]
        case, _ = spans_to_eval_case_for_span(spans, "llm1")
        assert case.output == "a"
        assert case.metadata["span_id"] == "llm1"

    def test_span_scope_selective_projection_keeps_span_id(self):
        case, _ = spans_to_eval_case_for_span(
            [_semconv_root(), _semconv_llm('{"status": "ok"}')],
            "llm1",
            metrics=[SchemaValidationMetric(schema={"type": "object"})],
        )

        assert case.metadata == {"span_id": "llm1", "source": "online_eval"}

    def test_selected_custom_metric_inputs_are_honored(self):
        class ToolOutputMetric(BaseMetric):
            eval_inputs = MetricInputs(required={MetricInput.OUTPUT}, optional={MetricInput.TOOL_CALLS})

            def __init__(self):
                super().__init__(name="tool_output", dimension=Dimension.CORRECTNESS)

            def measure(self, eval_case):
                return Score(name=self.name, value=1.0, threshold=self.threshold)

        case, _ = spans_to_eval_case(
            [_semconv_root(), _semconv_llm("answer"), _semconv_tool()],
            metrics=[ToolOutputMetric()],
        )

        assert case.output == "answer"
        assert case.tool_calls is not None and case.tool_calls[0].name == "search"
        assert case.input == ""
        assert case.messages is None
        assert case.metadata is None

    def test_selected_undeclared_custom_metric_materializes_every_field(self):
        class UndeclaredMetric(BaseMetric):
            def __init__(self):
                super().__init__(name="undeclared", dimension=Dimension.CORRECTNESS)

            def measure(self, eval_case):
                return Score(name=self.name, value=1.0, threshold=self.threshold)

        spans = [_semconv_root(), _semconv_llm("answer"), _semconv_tool()]

        full = spans_to_eval_case(spans)
        assert spans_to_eval_case(spans, metrics=[UndeclaredMetric()]) == full
        assert spans_to_eval_case(spans, metrics=[LatencyMetric(), UndeclaredMetric()]) == full

    def test_empty_metric_list_does_not_request_content(self, monkeypatch):
        def unexpected_extraction(*args, **kwargs):
            pytest.fail("an empty metric list must not extract span content")

        monkeypatch.setattr(trace_adapter, "_extract_output_from_span", unexpected_extraction)
        case, warnings = spans_to_eval_case([_semconv_root(), _semconv_llm("answer")], metrics=[])

        assert case.input == case.output == ""
        assert all(
            getattr(case, field.value) is None
            for field in MetricInput
            if field not in {MetricInput.INPUT, MetricInput.OUTPUT}
        )
        assert warnings == []

    def test_full_projection_populates_exactly_trace_inputs_and_metadata(self):
        turn = _semconv_llm("answer")
        turn["duration_ms"] = 15
        attrs = json.loads(turn["attributes"])
        attrs["gen_ai.usage.cost"] = "0.01"
        turn["attributes"] = json.dumps(attrs)

        root = _semconv_root()
        root["duration_ms"] = 30

        case, _ = spans_to_eval_case([root, turn, _semconv_tool()])

        populated = {field for field in MetricInput if getattr(case, field.value) not in (None, "")}
        assert populated == TRACE_INPUTS | {MetricInput.METADATA}

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

    @pytest.mark.parametrize("dialect", ["semconv", "langfuse"])
    @pytest.mark.parametrize("scope", ["trace", "span"])
    @pytest.mark.parametrize("executed", [False, True])
    @pytest.mark.parametrize("additional_call", [False, True])
    async def test_tool_evidence_does_not_depend_on_projected_messages(self, dialect, scope, executed, additional_call):
        root = _semconv_root()
        aggregate_calls = [{"name": "search", "input": {"q": "x"}}]
        if additional_call:
            aggregate_calls.append("Write")
        root["attributes"] = json.dumps({**json.loads(root["attributes"]), "custom.tool_calls": aggregate_calls})
        if dialect == "semconv":
            attrs = {
                "gen_ai.operation.name": "chat",
                "gen_ai.output.messages": [
                    {"role": "assistant", "parts": [{"type": "tool_call", "name": "search", "arguments": {"q": "x"}}]}
                ],
            }
        else:
            attrs = {
                "langfuse.observation.type": "generation",
                "langfuse.observation.output": {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "name": "search", "input": {"q": "x"}}],
                },
            }
        spans = [root, _span("intent", parent="root", attrs=attrs)]
        if executed:
            spans.append(_semconv_tool())
        metric = MCPTraceCompletenessMetric(expected_trace=[ToolCall(name="search", input={"q": "x"})])

        def adapt(metrics=None):
            kwargs = {"metrics": metrics, "aggregate_tool_call_attribute_keys": ("custom.tool_calls",)}
            if scope == "span":
                return spans_to_eval_case_for_span(spans, "root", **kwargs)[0]
            return spans_to_eval_case(spans, **kwargs)[0]

        full = adapt()
        expected = metric.measure(full)
        assert expected.value == float(executed)
        assert [call.name for call in full.tool_calls or []] == (
            (["search"] if executed else []) + (["Write"] if additional_call else [])
        )
        for metrics in (
            [metric],
            [metric, LatencyMetric()],
            [metric, TurnLatencyMetric()],
            [metric, LatencyMetric(), TurnLatencyMetric()],
        ):
            selected = adapt(metrics)
            assert selected.tool_calls == full.tool_calls
            for score in (metric.measure(selected), await metric.a_measure(selected)):
                assert replace(score, created_at=expected.created_at) == expected
            if any(MetricInput.MESSAGES in item.eval_inputs.fields for item in metrics):
                assert selected.messages == full.messages
            else:
                assert selected.messages is None

    def test_tool_only_projection_avoids_conversation_materialization(self, monkeypatch):
        def unexpected_conversation_work(*args, **kwargs):
            pytest.fail("tool-only adaptation must not build messages or per-turn telemetry")

        for name in ("Message", "_extract_input_from_span", "_span_duration_ms", "_observed_token_count"):
            monkeypatch.setattr(trace_adapter, name, unexpected_conversation_work)
        metric = MCPTraceCompletenessMetric(expected_trace=[ToolCall(name="search", input={"q": "x"})])
        case, warnings = spans_to_eval_case(
            [_semconv_root(), _semconv_llm("answer"), _semconv_tool()], metrics=[metric]
        )

        assert case.messages is None
        assert case.output == ""
        assert metric.measure(case).value == 1.0
        assert warnings == []

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
