"""Tests for the example sse_events_match metric."""

import pytest
from examples.sse_events_match_metric import SseEventsMatchMetric

from harness_evals.core.eval_case import EvalCase


@pytest.mark.unit
def test_sse_events_match_optional_skips_missing_event() -> None:
    metric = SseEventsMatchMetric(
        checks=[
            {
                "events": ["elicitation_form", "elicitation_confirm"],
                "optional": True,
                "match": [{"path": "$.subtitle", "contains": "Bucketing"}],
            }
        ],
    )
    eval_case = EvalCase(input="ccm", output="done", metadata={"sse_events": {}})

    score = metric.measure(eval_case)

    assert score.passed
    assert score.value == 1.0


@pytest.mark.unit
def test_sse_events_match_events_across_types() -> None:
    metric = SseEventsMatchMetric(
        checks=[
            {
                "events": ["elicitation_form", "elicitation_confirm"],
                "match": [{"path": "$.entity_info.entity_type", "equals": "cost_category"}],
            }
        ],
    )
    eval_case = EvalCase(
        input="ccm",
        output="done",
        metadata={
            "sse_events": {
                "elicitation_confirm": [{"entity_info": {"entity_type": "cost_category"}}],
            }
        },
    )

    score = metric.measure(eval_case)

    assert score.passed


@pytest.mark.unit
def test_sse_events_match_failure_includes_actual_values() -> None:
    metric = SseEventsMatchMetric(
        checks=[{"event": "entity_mutation", "path": "$.resource_type", "equals": "pipeline"}],
        threshold=0.8,
    )
    eval_case = EvalCase(
        input="create pipeline",
        output="done",
        metadata={
            "sse_events": {
                "entity_mutation": [{"entity_type": "pipeline", "identifier": "testk8spipeline"}],
            }
        },
    )

    score = metric.measure(eval_case)

    assert not score.passed
    assert score.value == 0.0
    failed = next(check for check in score.metadata["checks"] if not check["passed"])
    assert "actual=[null]" in failed["detail"]
    assert "expected equals='pipeline'" in failed["detail"]


@pytest.mark.unit
def test_sse_events_match_nested_not_contains_passes() -> None:
    metric = SseEventsMatchMetric(
        checks=[
            {
                "event": "assistant_tool_result",
                "path": "$.v[*]",
                "match": [
                    {"path": "$.name", "equals": "Skill"},
                    {"path": "$.result", "not_contains": "error"},
                ],
            }
        ],
    )
    eval_case = EvalCase(
        input="load skill",
        output="done",
        metadata={
            "sse_events": {
                "assistant_tool_result": [
                    {"v": [{"name": "Skill", "result": "loaded hql-reference skill"}]},
                ]
            }
        },
    )

    score = metric.measure(eval_case)
    assert score.passed
    assert score.value == 1.0


@pytest.mark.unit
def test_sse_events_match_nested_not_contains_fails_on_error() -> None:
    metric = SseEventsMatchMetric(
        checks=[
            {
                "event": "assistant_tool_result",
                "path": "$.v[*]",
                "match": [
                    {"path": "$.name", "equals": "Skill"},
                    {"path": "$.result", "not_contains": "error"},
                ],
            }
        ],
    )
    eval_case = EvalCase(
        input="load skill",
        output="done",
        metadata={
            "sse_events": {
                "assistant_tool_result": [
                    {"v": [{"name": "Skill", "result": "skill load error: not found"}]},
                ]
            }
        },
    )

    score = metric.measure(eval_case)
    assert not score.passed
    assert score.value == 0.0


@pytest.mark.unit
def test_sse_events_match_forbidden_contains_fails_on_mcp_error() -> None:
    metric = SseEventsMatchMetric(
        checks=[
            {
                "event": "assistant_tool_result",
                "path": "$.v[*]",
                "forbidden_contains": "MCP error",
            }
        ],
    )
    eval_case = EvalCase(
        input="list resources",
        output="done",
        metadata={
            "sse_events": {
                "assistant_tool_result": [
                    {
                        "v": [
                            {
                                "name": "mcp__harness__harness_list",
                                "result": "MCP error -32603: Internal Server Error",
                            }
                        ]
                    },
                ]
            }
        },
    )

    score = metric.measure(eval_case)
    assert not score.passed
    assert score.value == 0.0


@pytest.mark.unit
def test_sse_events_match_forbidden_contains_passes_when_clean() -> None:
    metric = SseEventsMatchMetric(
        checks=[
            {
                "event": "assistant_tool_result",
                "path": "$.v[*]",
                "forbidden_contains": "MCP error",
            }
        ],
    )
    eval_case = EvalCase(
        input="list resources",
        output="done",
        metadata={
            "sse_events": {
                "assistant_tool_result": [
                    {"v": [{"name": "mcp__harness__harness_list", "result": '{"items": []}'}]},
                ]
            }
        },
    )

    score = metric.measure(eval_case)
    assert score.passed
    assert score.value == 1.0


@pytest.mark.unit
def test_sse_events_match_skill_equals_in_match_any() -> None:
    metric = SseEventsMatchMetric(
        checks=[
            {
                "event": "assistant_tool_request",
                "path": "$.v[*]",
                "match_any": [
                    [
                        {"path": "$.name", "equals": "Skill"},
                        {"path": "$.arguments.skill", "skill_equals": "hql-reference"},
                    ],
                    [
                        {"path": "$.name", "equals": "Skill"},
                        {"path": "$.arguments.skill", "skill_equals": "scs-routing"},
                    ],
                ],
            }
        ],
    )
    eval_case = EvalCase(
        input="scs",
        output="done",
        metadata={
            "sse_events": {
                "assistant_tool_request": [
                    {"v": [{"name": "Skill", "arguments": {"skill": "scs_routing"}}]},
                ]
            }
        },
    )

    score = metric.measure(eval_case)
    assert score.passed


@pytest.mark.unit
def test_sse_events_match_entity_mutation_create_update_update_ordinals() -> None:
    """Row 4 pattern: three persisted mutations at 1st, 2nd, 3rd entity_mutation."""
    metric = SseEventsMatchMetric(
        checks=[
            {
                "event": "entity_mutation",
                "occurrence": "first",
                "match": [
                    {"path": "$.action", "equals": "create"},
                    {"path": "$.identifier", "contains": "eval_maxconc_"},
                ],
            },
            {
                "event": "entity_mutation",
                "occurrence": "second",
                "match": [
                    {"path": "$.action", "equals": "update"},
                    {"path": "$.identifier", "contains": "eval_maxconc_"},
                ],
            },
            {
                "event": "entity_mutation",
                "occurrence": "third",
                "match": [
                    {"path": "$.action", "equals": "update"},
                    {"path": "$.identifier", "contains": "eval_maxconc_"},
                ],
            },
        ],
    )
    eval_case = EvalCase(
        input="max concurrency",
        output="done",
        metadata={
            "sse_events": {
                "entity_mutation": [
                    {"action": "create", "resource_type": "pipeline", "identifier": "eval_maxconc_1"},
                    {"action": "update", "resource_type": "pipeline", "identifier": "eval_maxconc_1"},
                    {"action": "update", "resource_type": "pipeline", "identifier": "eval_maxconc_1"},
                ]
            }
        },
    )

    score = metric.measure(eval_case)
    assert score.passed
    assert score.value == 1.0


@pytest.mark.unit
def test_sse_events_match_entity_mutation_third_fails_when_only_two() -> None:
    metric = SseEventsMatchMetric(
        checks=[
            {
                "event": "entity_mutation",
                "occurrence": "third",
                "match": [{"path": "$.action", "equals": "update"}],
            }
        ],
    )
    eval_case = EvalCase(
        input="max concurrency",
        output="done",
        metadata={
            "sse_events": {
                "entity_mutation": [
                    {"action": "create", "resource_type": "pipeline"},
                    {"action": "update", "resource_type": "pipeline"},
                ]
            }
        },
    )

    score = metric.measure(eval_case)
    assert not score.passed
    assert score.value == 0.0
    failed = score.metadata["checks"][0]
    assert "occurrence='third'" in failed["detail"]


@pytest.mark.unit
def test_sse_events_match_forbidden_contains_passes_when_event_absent() -> None:
    metric = SseEventsMatchMetric(
        checks=[
            {
                "event": "assistant_tool_result",
                "path": "$.v[*]",
                "forbidden_contains": "MCP error",
            }
        ],
    )
    eval_case = EvalCase(input="read only", output="done", metadata={"sse_events": {}})

    score = metric.measure(eval_case)
    assert score.passed
    assert score.value == 1.0


@pytest.mark.unit
def test_sse_events_match_ordinal_presence_only_fails_when_missing() -> None:
    metric = SseEventsMatchMetric(
        checks=[{"event": "entity_mutation", "occurrence": "third"}],
    )
    eval_case = EvalCase(
        input="mutations",
        output="done",
        metadata={
            "sse_events": {
                "entity_mutation": [
                    {"action": "create"},
                    {"action": "update"},
                ]
            }
        },
    )

    score = metric.measure(eval_case)
    assert not score.passed
    failed = score.metadata["checks"][0]
    assert "occurrence='third'" in failed["detail"]
