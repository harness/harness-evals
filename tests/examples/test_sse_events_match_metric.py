"""Tests for the example sse_events_match metric."""

import pytest
from examples.sse_events_match_metric import SseEventsMatchMetric

from harness_evals.core.eval_case import EvalCase


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
def test_sse_events_match_labels_and_details_describe_assertions() -> None:
    metric = SseEventsMatchMetric(
        checks=[
            {"event": "assistant_tool_request", "path": "$.v[*]", "match": [{"path": "$.name", "contains": "harness_list"}]},
            {"event": "assistant_tool_request", "exists": True},
            {"event": "assistant_message", "exists": True},
        ],
        threshold=1.0,
    )
    eval_case = EvalCase(
        input="list pipelines",
        output="done",
        metadata={
            "sse_events": {
                "assistant_tool_request": [{"v": [{"name": "mcp__harness__harness_list", "arguments": {}}]}],
                "assistant_message": [{"text": "here you go"}],
            }
        },
    )

    score = metric.measure(eval_case)

    assert score.passed
    checks = score.metadata["checks"]
    match_check = checks[0]
    assert "correlated fields" in match_check["label"]
    assert "name contains 'harness_list'" in match_check["label"]
    assert "matched correlated fields: name contains 'harness_list'" in match_check["detail"]

    exists_check = checks[1]
    assert exists_check["label"] == "assistant_tool_request.exists=True"
    assert exists_check["detail"] == "present=True want=True"
