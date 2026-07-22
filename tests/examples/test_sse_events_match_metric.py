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
