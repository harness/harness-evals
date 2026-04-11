import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.metrics.composite.composite import CompositeMetric


@pytest.mark.unit
def test_composite_metric_basic():
    config = [
        {
            "name": "check_a",
            "weight": 0.5,
            "check": {"type": "equals", "field": "$.output.a", "expected_field": "$.expected.a"},
        },
        {
            "name": "check_b",
            "weight": 0.5,
            "check": {"type": "field_exists", "field": "$.output.b"},
        },
    ]
    
    metric = CompositeMetric(sub_scores=config, output_format="json", threshold=0.8)
    
    ec = EvalCase(
        input="",
        expected={"a": 100},
        output={"a": 100, "b": "exists"},
    )
    score = metric.measure(ec)
    
    assert score.passed
    assert score.value == 1.0
    assert score.metadata["sub_scores"]["check_a"]["value"] == 1.0
    assert score.metadata["sub_scores"]["check_b"]["value"] == 1.0


@pytest.mark.unit
def test_composite_metric_partial_failure():
    config = [
        {
            "name": "check_a",
            "weight": 0.5,
            "check": {"type": "equals", "field": "$.output.a", "value": 100},
        },
        {
            "name": "check_b",
            "weight": 0.5,
            "check": {"type": "equals", "field": "$.output.b", "value": 200},
        },
    ]
    
    metric = CompositeMetric(sub_scores=config, output_format="json", threshold=0.8)
    
    ec = EvalCase(
        input="",
        expected={},
        output={"a": 100, "b": 999},  # b fails
    )
    score = metric.measure(ec)
    
    assert not score.passed
    assert score.value == 0.5
    assert score.metadata["sub_scores"]["check_a"]["value"] == 1.0
    assert score.metadata["sub_scores"]["check_b"]["value"] == 0.0


@pytest.mark.unit
def test_composite_metric_skip_when_missing():
    config = [
        {
            "name": "check_a",
            "weight": 0.5,
            "check": {"type": "equals", "field": "$.output.a", "value": 100},
        },
        {
            "name": "check_opt",
            "weight": 0.5,
            "skip_when_missing": True,
            "check": {"type": "equals", "field": "$.output.opt", "value": 200},
        },
    ]
    
    metric = CompositeMetric(sub_scores=config, output_format="json", threshold=0.8)
    
    ec = EvalCase(
        input="",
        expected={},
        output={"a": 100},  # opt is missing
    )
    score = metric.measure(ec)
    
    assert score.passed
    assert score.value == 1.0  # check_opt skipped, check_a is 100% of effective weight
    assert score.metadata["sub_scores"]["check_opt"]["status"] == "skipped"
    assert score.metadata["effective_weights"]["check_a"] == 1.0
