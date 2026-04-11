import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.metrics.structural.structural_similarity import StructuralSimilarityMetric


@pytest.mark.unit
def test_structural_similarity_raw_json():
    metric = StructuralSimilarityMetric(format="json", level="raw", threshold=0.9)
    ec = EvalCase(
        input="",
        expected={"a": 1, "b": 2},
        output={"b": 2, "a": 1},
    )
    score = metric.measure(ec)
    assert score.passed
    assert score.value == 1.0


@pytest.mark.unit
def test_structural_similarity_structural_yaml():
    metric = StructuralSimilarityMetric(
        format="yaml",
        level="structural",
        ignore_keys=["id", "name"],
        extra_keys="penalize",
        threshold=0.8,
    )
    
    expected = "id: 123\nname: test\nvalue: 100"
    actual = "id: 456\nname: changed\nvalue: 100\nextra: 50"
    
    ec = EvalCase(input="", expected=expected, output=actual)
    score = metric.measure(ec)
    
    # "id" and "name" are ignored, so they match perfectly.
    # But there's an extra key ("extra"), and extra_keys="penalize".
    # Expected penalty is 0.8 * deep_distance match.
    # Because extra item also reduces the base distance match to ~0.916,
    # the final score is ~0.733
    assert score.value == pytest.approx(0.733, abs=0.01)
    assert not score.passed


@pytest.mark.unit
def test_structural_similarity_schema_json():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    metric = StructuralSimilarityMetric(
        format="json",
        level="schema_validated",
        schema_validator={"type": "json_schema", "schema": schema},
        threshold=0.9,
    )
    
    # Valid schema, perfectly matches expected
    ec1 = EvalCase(input="", expected={"name": "test"}, output={"name": "test"})
    score1 = metric.measure(ec1)
    assert score1.passed
    assert score1.value == 1.0
    
    # Invalid schema (missing 'name')
    ec2 = EvalCase(input="", expected={"name": "test"}, output={"value": 100})
    score2 = metric.measure(ec2)
    assert not score2.passed
    assert score2.value == 0.0
    assert "Schema validation failed" in score2.reason


@pytest.mark.unit
def test_structural_similarity_online_mode_fallback():
    # In online mode, the output field might resolve to null because it's a string, 
    # but it should fall back to the raw string.
    metric = StructuralSimilarityMetric(
        format="yaml",
        level="raw",
        expected_field="$.expected.yaml",
        output_field="$.output.yaml",
    )
    
    ec = EvalCase(
        input="",
        expected={"yaml": "key: val"},
        output="key: val"  # Missing {"yaml": ...} wrapper like online mode
    )
    
    score = metric.measure(ec)
    assert score.passed
    assert score.value == 1.0
