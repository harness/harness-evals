"""Tests for structural metrics."""

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.metrics.structural import JsonDiffMetric, SchemaValidationMetric


@pytest.mark.unit
class TestJsonDiff:
    def test_identical(self, json_eval_case):
        score = JsonDiffMetric().measure(json_eval_case)
        assert score.value == 1.0
        assert score.passed

    def test_different(self):
        ec = EvalCase(
            input="q",
            output={"a": 1},
            expected={"a": 2, "b": 3},
        )
        score = JsonDiffMetric(threshold=0.5).measure(ec)
        assert score.value < 1.0

    def test_json_strings(self):
        ec = EvalCase(
            input="q",
            output='{"key": "value"}',
            expected='{"key": "value"}',
        )
        score = JsonDiffMetric().measure(ec)
        assert score.value == 1.0

    def test_invalid_json(self):
        ec = EvalCase(input="q", output="not json", expected='{"a": 1}')
        score = JsonDiffMetric().measure(ec)
        assert not score.passed
        assert "JSON parse" in score.reason


@pytest.mark.unit
class TestSchemaValidation:
    def test_valid(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        ec = EvalCase(input="q", output={"name": "test"})
        assert SchemaValidationMetric(schema=schema).measure(ec).passed

    def test_invalid(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        ec = EvalCase(input="q", output={"count": 42})
        score = SchemaValidationMetric(schema=schema).measure(ec)
        assert not score.passed
        assert "Validation failed" in score.reason
