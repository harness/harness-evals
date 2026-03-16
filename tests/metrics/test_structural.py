"""Tests for structural metrics."""

import pytest

from harness_evals.core.test_case import TestCase
from harness_evals.metrics.structural import JsonDiffMetric, SchemaValidationMetric


@pytest.mark.unit
class TestJsonDiff:
    def test_identical(self, json_test_case):
        score = JsonDiffMetric().measure(json_test_case)
        assert score.value == 1.0
        assert score.success

    def test_different(self):
        tc = TestCase(
            input="q",
            actual_output={"a": 1},
            expected_output={"a": 2, "b": 3},
        )
        score = JsonDiffMetric(threshold=0.5).measure(tc)
        assert score.value < 1.0

    def test_json_strings(self):
        tc = TestCase(
            input="q",
            actual_output='{"key": "value"}',
            expected_output='{"key": "value"}',
        )
        score = JsonDiffMetric().measure(tc)
        assert score.value == 1.0

    def test_invalid_json(self):
        tc = TestCase(input="q", actual_output="not json", expected_output='{"a": 1}')
        score = JsonDiffMetric().measure(tc)
        assert not score.success
        assert "JSON parse" in score.reason


@pytest.mark.unit
class TestSchemaValidation:
    def test_valid(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        tc = TestCase(input="q", actual_output={"name": "test"}, expected_output=schema)
        assert SchemaValidationMetric().measure(tc).success

    def test_invalid(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        tc = TestCase(input="q", actual_output={"count": 42}, expected_output=schema)
        score = SchemaValidationMetric().measure(tc)
        assert not score.success
        assert "Validation failed" in score.reason
