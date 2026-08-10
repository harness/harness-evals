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
        assert "No structural differences" in score.reason

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
        score = SchemaValidationMetric(schema=schema).measure(ec)
        assert score.passed
        assert "conforms" in score.reason

    def test_invalid(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        ec = EvalCase(input="q", output={"count": 42})
        score = SchemaValidationMetric(schema=schema).measure(ec)
        assert not score.passed
        assert "Validation failed" in score.reason

    def test_markdown_fenced_json_object(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        ec = EvalCase(
            input="q",
            output='```json\n{"name": "test"}\n```',
        )
        score = SchemaValidationMetric(schema=schema).measure(ec)
        assert score.passed

    def test_markdown_fenced_json_array(self):
        schema = {
            "type": "array",
            "items": {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]},
        }
        ec = EvalCase(input="q", output='```json\n[{"title": "a"}]\n```')
        score = SchemaValidationMetric(schema=schema).measure(ec)
        assert score.passed

    def test_markdown_fenced_json_with_leading_prose(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        ec = EvalCase(
            input="q",
            output='Sure — here you go:\n```json\n{"name": "test"}\n```',
        )
        score = SchemaValidationMetric(schema=schema).measure(ec)
        assert score.passed

    def test_markdown_fenced_json_fails_schema_not_parse(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        ec = EvalCase(
            input="q",
            output='```json\n{"count": 42}\n```',
        )
        score = SchemaValidationMetric(schema=schema).measure(ec)
        assert not score.passed
        assert score.value == 0.0
        assert "Validation failed" in score.reason
        assert "could not be parsed" not in score.reason.lower()

    def test_unparseable_string_includes_detail(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        ec = EvalCase(input="q", output="not json at all")
        score = SchemaValidationMetric(schema=schema).measure(ec)
        assert not score.passed
        assert "Output could not be parsed as valid JSON" in score.reason
        assert "no JSON object or array found" in score.reason

    def test_invalid_fenced_json_includes_decode_detail(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        ec = EvalCase(input="q", output="```json\n{not valid json}\n```")
        score = SchemaValidationMetric(schema=schema).measure(ec)
        assert not score.passed
        assert "Output could not be parsed as valid JSON" in score.reason
        assert "Expecting" in score.reason or "could not decode" in score.reason
