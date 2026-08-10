"""Tests for json_output parsing helpers."""

import pytest

from harness_evals.utils.json_output import json_parse_failure_reason, parse_json_value


@pytest.mark.unit
class TestParseJsonValue:
    def test_plain_object(self):
        assert parse_json_value('{"a": 1}') == {"a": 1}

    def test_fenced_object(self):
        assert parse_json_value('```json\n{"a": 1}\n```') == {"a": 1}

    def test_fenced_array(self):
        assert parse_json_value("```\n[1, 2]\n```") == [1, 2]

    def test_passthrough_dict(self):
        assert parse_json_value({"x": 1}) == {"x": 1}

    def test_prose_before_fenced_object(self):
        text = 'Here is the JSON:\n```json\n{"a": 1}\n```'
        assert parse_json_value(text) == {"a": 1}

    def test_garbage_returns_none(self):
        assert parse_json_value("not json at all") is None

    def test_garbage_failure_reason(self):
        assert "no JSON object or array found" in json_parse_failure_reason("not json at all")

    def test_invalid_fenced_json_returns_none(self):
        assert parse_json_value("```json\n{broken\n```") is None

    def test_invalid_fenced_json_failure_reason(self):
        reason = json_parse_failure_reason("```json\n{broken\n```")
        assert "Expecting" in reason or "could not decode" in reason
