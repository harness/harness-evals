"""Tests for json_output parsing helpers."""

import pytest

from harness_evals.utils.json_output import parse_json_value


@pytest.mark.unit
class TestParseJsonValue:
    def test_plain_object(self):
        assert parse_json_value('{"a": 1}') == {"a": 1}

    def test_fenced_object(self):
        assert parse_json_value('```json\n{"a": 1}\n```') == {"a": 1}

    def test_fenced_array(self):
        assert parse_json_value('```\n[1, 2]\n```') == [1, 2]

    def test_passthrough_dict(self):
        assert parse_json_value({"x": 1}) == {"x": 1}
