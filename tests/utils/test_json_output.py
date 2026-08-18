"""Tests for json_output parsing helpers."""

import pytest

from harness_evals.utils.json_output import JSON_PARSE_FAILED, json_parse_failure_reason, parse_json_value


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

    def test_prose_then_bare_object_after_colon_newline(self):
        text = 'Sure — here you go:\n{"a": 1}'
        assert parse_json_value(text) == {"a": 1}

    def test_bare_object_with_trailing_prose(self):
        text = '{"a": 1}\n\nHope this helps!'
        assert parse_json_value(text) == {"a": 1}

    def test_bare_object_with_trailing_brace_in_prose(self):
        text = '{"a": 1}\nNote: the {feature_map} field is empty.'
        assert parse_json_value(text) == {"a": 1}

    def test_bare_array_with_trailing_bracket_in_prose(self):
        text = '[{"id": 1}]\nGenerated from surface [main] page.'
        assert parse_json_value(text) == [{"id": 1}]

    def test_bare_object_wins_over_trailing_fenced_example(self):
        text = '{"a": 1}\n\nExample usage:\n```json\n{"b": 2}\n```'
        assert parse_json_value(text) == {"a": 1}

    def test_prose_anchored_bare_answer_wins_over_trailing_schema_fence(self):
        text = (
            "Answer:\n"
            '{"feature_map": ["login"], "test_cases": [1]}\n\n'
            'Schema I followed:\n```json\n{"type": "object"}\n```'
        )
        assert parse_json_value(text) == {"feature_map": ["login"], "test_cases": [1]}

    def test_invalid_pseudo_schema_then_valid_answer(self):
        text = (
            "Here is the schema I used:\n"
            "{feature_map: array, test_cases: array}\n"
            "And here is the output:\n"
            '{"feature_map": ["login"], "test_cases": [1]}'
        )
        assert parse_json_value(text) == {"feature_map": ["login"], "test_cases": [1]}

    def test_unclosed_fence_does_not_hang(self):
        text = "```json\n" + (" " * 5000)
        assert parse_json_value(text) is JSON_PARSE_FAILED

    def test_fenced_object_without_trailing_newline_before_close(self):
        assert parse_json_value('```json\n{"a": 1}```') == {"a": 1}

    def test_refusal_quoting_schema_does_not_parse(self):
        text = (
            "I cannot generate tests for this surface. For reference, the expected "
            'format is {"feature_map": [], "test_cases": []}.'
        )
        assert parse_json_value(text) is JSON_PARSE_FAILED

    def test_error_message_with_empty_object_does_not_parse(self):
        assert parse_json_value("Error: unexpected token at {} in input") is JSON_PARSE_FAILED

    def test_garbage_returns_parse_failed(self):
        assert parse_json_value("not json at all") is JSON_PARSE_FAILED

    def test_garbage_failure_reason(self):
        assert "no JSON object or array found" in json_parse_failure_reason("not json at all")

    def test_parseable_output_failure_reason(self):
        reason = json_parse_failure_reason('{"a": 1}')
        assert reason == "output parsed successfully; failure was not a parse error"

    def test_json_null_literal(self):
        assert parse_json_value("null") is None
        assert parse_json_value("null") is not JSON_PARSE_FAILED

    def test_fenced_json_null_literal(self):
        assert parse_json_value("```json\nnull\n```") is None
        assert parse_json_value("```json\nnull\n```") is not JSON_PARSE_FAILED

    def test_fenced_json_string_literal(self):
        assert parse_json_value('```json\n"hello"\n```') == "hello"

    def test_object_fence_wins_over_leading_scalar_fence(self):
        text = '```json\n7\n```\n```json\n{"a": 1}\n```'
        assert parse_json_value(text) == {"a": 1}

    def test_invalid_fenced_json_returns_parse_failed(self):
        assert parse_json_value("```json\n{broken\n```") is JSON_PARSE_FAILED

    def test_invalid_fenced_json_failure_reason(self):
        reason = json_parse_failure_reason("```json\n{broken\n```")
        assert "Expecting" in reason or "could not decode" in reason

    def test_reasoning_then_fenced_object(self):
        text = '<reasoning>thinking</reasoning>\n```json\n{"a": 1}\n```'
        assert parse_json_value(text) == {"a": 1}

    def test_reasoning_then_plain_object(self):
        text = '<reasoning>done</reasoning>\n{"a": 1}'
        assert parse_json_value(text) == {"a": 1}

    def test_reasoning_substring_inside_json_value_preserved(self):
        text = '{"note": "<reasoning>a</reasoning>tail"}'
        assert parse_json_value(text) == {"note": "<reasoning>a</reasoning>tail"}

    def test_reasoning_substring_inside_fenced_json_value_preserved(self):
        text = '```json\n{"note": "<reasoning>a</reasoning>tail"}\n```'
        assert parse_json_value(text) == {"note": "<reasoning>a</reasoning>tail"}

    def test_last_valid_fenced_json_wins_over_draft_in_reasoning(self):
        text = (
            '<reasoning>Draft:\n```json\n{"name": "DRAFT"}\n```\nnow final</reasoning>\n```json\n{"name": "FINAL"}\n```'
        )
        assert parse_json_value(text) == {"name": "FINAL"}

    def test_draft_in_thinking_block_is_skipped(self):
        text = '<thinking>```json\n{"name": "DRAFT"}\n```</thinking>\n```json\n{"name": "FINAL"}\n```'
        assert parse_json_value(text) == {"name": "FINAL"}

    def test_first_fenced_answer_wins_over_trailing_example(self):
        text = '```json\n{"test_cases": [1]}\n```\nExample:\n```json\n{"id": "tc-1"}\n```'
        assert parse_json_value(text) == {"test_cases": [1]}

    def test_first_fenced_answer_wins_over_longer_trailing_example(self):
        text = (
            '```json\n{"feature_map": ["login"], "test_cases": [1]}\n```\n'
            "Example:\n"
            '```json\n{"id": "tc-1", "steps": ["a", "b", "c"], "expected": "ok", "priority": "high"}\n```'
        )
        assert parse_json_value(text) == {"feature_map": ["login"], "test_cases": [1]}

    def test_answer_wins_over_reasoning_block_after_answer(self):
        text = '```json\n{"name": "FINAL"}\n```\n<reasoning>alt: ```json\n{"name": "DRAFT"}\n```</reasoning>'
        assert parse_json_value(text) == {"name": "FINAL"}
