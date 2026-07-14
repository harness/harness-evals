"""Tests for answer extraction utilities."""

from __future__ import annotations

import pytest

from harness_evals.benchmarks._answer_utils import (
    compute_f1,
    extract_choice,
    extract_number,
    normalize_text,
)


@pytest.mark.unit
class TestExtractChoice:
    def test_simple_letter(self):
        assert extract_choice("A") == "A"
        assert extract_choice("B") == "B"

    def test_answer_is_pattern(self):
        assert extract_choice("The answer is C") == "C"
        assert extract_choice("the answer is (D)") == "D"

    def test_answer_colon_pattern(self):
        assert extract_choice("Answer: B") == "B"

    def test_letter_at_end(self):
        assert extract_choice("I think the correct option is A.") == "A"

    def test_custom_choices(self):
        assert extract_choice("The answer is 1", choices=["1", "2", "3"]) == "1"

    def test_no_valid_choice(self):
        assert extract_choice("I don't know", choices=["X", "Y"]) is None

    def test_verbose_response(self):
        response = "Let me think about this. The key factor is B because..."
        assert extract_choice(response) == "B"


@pytest.mark.unit
class TestExtractNumber:
    def test_hash_pattern(self):
        assert extract_number("#### 42") == "42"
        assert extract_number("Step 1: ...\n#### 100") == "100"

    def test_answer_is_pattern(self):
        assert extract_number("The answer is 7") == "7"

    def test_equals_pattern(self):
        assert extract_number("2 + 2 = 4") == "4"

    def test_negative_number(self):
        assert extract_number("The answer is -5") == "-5"

    def test_comma_separated(self):
        assert extract_number("#### 1,234") == "1234"

    def test_decimal(self):
        assert extract_number("The answer is 3.14") == "3.14"

    def test_last_number_fallback(self):
        assert extract_number("First I got 10, then 20, finally 30") == "30"

    def test_no_number(self):
        assert extract_number("no numbers here") is None


@pytest.mark.unit
class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("Hello World") == "hello world"

    def test_collapse_whitespace(self):
        assert normalize_text("  hello   world  ") == "hello world"


@pytest.mark.unit
class TestComputeF1:
    def test_perfect_match(self):
        assert compute_f1("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert compute_f1("hello", "world") == 0.0

    def test_partial_overlap(self):
        f1 = compute_f1("the cat sat", "the cat")
        assert 0.0 < f1 < 1.0

    def test_empty_strings(self):
        assert compute_f1("", "") == 1.0
        assert compute_f1("hello", "") == 0.0
        assert compute_f1("", "hello") == 0.0
