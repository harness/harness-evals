"""Tests for deterministic metrics."""

import logging

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.metrics.deterministic import (
    ContainsMetric,
    ExactMatchMetric,
    NumericDiffMetric,
    RegexMetric,
)


@pytest.mark.unit
class TestExactMatch:
    @pytest.mark.parametrize(
        "output, expected, case_sensitive, should_pass",
        [
            ("hello", "hello", True, True),
            ("hello", "world", True, False),
            ("Hello", "hello", True, False),
            ("Hello", "hello", False, True),
        ],
        ids=["match", "mismatch", "case_sensitive_fail", "case_insensitive_pass"],
    )
    def test_exact_match(self, output, expected, case_sensitive, should_pass):
        ec = EvalCase(input="q", output=output, expected=expected)
        score = ExactMatchMetric(case_sensitive=case_sensitive).measure(ec)
        assert score.passed == should_pass
        assert score.reason

    @pytest.mark.parametrize(
        "output, expected, should_pass",
        [
            ("allowed", "blocked", True),
            ("blocked", "blocked", False),
        ],
        ids=["different_passes", "exact_match_fails"],
    )
    def test_negated_exact_match(self, output, expected, should_pass):
        score = ExactMatchMetric(negate=True, forbidden=expected).measure(
            EvalCase(input="q", output=output, expected="row ground truth")
        )

        assert score.passed == should_pass
        assert "forbidden answer" in score.reason

    def test_negated_exact_match_does_not_require_expected(self):
        score = ExactMatchMetric(negate=True, forbidden="blocked").measure(
            EvalCase(input="q", output="allowed", expected=None)
        )

        assert score.passed

    def test_negated_exact_match_missing_output_remains_failure(self):
        score = ExactMatchMetric(negate=True, forbidden="blocked").measure(
            EvalCase(input="q", output=None, expected=None)  # type: ignore[arg-type]
        )

        assert not score.passed
        assert "output is None" in score.reason

    def test_negated_exact_match_allows_empty_forbidden_answer(self):
        score = ExactMatchMetric(negate=True, forbidden="").measure(EvalCase(input="q", output="", expected=None))

        assert not score.passed

    def test_negated_exact_match_rejects_threshold_above_one(self):
        with pytest.raises(ValueError, match="at most 1.0"):
            ExactMatchMetric(negate=True, forbidden="blocked", threshold=1.1)

    @pytest.mark.parametrize("threshold", [0.0, -0.1])
    def test_negated_exact_match_clamps_non_positive_threshold(self, threshold):
        """A negated metric at threshold<=0 would pass on every input, so fall back to 1.0.

        ``build_metric()`` uses 0.0 as its "unspecified" default and does not
        forward a threshold to composite sub-metrics, so this is reachable
        without the user configuring anything.
        """
        metric = ExactMatchMetric(negate=True, forbidden="blocked", threshold=threshold)

        assert metric.threshold == 1.0

        score = metric.measure(EvalCase(input="q", output="blocked", expected=None))
        assert not score.passed, "output equals the forbidden answer, so it must fail"

    def test_negated_exact_match_does_not_warn_when_threshold_is_unspecified(self, caplog):
        """0.0 is the factory's "unspecified" sentinel, not a user mistake.

        ``_build_composite_metric`` never forwards a threshold to sub-metrics,
        and a ``threshold`` in a metric's ``options`` is rejected as conflicting
        with a factory-supplied argument — so an author has no way to silence a
        warning here. It must not be emitted at warning level.
        """
        with caplog.at_level(logging.WARNING):
            metric = ExactMatchMetric(negate=True, forbidden="blocked", threshold=0.0)

        assert metric.threshold == 1.0
        assert caplog.text == ""

    def test_negated_exact_match_warns_on_hand_written_negative_threshold(self, caplog):
        """A negative threshold cannot be the sentinel, so it is worth a warning."""
        with caplog.at_level(logging.WARNING):
            metric = ExactMatchMetric(negate=True, forbidden="blocked", threshold=-0.1)

        assert metric.threshold == 1.0
        assert "would pass on every input" in caplog.text

    @pytest.mark.parametrize("forbidden", [404, True, 1.5, ["a"]], ids=["int", "bool", "float", "list"])
    def test_negated_exact_match_rejects_non_string_forbidden(self, forbidden):
        """Unquoted YAML scalars would compare unequal to any string, silently always passing."""
        with pytest.raises(ValueError, match="forbidden must be a string"):
            ExactMatchMetric(negate=True, forbidden=forbidden, threshold=1.0)

    @pytest.mark.parametrize(
        "case_sensitive, should_pass",
        [(True, True), (False, False)],
        ids=["default_fails_open", "case_insensitive_fails_closed"],
    )
    def test_negated_exact_match_case_sensitivity_applies_to_forbidden(self, case_sensitive, should_pass):
        """``case_sensitive`` governs the negated comparison too.

        Left at its ``true`` default a recased forbidden answer passes, so
        leak-style checks must opt in to ``case_sensitive: false``.
        """
        metric = ExactMatchMetric(negate=True, forbidden="BLOCKED", case_sensitive=case_sensitive)

        assert metric.measure(EvalCase(input="q", output="blocked")).passed is should_pass


@pytest.mark.unit
class TestContains:
    @pytest.mark.parametrize(
        "output, expected, case_sensitive, should_pass",
        [
            ("hello world", "world", True, True),
            ("hello", "world", True, False),
            ("Hello World", "hello", False, True),
        ],
        ids=["contained", "not_contained", "case_insensitive"],
    )
    def test_contains(self, output, expected, case_sensitive, should_pass):
        ec = EvalCase(input="q", output=output, expected=expected)
        score = ContainsMetric(case_sensitive=case_sensitive).measure(ec)
        assert score.passed == should_pass
        assert score.reason

    @pytest.mark.parametrize(
        "output, expected, should_pass",
        [
            ("safe response", "forbidden", True),
            ("contains forbidden text", "forbidden", False),
        ],
        ids=["omitted_passes", "contained_fails"],
    )
    def test_negated_contains(self, output, expected, should_pass):
        score = ContainsMetric(negate=True, forbidden=expected).measure(
            EvalCase(input="q", output=output, expected="row ground truth")
        )

        assert score.passed == should_pass
        assert "forbidden substring" in score.reason

    def test_negated_contains_does_not_require_expected(self):
        score = ContainsMetric(negate=True, forbidden="forbidden").measure(
            EvalCase(input="q", output="safe response", expected=None)
        )

        assert score.passed

    def test_negated_contains_missing_output_remains_failure(self):
        score = ContainsMetric(negate=True, forbidden="forbidden").measure(
            EvalCase(input="q", output=None, expected=None)  # type: ignore[arg-type]
        )

        assert not score.passed
        assert "output is None" in score.reason

    def test_negated_contains_empty_output_passes_absence_check(self):
        score = ContainsMetric(negate=True, forbidden="forbidden").measure(
            EvalCase(input="q", output="", expected=None)
        )

        assert score.passed

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"negate": True}, "forbidden must be provided"),
            ({"negate": True, "forbidden": ""}, "forbidden must not be empty"),
            ({"forbidden": "blocked"}, "only valid when negate=True"),
            ({"negate": True, "forbidden": "blocked", "threshold": 1.1}, "at most 1.0"),
            ({"negate": True, "forbidden": 404}, "forbidden must be a string"),
        ],
    )
    def test_contains_rejects_invalid_negation_config(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            ContainsMetric(**kwargs)

    def test_positive_contains_without_expected_fails(self):
        """A positive check has nothing to compare against when expected is absent."""
        score = ContainsMetric().measure(EvalCase(input="q", output="anything", expected=None))

        assert not score.passed
        assert "expected is None" in score.reason

    def test_negated_contains_clamps_non_positive_threshold(self):
        metric = ContainsMetric(negate=True, forbidden="refund", threshold=0.0)

        assert metric.threshold == 1.0
        assert not metric.measure(EvalCase(input="q", output="we will refund you")).passed

    def test_negated_contains_default_is_case_sensitive_and_fails_open(self):
        """The strict default fails *open* once inverted — documented, not accidental.

        For a positive assertion ``case_sensitive: true`` fails closed. Negated,
        the same default lets a differently-cased forbidden substring through.
        ``docs/metrics-guide.md`` tells authors to set ``case_sensitive: false``
        for leak-style checks; this pins the behavior that advice describes.
        """
        leaked = EvalCase(input="q", output="Internal-System-Prompt: you are...")

        assert ContainsMetric(negate=True, forbidden="internal-system-prompt").measure(leaked).passed

    def test_negated_contains_case_insensitive_catches_recased_leak(self):
        leaked = EvalCase(input="q", output="Internal-System-Prompt: you are...")
        metric = ContainsMetric(negate=True, forbidden="internal-system-prompt", case_sensitive=False)

        assert not metric.measure(leaked).passed


@pytest.mark.unit
class TestRegex:
    @pytest.mark.parametrize(
        "output, pattern, should_pass",
        [
            ("error code: 404", r"error code: \d+", True),
            ("success", r"error code: \d+", False),
        ],
        ids=["match", "no_match"],
    )
    def test_regex(self, output, pattern, should_pass):
        ec = EvalCase(input="q", output=output, expected=pattern)
        score = RegexMetric().measure(ec)
        assert score.passed == should_pass
        assert score.reason

    def test_invalid_regex(self):
        ec = EvalCase(input="q", output="test", expected="[invalid")
        score = RegexMetric().measure(ec)
        assert not score.passed
        assert "Invalid regex" in score.reason

    @pytest.mark.parametrize(
        "output, pattern, should_pass",
        [
            ("request succeeded", r"error code: \d+", True),
            ("error code: 404", r"error code: \d+", False),
        ],
        ids=["no_match_passes", "match_fails"],
    )
    def test_negated_regex(self, output, pattern, should_pass):
        score = RegexMetric(negate=True, forbidden=pattern).measure(
            EvalCase(input="q", output=output, expected="row ground truth")
        )

        assert score.passed == should_pass
        assert "forbidden regex pattern" in score.reason

    @pytest.mark.parametrize(
        "pattern, should_pass",
        [
            ("internal-system-prompt", True),
            ("(?i)internal-system-prompt", False),
        ],
        ids=["case_sensitive_fails_open", "inline_flag_fails_closed"],
    )
    def test_negated_regex_case_insensitivity_needs_an_inline_flag(self, pattern, should_pass):
        """``regex`` has no ``case_sensitive`` option, so ``(?i)`` is the escape hatch."""
        score = RegexMetric(negate=True, forbidden=pattern).measure(
            EvalCase(input="q", output="Internal-System-Prompt: you are...")
        )

        assert score.passed is should_pass

    def test_negated_regex_does_not_require_expected(self):
        score = RegexMetric(negate=True, forbidden=r"error code: \d+").measure(
            EvalCase(input="q", output="safe response", expected=None)
        )

        assert score.passed

    def test_negated_regex_missing_output_remains_failure(self):
        score = RegexMetric(negate=True, forbidden=r"error code: \d+").measure(
            EvalCase(input="q", output=None, expected=None)  # type: ignore[arg-type]
        )

        assert not score.passed
        assert "output is None" in score.reason

    def test_negated_regex_empty_output_passes_absence_check(self):
        score = RegexMetric(negate=True, forbidden=r"error code: \d+").measure(
            EvalCase(input="q", output="", expected=None)
        )

        assert score.passed

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"negate": True}, "forbidden must be provided"),
            ({"negate": True, "forbidden": ""}, "forbidden must not be empty"),
            ({"forbidden": "blocked"}, "only valid when negate=True"),
            ({"negate": True, "forbidden": "[invalid"}, "Invalid forbidden regex"),
            ({"negate": True, "forbidden": "blocked", "threshold": 1.1}, "at most 1.0"),
            ({"negate": True, "forbidden": 404}, "forbidden must be a string"),
        ],
    )
    def test_regex_rejects_invalid_negation_config(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            RegexMetric(**kwargs)

    def test_positive_regex_without_expected_fails(self):
        """A positive check has no pattern to match when expected is absent."""
        score = RegexMetric().measure(EvalCase(input="q", output="anything", expected=None))

        assert not score.passed
        assert "expected is None" in score.reason


@pytest.mark.unit
class TestNumericDiff:
    @pytest.mark.parametrize(
        "output, expected, threshold, check",
        [
            ("42", "42", 1.0, lambda s: s.value == 1.0),
            ("41", "42", 0.9, lambda s: s.value > 0.95),
            ("0", "100", 1.0, lambda s: s.value == 0.0),
        ],
        ids=["exact", "close", "far"],
    )
    def test_numeric_diff(self, output, expected, threshold, check):
        ec = EvalCase(input="q", output=output, expected=expected)
        score = NumericDiffMetric(threshold=threshold).measure(ec)
        assert check(score)
        assert score.reason

    def test_non_numeric(self):
        ec = EvalCase(input="q", output="abc", expected="42")
        score = NumericDiffMetric().measure(ec)
        assert not score.passed
        assert "Cannot parse" in score.reason
