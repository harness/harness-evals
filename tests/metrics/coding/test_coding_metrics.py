"""Coding-agent grading metrics: pure readers of stashed grading results.

These metrics score precomputed pipeline outputs stashed in
``EvalCase.metadata`` (a grading pipeline — e.g. one that ran a hidden test
suite with coverage — records its result before scoring). They are pure
readers: no subprocess, no I/O, safe inside the runner's unthrottled metric
fan-out, and skipped entirely (``None``) on ordinary evals.
"""

from __future__ import annotations

import pytest

from harness_evals import EvalCase, Golden
from harness_evals.metrics.coding import (
    StatementCoverageMetric,
    TestSuitePassMetric,
)


def _case(metadata: dict) -> EvalCase:
    return EvalCase.from_golden(Golden(input="task"), output="out", metadata_extra=metadata)


def _plain_case() -> EvalCase:
    return EvalCase.from_golden(Golden(input="task"), output="out")


class TestTestSuitePass:
    def test_grader_shape_pass_fraction(self):
        case = _case({"mb_grade": {"conformance": {"ran": True, "passed": 9, "total": 10}}})
        score = TestSuitePassMetric(path="mb_grade.conformance").measure(case)
        assert score is not None and score.value == pytest.approx(0.9)

    def test_grader_shape_not_ran_scores_zero(self):
        case = _case({"mb_grade": {"self_suite": {"ran": False, "reason": "no tests"}}})
        score = TestSuitePassMetric(path="mb_grade.self_suite").measure(case)
        assert score is not None and score.value == 0.0
        assert "no tests" in score.reason

    def test_ready_fraction_shape(self):
        case = _case({"grade": {"suite": {"pass_fraction": 0.5}}})
        score = TestSuitePassMetric(path="grade.suite").measure(case)
        assert score is not None and score.value == pytest.approx(0.5)

    def test_bare_fraction_shape(self):
        case = _case({"grade": {"suite": 0.75}})
        score = TestSuitePassMetric(path="grade.suite").measure(case)
        assert score is not None and score.value == pytest.approx(0.75)

    def test_missing_path_skips(self):
        assert TestSuitePassMetric(path="mb_grade.conformance").measure(_plain_case()) is None

    def test_partial_path_skips(self):
        case = _case({"mb_grade": {}})
        assert TestSuitePassMetric(path="mb_grade.conformance").measure(case) is None

    def test_uninterpretable_node_skips(self):
        case = _case({"grade": {"suite": "not-a-suite"}})
        assert TestSuitePassMetric(path="grade.suite").measure(case) is None

    def test_custom_name_and_dimension(self):
        metric = TestSuitePassMetric(path="mb_grade.conformance", name="conformance_pass")
        case = _case({"mb_grade": {"conformance": {"ran": True, "passed": 10, "total": 10}}})
        score = metric.measure(case)
        assert score.name == "conformance_pass"
        assert score.value == 1.0
        assert score.metadata["path"] == "mb_grade.conformance"


class TestStatementCoverage:
    def test_fraction_from_path(self):
        case = _case({"mb_grade": {"app_coverage": 0.83}})
        score = StatementCoverageMetric(path="mb_grade.app_coverage").measure(case)
        assert score is not None and score.value == pytest.approx(0.83)

    def test_zero_is_a_floor_score_not_a_skip(self):
        case = _case({"mb_grade": {"app_coverage": 0.0}})
        score = StatementCoverageMetric(path="mb_grade.app_coverage").measure(case)
        assert score is not None and score.value == 0.0

    def test_clamps_into_score_domain(self):
        case = _case({"grade": {"coverage": 1.5}})
        score = StatementCoverageMetric(path="grade.coverage").measure(case)
        assert score is not None and score.value == 1.0
        case = _case({"grade": {"coverage": -0.2}})
        assert StatementCoverageMetric(path="grade.coverage").measure(case).value == 0.0

    def test_missing_path_skips(self):
        assert StatementCoverageMetric(path="mb_grade.app_coverage").measure(_plain_case()) is None

    def test_uninterpretable_node_skips(self):
        case = _case({"grade": {"coverage": {"weird": True}}})
        assert StatementCoverageMetric(path="grade.coverage").measure(case) is None


class TestCatalogRegistration:
    def test_kinds_present_in_catalog(self):
        from harness_evals.catalog import catalog

        kinds = {entry.kind for entry in catalog()}
        assert "test_suite_pass" in kinds
        assert "statement_coverage" in kinds

    def test_catalog_entries_use_default_names(self):
        from harness_evals.catalog import catalog

        entries = {e.kind: e for e in catalog()}
        assert entries["test_suite_pass"].dimension.value == "correctness"
        assert entries["statement_coverage"].dimension.value == "correctness"
