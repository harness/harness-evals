"""Tests for the summarize() aggregation utilities."""

import pytest

from harness_evals import Score, summarize
from harness_evals.summary import MetricSummary, ScoreSummary


@pytest.mark.unit
class TestSummarize:
    def test_basic_aggregation(self):
        all_scores = [
            [Score(name="exact_match", value=1.0, threshold=0.8)],
            [Score(name="exact_match", value=0.0, threshold=0.8)],
        ]
        result = summarize(all_scores)

        assert isinstance(result, ScoreSummary)
        assert result.total_cases == 2
        assert "exact_match" in result.by_metric

        ms = result.by_metric["exact_match"]
        assert isinstance(ms, MetricSummary)
        assert ms.mean == 0.5
        assert ms.pass_rate == 0.5
        assert ms.count == 2
        assert ms.min_value == 0.0
        assert ms.max_value == 1.0
        assert ms.passed_count == 1
        assert ms.failed_count == 1

    def test_multiple_metrics(self):
        all_scores = [
            [
                Score(name="exact_match", value=1.0, threshold=0.8),
                Score(name="latency", value=0.9, threshold=0.5),
            ],
            [
                Score(name="exact_match", value=0.5, threshold=0.8),
                Score(name="latency", value=0.7, threshold=0.5),
            ],
        ]
        result = summarize(all_scores)

        assert len(result.by_metric) == 2
        assert result.by_metric["exact_match"].mean == 0.75
        assert result.by_metric["latency"].mean == 0.8
        assert result.by_metric["latency"].pass_rate == 1.0

    def test_overall_pass_rate(self):
        all_scores = [
            [
                Score(name="m1", value=1.0, threshold=0.8),
                Score(name="m2", value=0.3, threshold=0.8),
            ],
            [
                Score(name="m1", value=1.0, threshold=0.8),
                Score(name="m2", value=1.0, threshold=0.8),
            ],
        ]
        result = summarize(all_scores)
        assert result.overall_pass_rate == 0.75  # 3 of 4 passed

    def test_empty_input(self):
        result = summarize([])
        assert result.total_cases == 0
        assert result.by_metric == {}
        assert result.overall_pass_rate == 0.0

    def test_empty_score_lists(self):
        result = summarize([[], []])
        assert result.total_cases == 2
        assert result.by_metric == {}
        assert result.overall_pass_rate == 0.0

    def test_all_pass(self):
        all_scores = [
            [Score(name="m", value=1.0, threshold=0.5)],
            [Score(name="m", value=0.8, threshold=0.5)],
            [Score(name="m", value=0.9, threshold=0.5)],
        ]
        result = summarize(all_scores)
        assert result.by_metric["m"].pass_rate == 1.0
        assert result.overall_pass_rate == 1.0

    def test_all_fail(self):
        all_scores = [
            [Score(name="m", value=0.1, threshold=0.5)],
            [Score(name="m", value=0.2, threshold=0.5)],
        ]
        result = summarize(all_scores)
        assert result.by_metric["m"].pass_rate == 0.0
        assert result.overall_pass_rate == 0.0

    def test_uneven_score_lists(self):
        """Cases may have different numbers of scores (e.g. after None filtering)."""
        all_scores = [
            [Score(name="m1", value=1.0, threshold=0.8)],
            [
                Score(name="m1", value=0.5, threshold=0.8),
                Score(name="m2", value=0.9, threshold=0.5),
            ],
        ]
        result = summarize(all_scores)
        assert result.by_metric["m1"].count == 2
        assert result.by_metric["m2"].count == 1
