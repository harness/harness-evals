"""Tests for the summarize() aggregation utilities."""

import pytest

from harness_evals import Score, summarize
from harness_evals.summary import (
    DimensionSummary,
    MetricSummary,
    ScoreSummary,
    build_dimension_summary,
    dimension_of,
)


def _score(name, value, threshold, dimension=None):
    """Build a Score, stamping a dimension into metadata like the runner does."""
    s = Score(name=name, value=value, threshold=threshold)
    if dimension is not None:
        s.metadata = {"dimension": dimension}
    return s


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

    def test_quality_pass_rate(self):
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
        assert result.quality_pass_rate == 0.75  # 3 of 4 passed

    def test_empty_input(self):
        result = summarize([])
        assert result.total_cases == 0
        assert result.by_metric == {}
        assert result.quality_pass_rate == 0.0

    def test_empty_score_lists(self):
        result = summarize([[], []])
        assert result.total_cases == 2
        assert result.by_metric == {}
        assert result.quality_pass_rate == 0.0

    def test_all_pass(self):
        all_scores = [
            [Score(name="m", value=1.0, threshold=0.5)],
            [Score(name="m", value=0.8, threshold=0.5)],
            [Score(name="m", value=0.9, threshold=0.5)],
        ]
        result = summarize(all_scores)
        assert result.by_metric["m"].pass_rate == 1.0
        assert result.quality_pass_rate == 1.0

    def test_all_fail(self):
        all_scores = [
            [Score(name="m", value=0.1, threshold=0.5)],
            [Score(name="m", value=0.2, threshold=0.5)],
        ]
        result = summarize(all_scores)
        assert result.by_metric["m"].pass_rate == 0.0
        assert result.quality_pass_rate == 0.0

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


@pytest.mark.unit
class TestDimensionAggregation:
    """ADR-009 dimension roll-up and ADR-003 safety separation."""

    def test_by_dimension_populated(self):
        all_scores = [
            [
                _score("exact_match", 1.0, 0.8, "correctness"),
                _score("faithfulness", 0.6, 0.8, "groundedness"),
            ],
            [
                _score("exact_match", 0.0, 0.8, "correctness"),
                _score("faithfulness", 1.0, 0.8, "groundedness"),
            ],
        ]
        result = summarize(all_scores)

        assert set(result.by_dimension) == {"correctness", "groundedness"}
        corr = result.by_dimension["correctness"]
        assert isinstance(corr, DimensionSummary)
        assert corr.dimension == "correctness"
        assert corr.mean == 0.5  # (1.0 + 0.0) / 2
        assert corr.pass_rate == 0.5  # one of two >= 0.8
        assert corr.metric_count == 2
        assert corr.is_safety is False

        grnd = result.by_dimension["groundedness"]
        assert grnd.mean == 0.8  # (0.6 + 1.0) / 2
        assert grnd.pass_rate == 0.5

    def test_safety_separated_from_quality(self):
        """Safety failure must not dilute quality_pass_rate (ADR-003)."""
        all_scores = [
            [
                _score("exact_match", 1.0, 0.8, "correctness"),
                _score("faithfulness", 0.9, 0.8, "groundedness"),
                _score("plan_quality", 1.0, 0.8, "trajectory"),
                _score("pii_leak", 0.0, 1.0, "safety"),  # violation
            ]
        ]
        result = summarize(all_scores)

        # Quality excludes safety: all three non-safety metrics passed.
        assert result.quality_pass_rate == 1.0
        # Safety surfaced separately.
        assert result.safety_pass_rate == 0.0
        assert result.safety_violations == 1
        assert result.by_dimension["safety"].is_safety is True

    def test_all_safety_run(self):
        all_scores = [
            [
                _score("pii_leak", 1.0, 1.0, "safety"),
                _score("toxicity", 0.0, 1.0, "safety"),
            ]
        ]
        result = summarize(all_scores)

        # No non-safety scores -> quality pass rate defined as 0.0.
        assert result.quality_pass_rate == 0.0
        assert result.safety_pass_rate == 0.5
        assert result.safety_violations == 1
        assert set(result.by_dimension) == {"safety"}

    def test_scores_without_dimension_bucketed_unknown(self):
        """Scores lacking a dimension key land in 'unknown', not dropped, and count as quality."""
        all_scores = [
            [
                _score("mystery", 1.0, 0.5),  # no dimension metadata
                _score("exact_match", 0.0, 0.5, "correctness"),
            ]
        ]
        result = summarize(all_scores)

        assert "unknown" in result.by_dimension
        assert result.by_dimension["unknown"].is_safety is False
        assert result.by_dimension["unknown"].metric_count == 1
        # 'unknown' is non-safety, so it participates in quality (1 of 2 passed).
        assert result.quality_pass_rate == 0.5
        assert result.safety_violations == 0
        assert result.safety_pass_rate == 0.0

    def test_no_safety_scores(self):
        all_scores = [[_score("exact_match", 1.0, 0.8, "correctness")]]
        result = summarize(all_scores)

        assert result.safety_violations == 0
        assert result.safety_pass_rate == 0.0
        assert result.quality_pass_rate == 1.0
        assert "safety" not in result.by_dimension

    def test_empty_input_has_empty_dimensions(self):
        result = summarize([])
        assert result.by_dimension == {}
        assert result.quality_pass_rate == 0.0
        assert result.safety_pass_rate == 0.0
        assert result.safety_violations == 0


@pytest.mark.unit
class TestDimensionHelpers:
    """Shared aggregation helpers used by both summarize() and streaming sinks."""

    def test_dimension_of_reads_metadata(self):
        assert dimension_of(_score("m", 1.0, 0.5, "safety")) == "safety"

    def test_dimension_of_defaults_to_unknown(self):
        # No metadata at all, and metadata without a dimension key.
        assert dimension_of(Score(name="m", value=1.0, threshold=0.5)) == "unknown"
        s = Score(name="m", value=1.0, threshold=0.5)
        s.metadata = {"other": "x"}
        assert dimension_of(s) == "unknown"

    def test_build_dimension_summary_computes_aggregates(self):
        ds = build_dimension_summary("correctness", [1.0, 0.0], passed_count=1)
        assert ds.dimension == "correctness"
        assert ds.mean == 0.5
        assert ds.pass_rate == 0.5
        assert ds.metric_count == 2
        assert ds.is_safety is False

    def test_build_dimension_summary_flags_safety(self):
        ds = build_dimension_summary("safety", [0.0, 1.0], passed_count=1)
        assert ds.is_safety is True

    def test_build_dimension_summary_empty_is_safe(self):
        ds = build_dimension_summary("correctness", [], passed_count=0)
        assert ds.mean == 0.0
        assert ds.pass_rate == 0.0
        assert ds.metric_count == 0

    def test_summarize_matches_helper_for_a_dimension(self):
        """summarize()'s by_dimension entry equals a direct helper call (one source of truth)."""
        scores = [
            _score("a", 1.0, 0.5, "correctness"),
            _score("b", 0.0, 0.5, "correctness"),
        ]
        result = summarize([scores])
        direct = build_dimension_summary("correctness", [1.0, 0.0], passed_count=1)
        assert result.by_dimension["correctness"] == direct
