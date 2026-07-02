"""Score aggregation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

from harness_evals.core.score import Score

# The dimension whose scores are hard constraints (ADR-003): reported
# separately and never averaged into the quality pass rate.
SAFETY_DIMENSION = "safety"
# Bucket for scores whose metric did not declare a dimension.
UNKNOWN_DIMENSION = "unknown"


@dataclass
class MetricSummary:
    """Aggregate statistics for a single metric across multiple eval cases."""

    name: str
    mean: float
    pass_rate: float
    count: int
    min_value: float
    max_value: float
    passed_count: int
    failed_count: int


@dataclass
class DimensionSummary:
    """Aggregate statistics for a single evaluation dimension (ADR-009).

    A dimension groups every score whose metric declared that dimension
    (via ``score.metadata["dimension"]``). ``is_safety`` marks the Safety
    dimension, whose scores are hard constraints per ADR-003.
    """

    dimension: str
    mean: float
    pass_rate: float
    metric_count: int
    is_safety: bool


@dataclass
class ScoreSummary:
    """Aggregate statistics across all metrics and eval cases.

    ``overall_pass_rate`` blends every score together (kept for backward
    compatibility). Prefer ``quality_pass_rate`` (non-safety scores only)
    together with ``safety_pass_rate`` / ``safety_violations``, which keep
    safety as a separate hard constraint per ADR-003.
    """

    by_metric: dict[str, MetricSummary] = field(default_factory=dict)
    by_dimension: dict[str, DimensionSummary] = field(default_factory=dict)
    total_cases: int = 0
    overall_pass_rate: float = 0.0
    quality_pass_rate: float = 0.0
    safety_pass_rate: float = 0.0
    safety_violations: int = 0


def summarize(all_scores: list[list[Score]]) -> ScoreSummary:
    """Aggregate a batch of per-case score lists into summary statistics.

    ``all_scores`` is the return value of ``evaluate_cases()`` — a list
    of score lists, one per eval case.  ``None`` entries (from skipped
    metrics) are excluded automatically.
    """
    buckets: dict[str, list[Score]] = {}
    dim_buckets: dict[str, list[Score]] = {}
    for case_scores in all_scores:
        for score in case_scores:
            if score is None:
                continue
            buckets.setdefault(score.name, []).append(score)
            dimension = (score.metadata or {}).get("dimension") or UNKNOWN_DIMENSION
            dim_buckets.setdefault(dimension, []).append(score)

    by_metric: dict[str, MetricSummary] = {}
    for name, scores in buckets.items():
        values = [s.value for s in scores]
        passed = sum(1 for s in scores if s.passed)
        by_metric[name] = MetricSummary(
            name=name,
            mean=sum(values) / len(values),
            pass_rate=passed / len(values),
            count=len(values),
            min_value=min(values),
            max_value=max(values),
            passed_count=passed,
            failed_count=len(values) - passed,
        )

    by_dimension: dict[str, DimensionSummary] = {}
    for dimension, scores in dim_buckets.items():
        values = [s.value for s in scores]
        passed = sum(1 for s in scores if s.passed)
        by_dimension[dimension] = DimensionSummary(
            dimension=dimension,
            mean=sum(values) / len(values),
            pass_rate=passed / len(values),
            metric_count=len(values),
            is_safety=dimension == SAFETY_DIMENSION,
        )

    total_cases = len(all_scores)
    total_scores = sum(ms.count for ms in by_metric.values())
    total_passed = sum(ms.passed_count for ms in by_metric.values())
    overall_pass_rate = total_passed / total_scores if total_scores > 0 else 0.0

    # Safety scores are hard constraints (ADR-003): reported separately and
    # excluded from the quality pass rate so a safety failure is never diluted.
    safety_scores = dim_buckets.get(SAFETY_DIMENSION, [])
    safety_passed = sum(1 for s in safety_scores if s.passed)
    safety_pass_rate = safety_passed / len(safety_scores) if safety_scores else 0.0
    safety_violations = len(safety_scores) - safety_passed

    quality_total = total_scores - len(safety_scores)
    quality_passed = total_passed - safety_passed
    quality_pass_rate = quality_passed / quality_total if quality_total > 0 else 0.0

    return ScoreSummary(
        by_metric=by_metric,
        by_dimension=by_dimension,
        total_cases=total_cases,
        overall_pass_rate=overall_pass_rate,
        quality_pass_rate=quality_pass_rate,
        safety_pass_rate=safety_pass_rate,
        safety_violations=safety_violations,
    )
