"""Score aggregation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

from harness_evals.core.score import Score


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
class ScoreSummary:
    """Aggregate statistics across all metrics and eval cases."""

    by_metric: dict[str, MetricSummary] = field(default_factory=dict)
    total_cases: int = 0
    overall_pass_rate: float = 0.0


def summarize(all_scores: list[list[Score]]) -> ScoreSummary:
    """Aggregate a batch of per-case score lists into summary statistics.

    ``all_scores`` is the return value of ``evaluate_cases()`` — a list
    of score lists, one per eval case.  ``None`` entries (from skipped
    metrics) are excluded automatically.
    """
    buckets: dict[str, list[Score]] = {}
    for case_scores in all_scores:
        for score in case_scores:
            if score is None:
                continue
            buckets.setdefault(score.name, []).append(score)

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

    total_cases = len(all_scores)
    total_scores = sum(ms.count for ms in by_metric.values())
    total_passed = sum(ms.passed_count for ms in by_metric.values())
    overall_pass_rate = total_passed / total_scores if total_scores > 0 else 0.0

    return ScoreSummary(
        by_metric=by_metric,
        total_cases=total_cases,
        overall_pass_rate=overall_pass_rate,
    )
