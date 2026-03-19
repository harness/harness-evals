"""Baseline comparison — detect regressions, improvements, and unchanged metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

from harness_evals.core.score import Score


@dataclass
class MetricDelta:
    """One metric's change between baseline and current runs."""

    metric: str
    baseline_value: float
    current_value: float
    delta: float


@dataclass
class BaselineResult:
    """Result of comparing current scores against a baseline.

    ``tolerance`` is the minimum absolute delta to be considered a change.
    """

    tolerance: float
    regressions: list[MetricDelta] = field(default_factory=list)
    improvements: list[MetricDelta] = field(default_factory=list)
    unchanged: list[MetricDelta] = field(default_factory=list)

    @property
    def has_regressions(self) -> bool:
        return len(self.regressions) > 0

    def summary(self) -> str:
        parts: list[str] = []
        if self.regressions:
            items = ", ".join(f"{r.metric} {r.delta:+.4f}" for r in self.regressions)
            parts.append(f"Regressions ({len(self.regressions)}): {items}")
        if self.improvements:
            items = ", ".join(f"{i.metric} {i.delta:+.4f}" for i in self.improvements)
            parts.append(f"Improvements ({len(self.improvements)}): {items}")
        if self.unchanged:
            parts.append(f"Unchanged ({len(self.unchanged)})")
        return "; ".join(parts) if parts else "No metrics to compare"


def _mean_value(scores: list[Score]) -> float:
    """Average score value across a list of Scores."""
    if not scores:
        return 0.0
    return sum(s.value for s in scores) / len(scores)


def compare_to_baseline(
    current: dict[str, list[Score]],
    baseline: dict[str, list[Score]],
    tolerance: float = 0.05,
) -> BaselineResult:
    """Compare current metric scores against a saved baseline.

    For each metric present in *both* current and baseline, the average
    score value is compared.  A decrease exceeding ``tolerance`` is a
    regression; an increase exceeding ``tolerance`` is an improvement;
    otherwise the metric is unchanged.

    Metrics present only in current or only in baseline are ignored
    (they represent added/removed metrics, not regressions).

    Args:
        current: Metric name -> scores from the current run.
        baseline: Metric name -> scores from the baseline run.
        tolerance: Minimum absolute delta to count as a change (default 0.05).

    Returns:
        A ``BaselineResult`` with categorized metric deltas.
    """
    result = BaselineResult(tolerance=tolerance)
    common_metrics = sorted(set(current) & set(baseline))

    for metric in common_metrics:
        cur_avg = _mean_value(current[metric])
        base_avg = _mean_value(baseline[metric])
        delta = round(cur_avg - base_avg, 10)
        detail = MetricDelta(
            metric=metric,
            baseline_value=base_avg,
            current_value=cur_avg,
            delta=delta,
        )

        if delta < -tolerance:
            result.regressions.append(detail)
        elif delta > tolerance:
            result.improvements.append(detail)
        else:
            result.unchanged.append(detail)

    return result
