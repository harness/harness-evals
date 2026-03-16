from __future__ import annotations

from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink
from harness_evals.core.test_case import TestCase


def evaluate(
    test_case: TestCase,
    metrics: list[BaseMetric],
    sinks: list[BaseSink] | None = None,
) -> list[Score]:
    """Run all metrics on a test case and return scores.

    Does NOT raise on failure — returns scores with success=False instead.
    Writes to all sinks after scoring.
    """
    scores: list[Score] = []
    for metric in metrics:
        try:
            score = metric.measure(test_case)
        except Exception as e:
            score = Score(
                name=metric.name,
                value=0.0,
                threshold=metric.threshold,
                success=False,
                reason=f"Metric raised: {e}",
            )
        scores.append(score)

    if sinks:
        for sink in sinks:
            sink.write(scores, test_case)

    return scores


def assert_test(
    test_case: TestCase,
    metrics: list[BaseMetric],
    sinks: list[BaseSink] | None = None,
) -> list[Score]:
    """Run all metrics on a test case, write to sinks, raise if any fail.

    Raises AssertionError listing all failed metrics and their reasons.
    """
    scores = evaluate(test_case, metrics, sinks)

    failures = [s for s in scores if not s.success]
    if failures:
        details = "; ".join(f"{s.name}={s.value:.2f} (threshold={s.threshold}, reason={s.reason})" for s in failures)
        raise AssertionError(f"{len(failures)} metric(s) failed: {details}")

    return scores
