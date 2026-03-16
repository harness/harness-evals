from __future__ import annotations

from collections.abc import Awaitable, Callable

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.golden import Golden
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink


def evaluate(
    eval_case: EvalCase,
    metrics: list[BaseMetric],
    sinks: list[BaseSink] | None = None,
) -> list[Score]:
    """Run all metrics on an eval case and return scores.

    Does NOT raise on failure — returns scores with passed=False instead.
    Writes to all sinks after scoring.
    """
    scores: list[Score] = []
    for metric in metrics:
        try:
            score = metric.measure(eval_case)
        except Exception as e:
            score = Score(
                name=metric.name,
                value=0.0,
                threshold=metric.threshold,
                reason=f"Metric raised: {e}",
            )
        scores.append(score)

    if sinks:
        for sink in sinks:
            sink.write(scores, eval_case)

    return scores


def assert_test(
    eval_case: EvalCase,
    metrics: list[BaseMetric],
    sinks: list[BaseSink] | None = None,
) -> list[Score]:
    """Run all metrics on an eval case, write to sinks, raise if any fail.

    Raises AssertionError listing all failed metrics and their reasons.
    """
    scores = evaluate(eval_case, metrics, sinks)

    failures = [s for s in scores if not s.passed]
    if failures:
        details = "; ".join(f"{s.name}={s.value:.2f} (threshold={s.threshold}, reason={s.reason})" for s in failures)
        raise AssertionError(f"{len(failures)} metric(s) failed: {details}")

    return scores


def evaluate_cases(
    cases: list[EvalCase],
    metrics: list[BaseMetric],
    sinks: list[BaseSink] | None = None,
) -> list[list[Score]]:
    """Batch evaluation of pre-captured eval cases.

    Runs ``evaluate()`` on each case and returns all score lists.
    """
    return [evaluate(case, metrics, sinks) for case in cases]


async def evaluate_dataset(
    goldens: list[Golden],
    agent_fn: Callable[[Golden], Awaitable[EvalCase]],
    metrics: list[BaseMetric],
    sinks: list[BaseSink] | None = None,
) -> list[list[Score]]:
    """Run an agent on goldens, then evaluate each resulting EvalCase.

    ``agent_fn`` is async because agent calls are I/O-bound. Each golden is
    passed to ``agent_fn`` to produce an EvalCase, which is then scored
    with ``evaluate()``.
    """
    results: list[list[Score]] = []
    for golden in goldens:
        eval_case = await agent_fn(golden)
        scores = evaluate(eval_case, metrics, sinks)
        results.append(scores)
    return results
