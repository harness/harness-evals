from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.golden import Golden
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink


def _finalize_sinks(sinks: list[BaseSink] | None) -> None:
    """Call finalize() and shutdown() on all sinks (no-op for sinks that don't override)."""
    if not sinks:
        return
    for sink in sinks:
        sink.finalize()
        sink.shutdown()


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


async def a_evaluate(
    eval_case: EvalCase,
    metrics: list[BaseMetric],
    sinks: list[BaseSink] | None = None,
) -> list[Score]:
    """Async variant of evaluate(). Calls ``a_measure()`` on each metric.

    Use this inside async contexts (event loops, Jupyter notebooks) to
    avoid the ``asyncio.run()`` crash that occurs when sync ``evaluate()``
    is called with LLM-judged metrics.
    """
    scores: list[Score] = []
    for metric in metrics:
        try:
            score = await metric.a_measure(eval_case)
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
    Calls ``finalize()`` on all sinks after all cases are processed.
    """
    results = [evaluate(case, metrics, sinks) for case in cases]
    _finalize_sinks(sinks)
    return results


def evaluate_batch_metrics(
    cases: list[EvalCase],
    outcomes: list[bool],
    metrics: list[BaseMetric],
) -> list[Score]:
    """Evaluate dataset-level metrics that require multiple cases.

    For metrics that implement ``measure_dataset(cases, outcomes)``
    (e.g. CalibrationMetric, DiscriminationMetric), this function
    calls that method. For standard metrics, it falls back to averaging
    ``measure()`` across all cases.

    Args:
        cases: All eval cases in the batch.
        outcomes: Whether each case was a success (True) or failure (False).
        metrics: Metrics to evaluate.

    Returns:
        One Score per metric.
    """
    scores: list[Score] = []
    for metric in metrics:
        try:
            if hasattr(metric, "measure_dataset"):
                score = metric.measure_dataset(cases, outcomes)
            else:
                case_scores = [metric.measure(c) for c in cases]
                avg = sum(s.value for s in case_scores) / len(case_scores) if case_scores else 0.0
                score = Score(
                    name=metric.name,
                    value=avg,
                    threshold=metric.threshold,
                    reason=f"Averaged over {len(case_scores)} cases",
                )
        except Exception as e:
            score = Score(
                name=metric.name,
                value=0.0,
                threshold=metric.threshold,
                reason=f"Metric raised: {e}",
            )
        scores.append(score)
    return scores


async def evaluate_dataset(
    goldens: list[Golden],
    agent_fn: Callable[[Golden], Awaitable[EvalCase]],
    metrics: list[BaseMetric],
    sinks: list[BaseSink] | None = None,
    *,
    concurrency: int | None = None,
) -> list[list[Score]]:
    """Run an agent on goldens, then evaluate each resulting EvalCase.

    ``agent_fn`` is async because agent calls are I/O-bound. Each golden is
    passed to ``agent_fn`` to produce an EvalCase, which is then scored
    with ``a_evaluate()`` (the async runner) to avoid event-loop conflicts.

    Args:
        goldens: List of Golden instances.
        agent_fn: Async function that produces an EvalCase from a Golden.
        metrics: Metrics to evaluate.
        sinks: Optional output sinks.
        concurrency: Max concurrent agent calls. None = unlimited.
    """
    semaphore = asyncio.Semaphore(concurrency) if concurrency else None

    async def _process(golden: Golden) -> list[Score]:
        if semaphore:
            async with semaphore:
                eval_case = await agent_fn(golden)
        else:
            eval_case = await agent_fn(golden)
        return await a_evaluate(eval_case, metrics, sinks)

    results = await asyncio.gather(*[_process(g) for g in goldens])
    _finalize_sinks(sinks)
    return list(results)
