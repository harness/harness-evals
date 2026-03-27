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
    Metrics that return ``None`` (not applicable) are excluded from the
    result list.  Writes to all sinks after scoring.

    Note: ``finalize()`` is NOT called here. When using sinks with
    ``evaluate()`` in a loop, call ``sink.finalize()`` yourself after
    all cases are processed. ``evaluate_cases()`` and
    ``evaluate_dataset()`` handle this automatically.
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
        if score is not None:
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

    Metrics that return ``None`` (not applicable) are excluded from the
    result list.

    Note: ``finalize()`` is NOT called here. When using sinks with
    ``evaluate()`` or ``a_evaluate()`` in a loop, call
    ``sink.finalize()`` yourself after all cases are processed.
    ``evaluate_cases()`` and ``evaluate_dataset()`` handle this
    automatically.
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
        if score is not None:
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
            score = metric.measure_dataset(cases, outcomes)
            if score is None:
                case_scores = [s for c in cases if (s := metric.measure(c)) is not None]
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

    The ``concurrency`` semaphore gates ``agent_fn`` calls only; metric
    evaluation (including LLM-judged metrics) runs without throttling.

    Sink writes happen sequentially after all concurrent work completes
    to avoid thread-safety issues with file-based sinks.

    Args:
        goldens: List of Golden instances.
        agent_fn: Async function that produces an EvalCase from a Golden.
        metrics: Metrics to evaluate.
        sinks: Optional output sinks.
        concurrency: Max concurrent agent calls. ``None`` = unlimited.

    Raises:
        ValueError: If ``concurrency`` is less than 1.
    """
    if concurrency is not None and concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")

    semaphore = asyncio.Semaphore(concurrency) if concurrency is not None else None

    async def _run_agent(golden: Golden) -> EvalCase:
        if semaphore:
            async with semaphore:
                return await agent_fn(golden)
        return await agent_fn(golden)

    eval_cases = await asyncio.gather(*[_run_agent(g) for g in goldens])

    scored = await asyncio.gather(*[a_evaluate(ec, metrics) for ec in eval_cases])

    if sinks:
        for eval_case, scores in zip(eval_cases, scored, strict=True):
            for sink in sinks:
                sink.write(scores, eval_case)

    _finalize_sinks(sinks)
    return list(scored)
