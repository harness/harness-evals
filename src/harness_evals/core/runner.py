from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.golden import Golden
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink

if TYPE_CHECKING:
    from harness_evals.conversation.golden import ConversationGolden
    from harness_evals.llm.base import BaseLLM


def _enrich_score(score: Score, metric: BaseMetric) -> None:
    """Attach metric metadata (dimension) to score for downstream sinks."""
    if score.metadata is None:
        score.metadata = {}
    score.metadata.setdefault("dimension", metric.dimension.value)


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
            _enrich_score(score, metric)
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
            _enrich_score(score, metric)
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


async def _evaluate_dataset_single(
    goldens: list[Golden],
    agent_fn: Callable[[Golden], Awaitable[EvalCase]],
    metrics: list[BaseMetric],
    sinks: list[BaseSink] | None = None,
    *,
    concurrency: int | None = None,
) -> list[list[Score]]:
    """Internal helper: evaluate a list of single-turn Golden instances."""
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


async def _evaluate_dataset_conversation(
    goldens: list[ConversationGolden],
    agent_fn: Callable,
    metrics: list[BaseMetric],
    sinks: list[BaseSink] | None = None,
    *,
    concurrency: int | None = None,
    simulator_llm: BaseLLM,
) -> list[list[Score]]:
    """Internal helper: evaluate a list of ConversationGolden instances."""
    from harness_evals.conversation.simulator import ConversationSimulator

    simulator = ConversationSimulator(simulator_llm, max_concurrent=concurrency or 10)
    eval_cases = await simulator.simulate_batch(goldens, agent_fn)

    scored = await asyncio.gather(*[a_evaluate(ec, metrics) for ec in eval_cases])

    if sinks:
        for eval_case, scores in zip(eval_cases, scored, strict=True):
            for sink in sinks:
                sink.write(scores, eval_case)

    _finalize_sinks(sinks)
    return list(scored)


async def evaluate_dataset(
    goldens: list[Golden] | list[ConversationGolden],
    agent_fn: Callable,
    metrics: list[BaseMetric],
    sinks: list[BaseSink] | None = None,
    *,
    concurrency: int | None = None,
    simulator_llm: BaseLLM | None = None,
) -> list[list[Score]]:
    """Run an agent on goldens, then evaluate each resulting EvalCase.

    Accepts either a list of :class:`~harness_evals.core.golden.Golden`
    (single-turn) or a list of
    :class:`~harness_evals.conversation.golden.ConversationGolden`
    (multi-turn). Mixed lists raise :exc:`TypeError`.

    For ``ConversationGolden`` inputs, ``simulator_llm`` must be provided;
    it drives the simulated user turns via
    :class:`~harness_evals.conversation.simulator.ConversationSimulator`.

    ``agent_fn`` is async because agent calls are I/O-bound. For
    single-turn goldens it receives a ``Golden`` and returns an
    ``EvalCase``; for conversation goldens it receives
    ``list[Message]`` and returns a ``Message``.

    The ``concurrency`` semaphore gates ``agent_fn`` calls only for
    single-turn mode; for conversation mode it sets ``max_concurrent``
    on the simulator.

    Args:
        goldens: List of ``Golden`` or ``ConversationGolden`` instances
            (must not be mixed).
        agent_fn: Async callable appropriate to the golden type.
        metrics: Metrics to evaluate.
        sinks: Optional output sinks.
        concurrency: Max concurrent agent calls (single-turn) or
            conversations (multi-turn). ``None`` = unlimited / default 10.
        simulator_llm: LLM used to simulate user turns. Required when
            ``goldens`` contains ``ConversationGolden`` instances.

    Raises:
        ValueError: If ``concurrency`` is less than 1 or if
            ``ConversationGolden`` inputs are provided without
            ``simulator_llm``.
        TypeError: If ``goldens`` contains a mix of ``Golden`` and
            ``ConversationGolden`` instances.
    """
    if concurrency is not None and concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")

    if not goldens:
        return []

    from harness_evals.conversation.golden import ConversationGolden as _ConvGolden

    types = {type(g) for g in goldens}
    if len(types) > 1:
        raise TypeError(
            "evaluate_dataset() received a mixed list of Golden and ConversationGolden. "
            "All goldens must be the same type."
        )

    if issubclass(next(iter(types)), _ConvGolden):
        if simulator_llm is None:
            raise ValueError(
                "ConversationGolden inputs require a simulator_llm. "
                "Pass simulator_llm=<BaseLLM instance> to evaluate_dataset()."
            )
        return await _evaluate_dataset_conversation(
            goldens,  # type: ignore[arg-type]
            agent_fn,
            metrics,
            sinks,
            concurrency=concurrency,
            simulator_llm=simulator_llm,
        )

    return await _evaluate_dataset_single(
        goldens,  # type: ignore[arg-type]
        agent_fn,
        metrics,
        sinks,
        concurrency=concurrency,
    )
