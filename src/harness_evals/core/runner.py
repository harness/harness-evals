from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.golden import Golden
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink
from harness_evals.llm.usage import TokenUsage, collect_token_usage
from harness_evals.logging_config import truncate_repr
from harness_evals.summary import ScoreSummary, summarize

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
        t0 = time.perf_counter()
        try:
            score = metric.measure(eval_case)
        except Exception as e:
            score = Score(
                name=metric.name,
                value=0.0,
                threshold=metric.threshold,
                reason=f"Metric raised: {e}",
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if score is not None:
            score.scoring_duration_ms = elapsed_ms
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
    """Async variant of evaluate(). Calls ``a_measure()`` on all metrics concurrently.

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

    async def _timed_measure(
        metric: BaseMetric,
    ) -> tuple[Score | None | BaseException, float, TokenUsage]:
        t0 = time.perf_counter()
        with collect_token_usage() as usage:
            try:
                result: Score | None | BaseException = await metric.a_measure(eval_case)
            except BaseException as exc:
                result = exc
        return result, (time.perf_counter() - t0) * 1000.0, usage

    timed_results = await asyncio.gather(*[_timed_measure(m) for m in metrics])

    scores: list[Score] = []
    for metric, (result, elapsed_ms, usage) in zip(metrics, timed_results, strict=True):
        if isinstance(result, BaseException):
            score = Score(
                name=metric.name,
                value=0.0,
                threshold=metric.threshold,
                reason=f"Metric raised: {result}",
            )
        else:
            score = result
        if score is not None:
            score.scoring_duration_ms = elapsed_ms
            _enrich_score(score, metric)
            if usage.input_tokens is not None or usage.output_tokens is not None:
                if score.metadata is None:
                    score.metadata = {}
                score.metadata.setdefault("input_tokens", usage.input_tokens)
                score.metadata.setdefault("output_tokens", usage.output_tokens)
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


_runner_logger = logging.getLogger(__name__)

# Per-item progress callback. Invoked once per dataset item as it finishes
# (in completion order, from the event loop) with
# ``(index, total, eval_case, scores)`` where ``index`` is 0-based. May be a
# plain function or an ``async def`` — a returned awaitable is awaited. Lets
# callers surface per-item progress/logging without the library picking a log
# level or format. Exceptions raised by the callback are caught and logged — a
# bad callback never aborts the run.
OnResult = Callable[[int, int, EvalCase, list[Score]], None | Awaitable[None]]


async def _evaluate_dataset_single(
    goldens: list[Golden],
    agent_fn: Callable[[Golden], Awaitable[EvalCase]],
    metrics: list[BaseMetric],
    sinks: list[BaseSink] | None = None,
    *,
    concurrency: int | None = None,
    on_result: OnResult | None = None,
) -> list[list[Score]]:
    """Internal helper: evaluate a list of single-turn Golden instances.

    Pipelines target invocation directly into scoring — no barrier between
    the two phases. A semaphore gates concurrent agent calls.

    Pass ``concurrency=None`` for unlimited parallelism (no semaphore).
    """
    total = len(goldens)
    sem: asyncio.Semaphore | None = asyncio.Semaphore(concurrency) if concurrency is not None else None
    sink_queue: asyncio.Queue[tuple[int, list[Score], EvalCase] | None] = asyncio.Queue() if sinks else None  # type: ignore[assignment]

    async def _sink_writer() -> None:
        """Background task that drains sink writes off the hot path.

        Cases finish out of order under concurrency, but sinks must see them in
        input (golden) order so appended rows line up with the dataset. Buffer
        items by index and flush the contiguous prefix as it becomes available.
        """
        assert sink_queue is not None
        pending: dict[int, tuple[list[Score], EvalCase]] = {}
        next_idx = 0
        while True:
            item = await sink_queue.get()
            if item is None:
                break
            idx, scores, eval_case = item
            pending[idx] = (scores, eval_case)
            while next_idx in pending:
                buf_scores, buf_case = pending.pop(next_idx)
                try:
                    for s in sinks:  # type: ignore[union-attr]
                        s.write(buf_scores, buf_case)
                except Exception:
                    _runner_logger.exception("Sink write failed for eval_case input=%s", buf_case.input)
                next_idx += 1

    async def _run_and_score(idx: int, golden: Golden) -> list[Score]:
        try:
            if sem is not None:
                async with sem:
                    eval_case = await agent_fn(golden)
            else:
                eval_case = await agent_fn(golden)
        except Exception as exc:
            # Target failure isolation: a raising agent_fn must not abort the
            # whole dataset. Emit failed scores for this item and continue.
            _runner_logger.exception("agent_fn raised for golden input=%s", golden.input)
            eval_case = EvalCase.from_golden(golden, output="")
            scores = []
            for metric in metrics:
                score = Score(
                    name=metric.name,
                    value=0.0,
                    threshold=metric.threshold,
                    reason=f"Target (agent_fn) raised: {exc}",
                    metadata={"target_error": True},
                )
                _enrich_score(score, metric)
                scores.append(score)
            target_error = True
        else:
            scores = await a_evaluate(eval_case, metrics)
            target_error = False
        metric_names = ", ".join(score.name for score in scores)
        target_error_suffix = " target_error=True" if target_error else ""
        _runner_logger.debug(
            "[%d/%d] input=%s output=%s metrics=[%s]%s",
            idx + 1,
            total,
            truncate_repr(eval_case.input),
            truncate_repr(eval_case.output),
            metric_names,
            target_error_suffix,
        )
        if on_result is not None:
            # Progress callbacks are observation-only: a raising callback must
            # never abort the eval run. Accept sync or async callbacks.
            try:
                result = on_result(idx, total, eval_case, scores)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                _runner_logger.exception("on_result callback raised for item %d", idx)
        if sink_queue is not None:
            await sink_queue.put((idx, scores, eval_case))
        return scores

    sink_task: asyncio.Task | None = None
    if sink_queue is not None:
        sink_task = asyncio.create_task(_sink_writer())

    scored = list(await asyncio.gather(*[_run_and_score(i, g) for i, g in enumerate(goldens)]))

    if sink_queue is not None:
        await sink_queue.put(None)
        assert sink_task is not None
        await sink_task

    _finalize_sinks(sinks)
    return scored


async def _evaluate_dataset_conversation(
    goldens: list[ConversationGolden],
    agent_fn: Callable,
    metrics: list[BaseMetric],
    sinks: list[BaseSink] | None = None,
    *,
    concurrency: int | None = None,
    simulator_llm: BaseLLM | None = None,
    on_result: OnResult | None = None,
) -> list[list[Score]]:
    """Internal helper: evaluate a list of ConversationGolden instances."""
    from harness_evals.conversation.simulator import ConversationSimulator

    simulator = ConversationSimulator(simulator_llm, max_concurrent=concurrency or 10)
    eval_cases = await simulator.simulate_batch(goldens, agent_fn)

    total = len(eval_cases)

    async def _score_and_report(idx: int, eval_case: EvalCase) -> list[Score]:
        # Fire on_result as each item's scoring finishes (completion order),
        # matching _evaluate_dataset_single — not in a post-gather barrier loop,
        # so a progress callback streams output during the run.
        scores = await a_evaluate(eval_case, metrics)
        if on_result is not None:
            try:
                result = on_result(idx, total, eval_case, scores)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                _runner_logger.exception("on_result callback raised for item %d", idx)
        return scores

    scored = list(await asyncio.gather(*[_score_and_report(i, ec) for i, ec in enumerate(eval_cases)]))

    if sinks:
        for eval_case, scores in zip(eval_cases, scored, strict=True):
            for sink in sinks:
                sink.write(scores, eval_case)

    _finalize_sinks(sinks)
    return scored


async def evaluate_dataset(
    goldens: list[Golden] | list[ConversationGolden],
    agent_fn: Callable,
    metrics: list[BaseMetric],
    sinks: list[BaseSink] | None = None,
    *,
    concurrency: int | None = None,
    simulator_llm: BaseLLM | None = None,
    on_result: OnResult | None = None,
) -> list[list[Score]]:
    """Run an agent on goldens, then evaluate each resulting EvalCase.

    Accepts either a list of :class:`~harness_evals.core.golden.Golden`
    (single-turn) or a list of
    :class:`~harness_evals.conversation.golden.ConversationGolden`
    (multi-turn). Mixed lists raise :exc:`TypeError`.

    For ``ConversationGolden`` inputs in SIMULATE or GRAPH mode,
    ``simulator_llm`` must be provided; it drives the simulated user turns
    via :class:`~harness_evals.conversation.simulator.ConversationSimulator`.
    SCRIPTED and REPLAY modes do not require a ``simulator_llm``.

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
            ``goldens`` contains ``ConversationGolden`` instances in
            SIMULATE or GRAPH mode.
        on_result: Optional per-item progress callback invoked as each item
            finishes with ``(index, total, eval_case, scores)`` (``index``
            0-based, completion order). May be sync or ``async`` — a returned
            awaitable is awaited. For observation only — exceptions it raises
            are caught and logged, never aborting the run.

    Raises:
        ValueError: If ``concurrency`` is less than 1 or if
            ``ConversationGolden`` inputs in SIMULATE/GRAPH mode are
            provided without ``simulator_llm``.
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
        from harness_evals.conversation.golden import ConversationMode as _ConvMode

        needs_llm = any(
            g.mode in (_ConvMode.SIMULATE, _ConvMode.GRAPH)  # type: ignore[union-attr]
            for g in goldens
        )
        if needs_llm and simulator_llm is None:
            raise ValueError(
                "ConversationGolden inputs with SIMULATE or GRAPH mode require a "
                "simulator_llm. Pass simulator_llm=<BaseLLM instance> to evaluate_dataset()."
            )
        return await _evaluate_dataset_conversation(
            goldens,  # type: ignore[arg-type]
            agent_fn,
            metrics,
            sinks,
            concurrency=concurrency,
            simulator_llm=simulator_llm,
            on_result=on_result,
        )

    return await _evaluate_dataset_single(
        goldens,  # type: ignore[arg-type]
        agent_fn,
        metrics,
        sinks,
        concurrency=concurrency,
        on_result=on_result,
    )


async def evaluate_dataset_pair(
    goldens: list[Golden],
    candidate_a_fn: Callable[[Golden], Awaitable[str]],
    candidate_b_fn: Callable[[Golden], Awaitable[str]],
    metric: BaseMetric,
    *,
    concurrency: int = 10,
) -> ScoreSummary:
    """Run two candidates on goldens, then evaluate pairwise.

    Calls both candidate functions for each golden, constructs
    ``EvalCase`` objects with candidate A as ``output`` and candidate B
    as ``expected``, then scores them with the provided metric
    (typically ``PairwiseMetric``).

    Returns a ``ScoreSummary`` with win-rate statistics.

    Args:
        goldens: Input prompts to evaluate.
        candidate_a_fn: Async callable that produces candidate A's response.
        candidate_b_fn: Async callable that produces candidate B's response.
        metric: Metric to evaluate (typically ``PairwiseMetric``).
        concurrency: Max concurrent golden evaluations (default 10).
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")

    sem = asyncio.Semaphore(concurrency)

    async def _run_pair(golden: Golden) -> list[Score]:
        async with sem:
            output_a, output_b = await asyncio.gather(
                candidate_a_fn(golden),
                candidate_b_fn(golden),
            )
            eval_case = EvalCase(
                input=golden.input,
                output=output_a,
                expected=output_b,
                context=golden.context,
            )
            return await a_evaluate(eval_case, [metric])

    all_scores = list(await asyncio.gather(*[_run_pair(g) for g in goldens]))
    return summarize(all_scores)
