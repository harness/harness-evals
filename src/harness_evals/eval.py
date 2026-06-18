"""Code-first ``run_eval()`` one-liner for running evaluations."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from harness_evals._async_compat import _run_async
from harness_evals.config.runner import gate_against_baseline
from harness_evals.config.schema import BaselineSpec
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.golden import Golden
from harness_evals.core.metric import BaseMetric
from harness_evals.core.runner import evaluate_dataset
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink
from harness_evals.plugins import load_plugins
from harness_evals.refs import ResourceRef, resolve
from harness_evals.sinks.stdout import StdoutSink
from harness_evals.targets.base import BaseTarget


class _CallableTarget(BaseTarget):
    """Wraps a plain callable (sync or async) as a ``BaseTarget``."""

    def __init__(self, fn: Callable) -> None:
        self._fn = fn
        self._is_async = asyncio.iscoroutinefunction(fn)

    async def ainvoke(self, golden: Golden) -> EvalCase:
        if self._is_async:
            return await self._fn(golden)
        return await asyncio.to_thread(self._fn, golden)


def run_eval(
    name: str,
    data: str | ResourceRef | list[Golden],
    target: BaseTarget | Callable[[Golden], EvalCase | Awaitable[EvalCase]],
    metrics: list[BaseMetric],
    *,
    sinks: list[BaseSink] | None = None,
    baseline: BaselineSpec | str | None = None,
    plugins: list[str] | None = None,
) -> list[list[Score]]:
    """Run an eval in one call. Returns scores.

    This is the code-first counterpart to the YAML config.

    Args:
        name: Eval name (for display and baseline tracking).
        data: Dataset — a ref string, ``ResourceRef``, or literal ``list[Golden]``.
        target: A ``BaseTarget`` instance or any callable
            ``(Golden) -> EvalCase`` (sync or async).
        metrics: Metrics to evaluate.
        sinks: Output sinks. Defaults to ``[StdoutSink()]``.
        baseline: Baseline spec, path string, or ``None``. A bare path
            string is shorthand for ``BaselineSpec(store="json", path=s,
            tolerance=0.05)``. Pass a ``BaselineSpec`` directly to
            control tolerance.
        plugins: Plugin modules to import before running.

    Returns:
        Per-golden score lists — ``list[list[Score]]``.
    """

    if plugins:
        load_plugins(plugins)

    return _run_async(_eval_async(name, data, target, metrics, sinks, baseline))


async def _eval_async(
    name: str,
    data: str | ResourceRef | list[Golden],
    target: BaseTarget | Callable,
    metrics: list[BaseMetric],
    sinks: list[BaseSink] | None,
    baseline: BaselineSpec | str | None,
) -> list[list[Score]]:
    goldens = await _resolve_data(data)

    target_obj = target if isinstance(target, BaseTarget) else _CallableTarget(target)

    if sinks is None:
        sinks = [StdoutSink(label=name)]

    async with target_obj:
        scores = await evaluate_dataset(goldens, target_obj.ainvoke, metrics=metrics, sinks=sinks)

    baseline_spec = _resolve_baseline(baseline)
    if baseline_spec is not None:
        gate_against_baseline(scores, baseline_spec)

    return scores


async def _resolve_data(data: str | ResourceRef | list[Golden]) -> list[Golden]:
    if isinstance(data, list):
        return data

    ref = data if isinstance(data, ResourceRef) else resolve(data)

    from harness_evals.plugins import dataset_source as lookup_dataset_source

    source_cls = lookup_dataset_source(ref.source)
    source = source_cls()
    async with source:
        return await source.fetch(ref)


def _resolve_baseline(baseline: BaselineSpec | str | None) -> BaselineSpec | None:
    if baseline is None:
        return None
    if isinstance(baseline, str):
        return BaselineSpec(store="json", path=baseline)
    return baseline
