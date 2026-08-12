"""Stderr progress reporting for long-running eval runs."""

from __future__ import annotations

import sys
from collections.abc import Awaitable, Callable
from typing import Any

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score
from harness_evals.logging_config import truncate_repr

OnProgress = Callable[[int, int, str, str], None | Awaitable[None]]


def report_progress(message: str) -> None:
    """Print a progress line to stderr (visible regardless of log level)."""
    print(message, file=sys.stderr, flush=True)


def golden_label(golden: object) -> str:
    golden_id = getattr(golden, "id", None)
    if golden_id:
        return str(golden_id)
    scenario = getattr(golden, "scenario", None)
    if scenario:
        return truncate_repr(str(scenario), max_len=72)
    input_text = getattr(golden, "input", None)
    if input_text is not None:
        return truncate_repr(str(input_text), max_len=72)
    return "(item)"


def eval_case_label(eval_case: EvalCase) -> str:
    """Short label for stderr progress (prefer golden id over empty input)."""
    meta = eval_case.metadata or {}
    golden_id = meta.get("golden_id")
    if golden_id:
        return str(golden_id)
    scenario = meta.get("scenario")
    if scenario:
        return truncate_repr(str(scenario), max_len=72)
    return truncate_repr(eval_case.input or eval_case.output or "", max_len=72)


def make_stderr_progress_handlers() -> tuple[OnProgress, Callable[..., None]]:
    """Return ``(on_progress, on_result)`` callbacks that print to stderr."""

    def on_progress(index: int, total: int, phase: str, label: str) -> None:
        n = index + 1
        if phase == "running":
            report_progress(f"[{n}/{total}] running — {label}")
        elif phase == "scoring":
            report_progress(f"[{n}/{total}] scoring metrics…")

    def on_result(index: int, total: int, eval_case: EvalCase, scores: list[Score]) -> None:
        n = index + 1
        passed = sum(1 for score in scores if score.passed)
        failed = len(scores) - passed
        label = eval_case_label(eval_case)
        status = f"{passed}/{len(scores)} metrics passed"
        if failed:
            status += f", {failed} failed"
        report_progress(f"[{n}/{total}] done — {status} | {label}")

    return on_progress, on_result


async def invoke_progress_callback(callback: Callable[..., Any], *args: Any) -> None:
    """Invoke a sync or async progress/on_result callback; swallow errors."""
    import inspect
    import logging

    logger = logging.getLogger(__name__)
    try:
        result = callback(*args)
        if inspect.isawaitable(result):
            await result
    except Exception:
        logger.exception("Progress callback raised for args=%s", args[:2])
