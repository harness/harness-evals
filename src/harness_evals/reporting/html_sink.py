"""HtmlSink: a BaseSink that accumulates results and generates an HTML report.

Usage::

    sink = HtmlSink("report.html", title="My Evals")

    # Use with evaluate() — results accumulate automatically
    for case in cases:
        evaluate(case, metrics, sinks=[sink])

    # Generate the report when done
    sink.finalize()
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink
from harness_evals.reporting.html_reporter import HtmlReporter


class HtmlSink(BaseSink):
    """Accumulates eval results and generates an HTML report on ``finalize()``.

    The ``group_fn`` and ``variant_fn`` callbacks extract grouping and variant
    info from each EvalCase. Defaults use tags['group'] and tags['variant'].
    """

    def __init__(
        self,
        path: str | Path,
        title: str = "Eval Report",
        group_fn: Callable[[EvalCase], str] | None = None,
        variant_fn: Callable[[EvalCase], str] | None = None,
        label_fn: Callable[[EvalCase], str] | None = None,
    ) -> None:
        self.path = Path(path)
        self._reporter = HtmlReporter(title=title)
        self._group_fn = group_fn or _default_group
        self._variant_fn = variant_fn or _default_variant
        self._label_fn = label_fn or _default_label

    def write(self, scores: list[Score], eval_case: EvalCase) -> None:
        self._reporter.add(
            eval_case,
            scores,
            group=self._group_fn(eval_case),
            variant=self._variant_fn(eval_case),
            label=self._label_fn(eval_case),
        )

    def set_metric_categories(self, categories: dict[str, list[str]]) -> None:
        """Pass through to the underlying reporter."""
        self._reporter.set_metric_categories(categories)

    def set_overall_key(self, key: str) -> None:
        """Pass through to the underlying reporter."""
        self._reporter.set_overall_key(key)

    def set_description(self, html: str) -> None:
        """Pass through to the underlying reporter."""
        self._reporter.set_description(html)

    def set_variant_descriptions(self, descs: dict[str, str]) -> None:
        """Pass through to the underlying reporter."""
        self._reporter.set_variant_descriptions(descs)

    def set_how_to_read(self, html: str) -> None:
        """Pass through to the underlying reporter."""
        self._reporter.set_how_to_read(html)

    def finalize(self) -> str:
        """Generate the HTML report and write to file. Returns the file path."""
        self._reporter.generate(self.path)
        return str(self.path.resolve())


def _default_group(ec: EvalCase) -> str:
    if ec.tags and "group" in ec.tags:
        return ec.tags["group"]
    if isinstance(ec.input, str):
        return ec.input[:60]
    return str(ec.input)[:60]


def _default_variant(ec: EvalCase) -> str:
    if ec.tags and "variant" in ec.tags:
        return ec.tags["variant"]
    if ec.tags and "quality" in ec.tags:
        return ec.tags["quality"]
    return ""


def _default_label(ec: EvalCase) -> str:
    if ec.tags and "label" in ec.tags:
        return ec.tags["label"]
    if isinstance(ec.input, str):
        return ec.input
    return str(ec.input)
