from __future__ import annotations

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink
from harness_evals.summary import UNKNOWN_DIMENSION, order_dimensions, summarize


def _bar(value: float, width: int = 10) -> str:
    """A fixed-width unicode progress bar for a 0–1 value."""
    filled = round(max(0.0, min(1.0, value)) * width)
    return "█" * filled + "░" * (width - filled)


class StdoutSink(BaseSink):
    """Print scores to stdout in a human-readable format.

    When ``summary=True`` (the default), ``finalize()`` prints aggregate
    statistics (mean, pass rate) per metric after all cases are processed.
    """

    def __init__(self, *, summary: bool = True, label: str | None = None) -> None:
        self._summary = summary
        self._label = label
        self._all_scores: list[list[Score]] = []

    def write(self, scores: list[Score], eval_case: EvalCase) -> None:
        print(f"--- Eval: input={str(eval_case.input)[:60]!r} ---")
        for score in scores:
            status = "PASS" if score.passed else "FAIL"
            line = f"  [{status}] {score.name}: {score.value:.2f} (threshold={score.threshold})"
            if score.reason:
                line += f" — {score.reason}"
            print(line)
        if self._summary:
            self._all_scores.append(list(scores))

    def finalize(self) -> None:
        if not self._summary or not self._all_scores:
            return

        result = summarize(self._all_scores)
        header = f"\n=== Summary: {self._label} ===" if self._label else "\n=== Summary ==="
        print(header)
        for ms in result.by_metric.values():
            print(f"  {ms.name}: mean={ms.mean:.4f}, pass_rate={ms.pass_rate:.1%} ({ms.passed_count}/{ms.count})")
        print(f"  Quality pass rate: {result.quality_pass_rate:.1%}")

        # Safety is a hard constraint (ADR-003): surfaced separately, never
        # folded into the quality pass rate.
        has_safety = any(ds.is_safety for ds in result.by_dimension.values())
        if has_safety:
            print(f"  Safety: {result.safety_violations} violation(s), pass rate {result.safety_pass_rate:.1%}")

        # Per-dimension breakdown (ADR-009): where the target is strong/weak.
        # Canonical dimension order (shared with the HTML radar) so a run renders
        # dimensions consistently across outputs. The "unknown" bucket is excluded
        # from the block and noted in a footnote instead.
        ordered = [d for d in order_dimensions(list(result.by_dimension)) if d != UNKNOWN_DIMENSION]
        if ordered:
            print("  Dimensions:")
            for dim in ordered:
                ds = result.by_dimension[dim]
                status = f"{result.safety_violations} violation(s)" if ds.is_safety else f"pass_rate={ds.pass_rate:.0%}"
                print(f"    {ds.dimension:<13} {_bar(ds.mean)} {ds.mean:.2f}  {status}  (n={ds.metric_count})")
            # Footnote only makes sense alongside the block it annotates.
            unknown = result.by_dimension.get(UNKNOWN_DIMENSION)
            if unknown is not None:
                print(
                    f"  ({unknown.metric_count} metric(s) with no declared dimension, shown as '{UNKNOWN_DIMENSION}')"
                )
        self._all_scores.clear()
