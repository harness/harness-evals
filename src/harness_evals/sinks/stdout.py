from __future__ import annotations

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink
from harness_evals.summary import summarize


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
        if result.by_dimension:
            print("  Dimensions:")
            for ds in result.by_dimension.values():
                status = (
                    f"{result.safety_violations} violation(s)"
                    if ds.is_safety
                    else f"pass_rate={ds.pass_rate:.0%}"
                )
                print(f"    {ds.dimension:<13} mean={ds.mean:.2f}  {status}  (n={ds.metric_count})")
        self._all_scores.clear()
