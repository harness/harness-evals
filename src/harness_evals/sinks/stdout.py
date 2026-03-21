from __future__ import annotations

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink


class StdoutSink(BaseSink):
    """Print scores to stdout in a human-readable format.

    When ``summary=True`` (the default), ``finalize()`` prints aggregate
    statistics (mean, pass rate) per metric after all cases are processed.
    """

    def __init__(self, *, summary: bool = True) -> None:
        self._summary = summary
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

        from harness_evals.summary import summarize

        result = summarize(self._all_scores)
        print("\n=== Summary ===")
        for ms in result.by_metric.values():
            print(f"  {ms.name}: mean={ms.mean:.4f}, pass_rate={ms.pass_rate:.1%} ({ms.passed_count}/{ms.count})")
        print(f"  Overall pass rate: {result.overall_pass_rate:.1%}")
        self._all_scores.clear()
