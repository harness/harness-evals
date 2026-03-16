from __future__ import annotations

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink


class StdoutSink(BaseSink):
    """Print scores to stdout in a human-readable format."""

    def write(self, scores: list[Score], eval_case: EvalCase) -> None:
        print(f"--- Eval: input={str(eval_case.input)[:60]!r} ---")
        for score in scores:
            status = "PASS" if score.passed else "FAIL"
            line = f"  [{status}] {score.name}: {score.value:.2f} (threshold={score.threshold})"
            if score.reason:
                line += f" — {score.reason}"
            print(line)
