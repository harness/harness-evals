from __future__ import annotations

from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink
from harness_evals.core.test_case import TestCase


class StdoutSink(BaseSink):
    """Print scores to stdout in a human-readable format."""

    def write(self, scores: list[Score], test_case: TestCase) -> None:
        print(f"--- Eval: input={test_case.input[:60]!r} ---")
        for score in scores:
            status = "PASS" if score.success else "FAIL"
            line = f"  [{status}] {score.name}: {score.value:.2f} (threshold={score.threshold})"
            if score.reason:
                line += f" — {score.reason}"
            print(line)
