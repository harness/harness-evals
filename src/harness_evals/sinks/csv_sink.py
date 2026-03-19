"""CSV sink — append scores to a CSV file, one row per (eval_case, metric) pair."""

from __future__ import annotations

import csv
from pathlib import Path

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink

_FIELDNAMES = [
    "input",
    "metric",
    "value",
    "threshold",
    "passed",
    "reason",
    "created_at",
]


class CsvSink(BaseSink):
    """Append scores to a CSV file. One row per (eval_case, metric) pair.

    Creates the file with a header row on first write. Subsequent writes
    append rows without repeating the header.
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def write(self, scores: list[Score], eval_case: EvalCase) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.path.exists() and self.path.stat().st_size > 0

        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            input_preview = str(eval_case.input)[:120]
            for score in scores:
                writer.writerow(
                    {
                        "input": input_preview,
                        "metric": score.name,
                        "value": f"{score.value:.4f}",
                        "threshold": f"{score.threshold:.4f}",
                        "passed": score.passed,
                        "reason": score.reason or "",
                        "created_at": score.created_at.isoformat(),
                    }
                )
