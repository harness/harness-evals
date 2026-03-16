from __future__ import annotations

import json
from pathlib import Path

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink


class JsonSink(BaseSink):
    """Append scores as JSON lines to a file. One JSON object per write() call."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def write(self, scores: list[Score], eval_case: EvalCase) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "input": eval_case.input,
            "scores": [s.to_dict() for s in scores],
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()
