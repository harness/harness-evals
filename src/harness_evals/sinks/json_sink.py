from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink
from harness_evals.core.test_case import TestCase


class JsonSink(BaseSink):
    """Append scores as JSON lines to a file. One JSON object per write() call."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def write(self, scores: list[Score], test_case: TestCase) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "input": test_case.input,
            "scores": [dataclasses.asdict(s) for s in scores],
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
