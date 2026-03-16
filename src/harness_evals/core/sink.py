from __future__ import annotations

from abc import ABC, abstractmethod

from harness_evals.core.score import Score
from harness_evals.core.test_case import TestCase


class BaseSink(ABC):
    """Base class for output destinations.

    Sinks receive scores after evaluation and write them somewhere:
    stdout, JSON file, JUnit XML, CSV, remote API, etc.
    """

    @abstractmethod
    def write(self, scores: list[Score], test_case: TestCase) -> None:
        """Emit scores for a single test case to the output destination."""
        ...
