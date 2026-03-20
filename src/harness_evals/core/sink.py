from __future__ import annotations

from abc import ABC, abstractmethod

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score


class BaseSink(ABC):
    """Base class for output destinations.

    Sinks receive scores after evaluation and write them somewhere:
    stdout, JSON file, JUnit XML, CSV, remote API, etc.

    Subclasses that buffer output (e.g. JUnitSink) should override
    ``finalize()`` to flush. Subclasses that hold external resources
    (e.g. OtlpSink) should override ``shutdown()`` to release them.
    """

    @abstractmethod
    def write(self, scores: list[Score], eval_case: EvalCase) -> None:
        """Emit scores for a single eval case to the output destination."""
        ...

    def finalize(self) -> None:
        """Flush any buffered output. No-op by default."""
        return  # noqa: B027

    def shutdown(self) -> None:
        """Release external resources. No-op by default."""
        return  # noqa: B027
