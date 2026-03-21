"""BaselineStore ABC — persistence interface for baseline score snapshots."""

from __future__ import annotations

from abc import ABC, abstractmethod

from harness_evals.core.score import Score


class BaselineStore(ABC):
    """Abstract interface for saving and loading baseline score snapshots.

    A baseline is a ``dict[str, list[Score]]`` mapping metric names to
    the list of per-case scores from a single evaluation run.  Stores
    must support saving by run ID and loading either a specific run or
    the most recent one.
    """

    @abstractmethod
    def save(self, run_id: str, scores: dict[str, list[Score]]) -> None:
        """Persist scores for a given run.

        Args:
            run_id: Unique identifier for this evaluation run.
            scores: Metric name -> list of per-case scores.
        """
        ...

    @abstractmethod
    def load(self, run_id: str | None = None) -> dict[str, list[Score]]:
        """Load scores for a run.

        Args:
            run_id: Run to load. ``None`` loads the most recent run.

        Raises:
            FileNotFoundError: If the requested run (or any run when
                ``run_id`` is None) does not exist.
        """
        ...

    @abstractmethod
    def list_runs(self) -> list[str]:
        """Return all saved run IDs in chronological order (oldest first)."""
        ...
