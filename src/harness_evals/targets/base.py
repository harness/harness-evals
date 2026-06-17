"""Base target abstraction — the system under test."""

from __future__ import annotations

from abc import ABC, abstractmethod

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.golden import Golden


class BaseTarget(ABC):
    """Invokes the system under test and produces an EvalCase from a Golden.

    ``ainvoke`` is exactly the ``agent_fn`` signature that
    ``evaluate_dataset()`` expects — a BaseTarget is a drop-in.
    """

    @abstractmethod
    async def ainvoke(self, golden: Golden) -> EvalCase:
        """Run the system under test on a single golden and return an EvalCase."""

    async def close(self) -> None:
        """Release any target-owned resources (HTTP sessions, token caches)."""
        return None

    async def __aenter__(self) -> BaseTarget:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
