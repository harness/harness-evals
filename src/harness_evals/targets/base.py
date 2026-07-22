"""Base target abstraction — the system under test."""

from __future__ import annotations

from abc import ABC, abstractmethod

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.golden import Golden
from harness_evals.core.types import Message


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


class ConversationTarget(ABC):
    """Invokes a conversational system-under-test one assistant turn at a time."""

    @abstractmethod
    async def agenerate(
        self,
        messages: list[Message],
        human_input: dict | None = None,
        *,
        system_event: dict | None = None,
    ) -> Message:
        """Return the agent's next assistant message for a conversation history.

        ``human_input`` carries a protocol-specific continuation payload when the
        agent is waiting for human input. ``system_event`` is a deprecated alias.
        """
