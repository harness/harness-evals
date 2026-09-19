"""Base decision-provider abstraction for typed, calibrated decision primitives."""

from __future__ import annotations

from abc import ABC, abstractmethod

from harness_evals._async_compat import _run_async
from harness_evals.decision.types import DecisionResponse, Question


class BaseDecisionProvider(ABC):
    """Minimal interface for a System-One-style decision provider.

    Async-first, batched: one call answers many named questions about one
    ``state``. Deliberately not a ``BaseLLM`` — the request/response shape
    (typed multi-question batch in, typed multi-answer batch out) does not
    fit prompt-in/text-out.

    Transport/API errors must propagate out of ``a_ask()`` unchanged — never
    caught and converted into a fabricated ``DecisionResponse``.
    """

    @abstractmethod
    async def a_ask(self, state: str | dict | list, questions: dict[str, Question]) -> DecisionResponse:
        """Answer ``questions`` about ``state`` in one batched call."""
        ...

    def ask(self, state: str | dict | list, questions: dict[str, Question]) -> DecisionResponse:
        """Sync wrapper around ``a_ask()``. Safe inside running event loops."""
        return _run_async(self.a_ask(state, questions))
