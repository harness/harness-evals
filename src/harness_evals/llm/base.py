"""Base LLM abstraction for judge metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod

from harness_evals._async_compat import _run_async


class BaseLLM(ABC):
    """Minimal LLM interface for evaluation metrics.

    Async-first design. Metrics call ``generate()`` or ``generate_json()``
    and pass an ``llm: BaseLLM`` parameter — no global state.
    """

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: object) -> str:
        """Send prompt to LLM, return completion text."""
        ...

    @abstractmethod
    async def generate_json(self, prompt: str, schema: dict, **kwargs: object) -> dict:
        """Send prompt, parse response as JSON conforming to schema."""
        ...

    def generate_sync(self, prompt: str, **kwargs: object) -> str:
        """Sync wrapper around generate(). Safe inside running event loops."""
        return _run_async(self.generate(prompt, **kwargs))

    def generate_json_sync(self, prompt: str, schema: dict, **kwargs: object) -> dict:
        """Sync wrapper around generate_json(). Safe inside running event loops."""
        return _run_async(self.generate_json(prompt, schema, **kwargs))
