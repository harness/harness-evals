"""Prompt source adapter interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from harness_evals.prompts.template import PromptTemplate
from harness_evals.refs import ResourceRef


class BasePromptSource(ABC):
    """Fetch prompt templates from an adapter-backed prompt source."""

    name: str

    @abstractmethod
    async def fetch(self, ref: ResourceRef) -> PromptTemplate:
        """Return the prompt template identified by ``ref``."""

    async def close(self) -> None:
        """Release any adapter-owned resources."""
        return None

    async def __aenter__(self) -> BasePromptSource:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
