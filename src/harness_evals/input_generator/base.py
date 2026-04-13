"""Base input generation strategy with batching and deduplication."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from harness_evals.core.golden import Golden
from harness_evals.datasets import Dataset
from harness_evals.llm.base import BaseLLM

logger = logging.getLogger(__name__)

__all__ = ["BaseInputStrategy"]


class BaseInputStrategy(ABC):
    """Abstract base for input generation strategies.

    Subclasses implement ``_build_prompt``, ``_response_schema``, and
    ``_parse_response``.  The base class handles batching, LLM calls,
    and deduplication.
    """

    strategy_name: str = ""

    def __init__(self, llm: BaseLLM, batch_size: int = 10) -> None:
        self.llm = llm
        self.batch_size = batch_size

    @abstractmethod
    def _build_prompt(self, n: int, **kwargs) -> str:
        """Build the LLM prompt requesting *n* inputs."""
        ...

    @abstractmethod
    def _response_schema(self) -> dict:
        """Return the JSON schema the LLM should conform to."""
        ...

    @abstractmethod
    def _parse_response(self, response: dict, **kwargs) -> list[Golden]:
        """Convert raw LLM response into Golden objects."""
        ...

    async def generate(
        self,
        count: int,
        description: str | None = None,
        seed_inputs: list[str] | None = None,
        **kwargs,
    ) -> Dataset:
        """Generate *count* input goldens.

        Splits into batches, calls the LLM for each, and deduplicates.
        """
        schema = self._response_schema()
        goldens: list[Golden] = []

        for batch_start in range(0, count, self.batch_size):
            batch_n = min(self.batch_size, count - batch_start)
            prompt = self._build_prompt(
                n=batch_n,
                description=description,
                seed_inputs=seed_inputs,
                **kwargs,
            )
            try:
                response = await self.llm.generate_json(prompt, schema)
                parsed = self._parse_response(
                    response,
                    description=description,
                    seed_inputs=seed_inputs,
                    **kwargs,
                )
                goldens.extend(parsed)
            except Exception:
                logger.warning(
                    "LLM call failed for batch starting at %d. Skipping.",
                    batch_start,
                    exc_info=True,
                )

        goldens = self._deduplicate(goldens)

        if len(goldens) < count:
            logger.warning(
                "Generated %d inputs, fewer than requested %d.",
                len(goldens),
                count,
            )

        return goldens[:count]

    @staticmethod
    def _deduplicate(goldens: list[Golden]) -> list[Golden]:
        """Remove goldens with duplicate ``input`` values."""
        seen: set[str] = set()
        result: list[Golden] = []
        for g in goldens:
            key = str(g.input)
            if key not in seen:
                seen.add(key)
                result.append(g)
        return result
