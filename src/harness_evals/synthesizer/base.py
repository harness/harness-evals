"""Base synthesizer abstraction with chunking, batching, and deduplication."""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod

from harness_evals._async_compat import _run_async
from harness_evals.core.golden import Golden
from harness_evals.datasets import Dataset
from harness_evals.llm.base import BaseLLM

logger = logging.getLogger(__name__)

__all__ = ["BaseSynthesizer"]

_VALID_DIFFICULTIES = frozenset({"easy", "medium", "hard", "mixed"})


class BaseSynthesizer(ABC):
    """Abstract base for task-type-specific golden dataset generators.

    Subclasses implement ``_build_prompt``, ``_response_schema``, and
    ``_parse_response``.  The base class handles document chunking,
    distributing the requested count across chunks, calling the LLM,
    deduplication, and sync wrappers.
    """

    task_type: str = ""

    def __init__(
        self,
        llm: BaseLLM,
        max_chunk_chars: int = 4000,
        batch_size: int = 10,
    ) -> None:
        self.llm = llm
        self.max_chunk_chars = max_chunk_chars
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Abstract interface – subclasses implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_prompt(self, chunk: str, n: int, difficulty: str) -> str:
        """Build the LLM prompt for a single document chunk."""
        ...

    @abstractmethod
    def _response_schema(self) -> dict:
        """Return the JSON schema the LLM should conform to."""
        ...

    @abstractmethod
    def _parse_response(
        self,
        response: dict,
        doc_index: int,
        difficulty: str,
    ) -> list[Golden]:
        """Convert the raw LLM response dict into a list of Goldens."""
        ...

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        documents: list[str],
        n: int = 20,
        difficulty: str = "mixed",
    ) -> Dataset:
        """Generate *n* goldens from *documents*.

        Returns a ``Dataset`` (``list[Golden]``).  Failed LLM calls for
        individual chunks are logged and skipped.
        """
        if difficulty not in _VALID_DIFFICULTIES:
            raise ValueError(
                f"Invalid difficulty {difficulty!r}. Choose from: {', '.join(sorted(_VALID_DIFFICULTIES))}"
            )

        chunks = self._chunk_documents(documents, self.max_chunk_chars)
        if not chunks:
            return []

        allocation = self._distribute(n, chunks)
        schema = self._response_schema()

        goldens: list[Golden] = []
        for (doc_idx, chunk_text), chunk_n in zip(chunks, allocation, strict=True):
            if chunk_n <= 0:
                continue
            for batch_start in range(0, chunk_n, self.batch_size):
                batch_n = min(self.batch_size, chunk_n - batch_start)
                prompt = self._build_prompt(chunk_text, batch_n, difficulty)
                try:
                    response = await self._call_llm(prompt, schema)
                    parsed = self._parse_response(response, doc_idx, difficulty)
                    goldens.extend(parsed)
                except Exception:
                    logger.warning(
                        "LLM call failed for chunk (doc=%d, batch_start=%d). Skipping.",
                        doc_idx,
                        batch_start,
                        exc_info=True,
                    )

        goldens = self._deduplicate(goldens)

        if len(goldens) < n:
            logger.warning(
                "Generated %d goldens, fewer than requested %d.",
                len(goldens),
                n,
            )

        return goldens[:n]

    def generate_sync(
        self,
        documents: list[str],
        n: int = 20,
        difficulty: str = "mixed",
    ) -> Dataset:
        """Synchronous wrapper around :meth:`generate`."""
        return _run_async(self.generate(documents, n, difficulty))

    # ------------------------------------------------------------------
    # LLM call — override for non-strict schemas
    # ------------------------------------------------------------------

    async def _call_llm(self, prompt: str, schema: dict) -> dict:
        """Call the LLM and return a parsed dict.

        Default uses ``generate_json()`` (structured output with strict
        schema).  Subclasses can override this — e.g. to use plain
        ``generate()`` when the response contains free-form objects that
        strict schemas cannot represent.
        """
        return await self.llm.generate_json(prompt, schema)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_documents(
        documents: list[str],
        max_chars: int = 4000,
    ) -> list[tuple[int, str]]:
        """Split *documents* into ``(doc_index, chunk_text)`` tuples.

        Splitting happens at paragraph boundaries (double newline).  Chunks
        shorter than *max_chars* are returned as-is.
        """
        chunks: list[tuple[int, str]] = []
        for doc_idx, doc in enumerate(documents):
            if not doc.strip():
                continue
            if len(doc) <= max_chars:
                chunks.append((doc_idx, doc))
                continue

            paragraphs = doc.split("\n\n")
            current: list[str] = []
            current_len = 0

            for para in paragraphs:
                para_len = len(para)
                separator_len = 2 if current else 0
                if current and current_len + separator_len + para_len > max_chars:
                    chunks.append((doc_idx, "\n\n".join(current)))
                    current = [para]
                    current_len = para_len
                else:
                    current.append(para)
                    current_len += separator_len + para_len

            if current:
                chunks.append((doc_idx, "\n\n".join(current)))

        return chunks

    @staticmethod
    def _distribute(n: int, chunks: list[tuple[int, str]]) -> list[int]:
        """Distribute *n* items across *chunks* proportionally by length.

        When there are more chunks than *n*, only the *n* longest chunks
        each receive 1 item.
        """
        num_chunks = len(chunks)
        if num_chunks == 0:
            return []

        total_len = sum(len(text) for _, text in chunks)
        if total_len == 0:
            return [0] * num_chunks

        if n <= num_chunks:
            ranked = sorted(range(num_chunks), key=lambda i: len(chunks[i][1]), reverse=True)
            allocation = [0] * num_chunks
            for idx in ranked[:n]:
                allocation[idx] = 1
            return allocation

        allocation = [0] * num_chunks
        assigned = 0
        for i, (_, text) in enumerate(chunks):
            share = (len(text) / total_len) * n
            allocation[i] = max(1, math.floor(share))
            assigned += allocation[i]

        remainder = n - assigned
        for i in range(remainder):
            allocation[i % num_chunks] += 1

        return allocation

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
