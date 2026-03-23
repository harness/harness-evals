"""Base embedding abstraction for similarity metrics."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from harness_evals._async_compat import _run_async


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity. No numpy required."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class BaseEmbedding(ABC):
    """Minimal embedding interface for similarity metrics.

    Async-first design. Metrics pass an ``embedding: BaseEmbedding``
    parameter — no global state.
    """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors."""
        ...

    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Sync wrapper around embed(). Safe inside running event loops."""
        return _run_async(self.embed(texts))
