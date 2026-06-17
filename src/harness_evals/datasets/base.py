"""Dataset source adapter interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from harness_evals.core.golden import Golden
from harness_evals.refs import ResourceRef


class BaseDatasetSource(ABC):
    """Fetch authored goldens from an adapter-backed dataset source."""

    name: str

    @abstractmethod
    async def fetch(self, ref: ResourceRef) -> list[Golden]:
        """Return the dataset identified by ``ref``."""

    async def close(self) -> None:
        """Release any adapter-owned resources."""
        return None

    async def __aenter__(self) -> BaseDatasetSource:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
