"""Dataset source adapter interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from harness_evals.core.golden import Golden
from harness_evals.refs import ResourceRef


class BaseDatasetSource(ABC):
    """Fetch authored goldens from an adapter-backed dataset source.

    Subclasses that require credentials or configuration at init time
    should override :meth:`from_ref` to extract what they need from the
    ``ResourceRef.extra`` dict.
    """

    name: str

    @classmethod
    def from_ref(cls, ref: ResourceRef) -> BaseDatasetSource:
        """Construct an instance using information in *ref*.

        The default implementation ignores *ref* and calls ``cls()``.
        Override this when the source needs credentials, client objects,
        or other init-time configuration derived from the ref.
        """
        return cls()

    @abstractmethod
    async def fetch(self, ref: ResourceRef) -> list[Golden]:
        """Return the dataset identified by ``ref``."""

    async def fetch_iter(self, ref: ResourceRef) -> AsyncIterator[Golden]:
        """Yield goldens lazily. Override for page-at-a-time streaming.

        Default implementation collects from :meth:`fetch`.
        Streaming-capable adapters (e.g. Langfuse, HTTP chunked) should
        override this to avoid loading all items into memory at once.
        """
        for golden in await self.fetch(ref):
            yield golden

    async def close(self) -> None:
        """Release any adapter-owned resources."""
        return None

    async def __aenter__(self) -> BaseDatasetSource:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
