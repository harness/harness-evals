"""Langfuse dataset source — fetch authored golden datasets from Langfuse registry.

Requires: pip install harness-evals[langfuse]
"""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

try:
    from langfuse import Langfuse
except ImportError as _err:
    raise ImportError(
        "LangfuseDatasetSource requires the langfuse package. Install with: pip install harness-evals[langfuse]"
    ) from _err

from harness_evals._langfuse_compat import flush_langfuse_client, strip_ref_prefix
from harness_evals.core.golden import Golden
from harness_evals.datasets.base import BaseDatasetSource
from harness_evals.plugins import register_dataset_source
from harness_evals.refs import ResourceRef

_DEFAULT_PAGE_SIZE = 50


@register_dataset_source("langfuse")
class LangfuseDatasetSource(BaseDatasetSource):
    """Fetch a dataset (list of goldens) from Langfuse's dataset registry.

    URI forms::

        langfuse://datasets/my-dataset@2024-06-01T00:00:00Z
        langfuse://my-dataset

    Dict form::

        {source: langfuse, id: my-dataset, version: "2024-06-01T00:00:00Z"}

    ``ref.extra`` keys:

    - ``fetch_items_page_size``: int (default 50)

    Mapping:

    - ``DatasetItem.input``           → ``Golden.input``
    - ``DatasetItem.expected_output`` → ``Golden.expected``
    - ``DatasetItem.metadata``        → ``Golden.metadata`` (extended with provenance)
    - ARCHIVED items are skipped.
    """

    name = "langfuse"

    def __init__(self, client: Langfuse | None = None) -> None:
        self._client_arg = client

    @property
    def _client(self) -> Langfuse:
        if self._client_arg is None:
            try:
                self._client_arg = Langfuse()
            except Exception as exc:
                raise ImportError(
                    "No Langfuse client provided and LANGFUSE_PUBLIC_KEY is not set. "
                    "Either pass a Langfuse() client explicitly or set LANGFUSE_PUBLIC_KEY "
                    "and LANGFUSE_SECRET_KEY environment variables."
                ) from exc
        return self._client_arg

    async def close(self) -> None:
        """Flush the Langfuse client to prevent data loss."""
        if self._client_arg is not None:
            await flush_langfuse_client(self._client_arg)

    async def fetch(self, ref: ResourceRef) -> list[Golden]:
        """Fetch all dataset items into memory. For large datasets, prefer fetch_iter().

        Note: this implementation delegates to fetch_iter() (inverting the base
        class default where fetch_iter delegates to fetch). Both methods are
        overridden here, so no recursion occurs.
        """
        return [g async for g in self.fetch_iter(ref)]

    async def fetch_iter(self, ref: ResourceRef) -> AsyncIterator[Golden]:
        """Yield goldens page-by-page without loading the full dataset into memory.

        Uses the low-level ``dataset_items.list()`` API for pagination.

        Usage (note: no ``await`` — use ``async for`` directly)::

            async for golden in source.fetch_iter(ref):
                print(golden.input)
        """
        if not ref.id:
            raise ValueError("LangfuseDatasetSource requires a dataset name in ref.id")

        name = strip_ref_prefix(ref.id, "datasets/")
        page_size = int(ref.extra.get("fetch_items_page_size", _DEFAULT_PAGE_SIZE))
        dt_version = _parse_version_datetime(ref.version) if ref.version else None

        page_num = 1
        while True:
            page = await asyncio.to_thread(
                self._client.api.dataset_items.list,
                dataset_name=name,
                page=page_num,
                limit=page_size,
                version=dt_version,
            )
            items = page.data if hasattr(page, "data") else []
            if not items:
                break

            for item in items:
                if _is_active(item):
                    yield _item_to_golden(item)

            meta = getattr(page, "meta", None)
            total_pages = getattr(meta, "total_pages", None) if meta is not None else None
            # If the SDK omits pagination metadata, stop when a page returns
            # fewer items than requested rather than crashing on page.meta.
            if total_pages is None:
                if len(items) < page_size:
                    break
            elif page_num >= total_pages:
                break
            page_num += 1


def _parse_version_datetime(version: str) -> datetime | None:
    """Parse version string as ISO datetime for Langfuse dataset versioning.

    Returns None (fetches latest) if the string doesn't parse, with a warning.
    """
    try:
        dt = datetime.fromisoformat(version)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        warnings.warn(
            f"LangfuseDatasetSource: version {version!r} is not a valid ISO datetime; fetching latest.",
            UserWarning,
            stacklevel=2,
        )
        return None


def _is_active(item: Any) -> bool:
    """Return True if the dataset item is not archived."""
    status = getattr(item, "status", None)
    if status is None:
        return True
    status_str = status if isinstance(status, str) else getattr(status, "value", str(status))
    return status_str.upper() != "ARCHIVED"


def _item_to_golden(item: Any) -> Golden:
    """Map a Langfuse DatasetItem to a Golden."""
    raw_meta = getattr(item, "metadata", None)
    metadata: dict[str, Any] = dict(raw_meta) if isinstance(raw_meta, dict) else {}
    metadata["langfuse_dataset_item_id"] = item.id
    source_trace = getattr(item, "source_trace_id", None)
    if source_trace:
        metadata["langfuse_source_trace_id"] = source_trace

    return Golden(
        input=item.input,
        expected=getattr(item, "expected_output", None),
        metadata=metadata or None,
    )
