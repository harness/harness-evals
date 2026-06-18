"""Shared utilities for Langfuse adapters."""

from __future__ import annotations

import contextlib


async def flush_langfuse_client(client: object) -> None:
    """Flush a Langfuse client, suppressing errors.

    Shared by LangfuseDatasetSource, LangfusePromptSource, and
    LangfuseEvalCaseSource to ensure consistent resource cleanup.
    """
    flush = getattr(client, "flush", None)
    if callable(flush):
        with contextlib.suppress(Exception):
            flush()


def strip_ref_prefix(ref_id: str, prefix: str) -> str:
    """Strip an optional URI path prefix from a ResourceRef id.

    When a URI like ``langfuse://datasets/my-dataset`` is resolved, the
    ``ref.id`` becomes ``"datasets/my-dataset"``. This helper strips the
    leading ``"datasets/"`` (or ``"prompts/"``) so the Langfuse API
    receives just the resource name.
    """
    return ref_id[len(prefix) :] if ref_id.startswith(prefix) else ref_id
