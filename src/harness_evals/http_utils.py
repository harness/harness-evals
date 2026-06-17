"""Shared helpers for HTTP-backed adapters."""

from __future__ import annotations

from harness_evals.refs import ResourceRef


def ref_to_url(ref: ResourceRef) -> str:
    """Rebuild a URL from a resolved HTTP(S) resource ref without changing scheme."""

    if ref.id.startswith(("http://", "https://")):
        return ref.id
    return f"{ref.source}://{ref.id}"
