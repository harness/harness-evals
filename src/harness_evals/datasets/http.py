"""HTTP dataset source."""

from __future__ import annotations

import asyncio
from urllib.request import Request, urlopen

from harness_evals.core.golden import Golden
from harness_evals.datasets.base import BaseDatasetSource
from harness_evals.datasets.io import loads_dataset
from harness_evals.http_utils import ref_to_url
from harness_evals.plugins import register_dataset_source
from harness_evals.refs import ResourceRef


@register_dataset_source("https")
@register_dataset_source("http")
class HttpDatasetSource(BaseDatasetSource):
    """Fetch a JSON or JSONL dataset over HTTP(S).

    Uses stdlib urllib intentionally to keep source adapters dependency-free.
    HTTPS requests rely on Python's default TLS certificate verification.
    """

    name = "http"

    def __init__(self, timeout_s: float = 60.0) -> None:
        self.timeout_s = timeout_s

    async def fetch(self, ref: ResourceRef) -> list[Golden]:
        body, content_type = await asyncio.to_thread(_fetch_text, ref_to_url(ref), self.timeout_s)
        return loads_dataset(body, format=_detect_format(ref, body, content_type))


def _fetch_text(url: str, timeout_s: float) -> tuple[str, str]:
    request = Request(url, headers={"Accept": "application/json, application/x-ndjson, text/plain"})
    with urlopen(request, timeout=timeout_s) as response:
        content_type = _content_type(response)
        body = response.read().decode(_charset(content_type))
        return body, content_type


def _charset(content_type: str) -> str:
    """Extract the charset from a Content-Type header, defaulting to utf-8."""
    for part in content_type.split(";"):
        part = part.strip()
        if part.startswith("charset="):
            charset = part[len("charset=") :].strip().strip('"')
            if charset:
                return charset
    return "utf-8"


def _content_type(response: object) -> str:
    headers = getattr(response, "headers", None)
    if headers is None:
        return ""
    content_type = headers.get("Content-Type", "")
    return str(content_type).lower()


def _detect_format(ref: ResourceRef, body: str, content_type: str) -> str:
    explicit_format = ref.extra.get("format")
    if explicit_format is not None:
        return str(explicit_format)
    if "jsonl" in content_type or "ndjson" in content_type:
        return "jsonl"
    if "json" in content_type:
        return "json"
    trimmed = body.lstrip()
    if trimmed.startswith("["):
        return "json"
    if trimmed.startswith("{"):
        raise ValueError("Dataset must be a JSON array, got a JSON object")
    return "jsonl"
