"""HTTP prompt source."""

from __future__ import annotations

import asyncio
from urllib.request import Request, urlopen

from harness_evals.http_utils import ref_to_url
from harness_evals.plugins import register_prompt_source
from harness_evals.prompts.base import BasePromptSource
from harness_evals.prompts.template import PromptTemplate, infer_input_variables
from harness_evals.refs import ResourceRef


@register_prompt_source("https")
@register_prompt_source("http")
class HttpPromptSource(BasePromptSource):
    """Fetch a prompt template over HTTP(S).

    Uses stdlib urllib intentionally to keep source adapters dependency-free.
    HTTPS requests rely on Python's default TLS certificate verification.
    """

    name = "http"

    def __init__(self, timeout_s: float = 60.0) -> None:
        self.timeout_s = timeout_s

    async def fetch(self, ref: ResourceRef) -> PromptTemplate:
        template = await asyncio.to_thread(_fetch_text, ref_to_url(ref), self.timeout_s)
        return PromptTemplate(
            template=template,
            input_variables=infer_input_variables(template),
            version=ref.version,
            metadata=dict(ref.extra),
        )


def _fetch_text(url: str, timeout_s: float) -> str:
    request = Request(url, headers={"Accept": "text/plain, application/json"})
    with urlopen(request, timeout=timeout_s) as response:
        return response.read().decode("utf-8")
