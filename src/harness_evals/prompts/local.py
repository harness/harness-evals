"""Local filesystem prompt source."""

from __future__ import annotations

import asyncio
from pathlib import Path

from harness_evals.plugins import register_prompt_source
from harness_evals.prompts.base import BasePromptSource
from harness_evals.prompts.template import PromptTemplate, infer_input_variables
from harness_evals.refs import ResourceRef


@register_prompt_source("local")
class LocalPromptSource(BasePromptSource):
    """Fetch a prompt template from a local text file."""

    name = "local"

    async def fetch(self, ref: ResourceRef) -> PromptTemplate:
        template = await asyncio.to_thread(_read_prompt, ref.id)
        return PromptTemplate(
            template=template,
            input_variables=infer_input_variables(template),
            version=ref.version,
            metadata=dict(ref.extra),
        )


def _read_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")
