"""Langfuse prompt source — fetch prompt templates from Langfuse registry.

Requires: pip install harness-evals[langfuse]
"""

from __future__ import annotations

import asyncio
from typing import Any

try:
    from langfuse import Langfuse
except ImportError as _err:
    raise ImportError(
        "LangfusePromptSource requires the langfuse package. Install with: pip install harness-evals[langfuse]"
    ) from _err

from harness_evals._langfuse_compat import flush_langfuse_client, strip_ref_prefix
from harness_evals.plugins import register_prompt_source
from harness_evals.prompts.base import BasePromptSource
from harness_evals.prompts.template import PromptTemplate, infer_input_variables
from harness_evals.refs import ResourceRef


@register_prompt_source("langfuse")
class LangfusePromptSource(BasePromptSource):
    """Fetch a prompt template from Langfuse's prompt registry.

    URI forms::

        langfuse://prompts/support-bot@5
        langfuse://support-bot

    Dict form::

        {source: langfuse, id: support-bot, version: 5}

    ``ref.extra`` keys:

    - ``label``: str — fetch by label (``"production"``, ``"staging"``)
      instead of version number.
    - ``type``: ``"text"`` (default) or ``"chat"`` — prompt type.
      Chat prompts raise ``ValueError`` in v1.

    Langfuse prompts natively use ``{{var}}`` syntax — no conversion needed.
    """

    name = "langfuse"

    def __init__(self, client: Langfuse | None = None) -> None:
        self._client = client or Langfuse()

    async def close(self) -> None:
        """Flush the Langfuse client to prevent data loss."""
        await flush_langfuse_client(self._client)

    async def fetch(self, ref: ResourceRef) -> PromptTemplate:
        """Fetch a prompt by name, with optional version or label."""
        if not ref.id:
            raise ValueError("LangfusePromptSource requires a prompt name in ref.id")
        name = strip_ref_prefix(ref.id, "prompts/")
        version = int(ref.version) if ref.version and ref.version.isdigit() else None
        label = ref.extra.get("label")
        prompt_type = ref.extra.get("type", "text")
        return await asyncio.to_thread(self._fetch_sync, name, version, label, prompt_type)

    def _fetch_sync(
        self,
        name: str,
        version: int | None,
        label: str | None,
        prompt_type: str,
    ) -> PromptTemplate:
        if prompt_type != "text":
            raise ValueError(
                f"LangfusePromptSource only supports type='text' in v1, got {prompt_type!r}. "
                "Chat prompt support is planned for v2."
            )

        kwargs: dict[str, Any] = {"type": "text"}
        if version is not None:
            kwargs["version"] = version
        elif label is not None:
            kwargs["label"] = label

        prompt = self._client.get_prompt(name, **kwargs)

        model_hint = _extract_model_hint(prompt.config) if prompt.config else None

        return PromptTemplate(
            template=prompt.prompt,
            input_variables=infer_input_variables(prompt.prompt),
            model_hint=model_hint,
            version=str(prompt.version),
            metadata={
                "langfuse_prompt_name": prompt.name,
                "langfuse_prompt_labels": prompt.labels,
                "langfuse_prompt_tags": prompt.tags,
            },
        )


def _extract_model_hint(config: dict[str, Any]) -> dict[str, Any] | None:
    """Extract model configuration from Langfuse prompt config into model_hint format."""
    model_name = config.get("model") or config.get("modelName")
    if not model_name:
        return None
    hint: dict[str, Any] = {"name": model_name}
    provider = config.get("provider")
    if provider:
        hint["provider"] = provider
    params = {k: v for k, v in config.items() if k not in ("model", "modelName", "provider")}
    if params:
        hint["params"] = params
    return hint
