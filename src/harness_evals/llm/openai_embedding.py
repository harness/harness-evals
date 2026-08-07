"""OpenAI embedding provider."""

from __future__ import annotations

import os

from harness_evals.llm.embedding import BaseEmbedding


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI-backed embeddings. Requires ``pip install harness-evals[llm]``.

    API key resolution: constructor ``api_key`` > ``OPENAI_API_KEY`` env var.
    Pass ``base_url`` to target an OpenAI-compatible endpoint (e.g. Bedrock's
    ``https://bedrock-runtime.<region>.amazonaws.com/openai/v1`` or the Harness
    LLM gateway) instead of ``platform.openai.com``.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            import openai  # noqa: F811
        except ImportError as e:
            raise ImportError("Install openai: pip install harness-evals[llm]") from e

        self.model = model
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("No API key: pass api_key= or set OPENAI_API_KEY")
        client_kwargs: dict = {"api_key": resolved_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = openai.AsyncOpenAI(**client_kwargs)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]
