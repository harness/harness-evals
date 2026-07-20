"""Anthropic Claude and OpenAI-compatible models served through AWS Bedrock.

Two clients, each subclassing its direct-API counterpart and overriding only what differs
(so request logic + token-usage recording is inherited):

- ``BedrockAnthropicLLM`` — Claude via ``anthropic.AsyncAnthropicBedrock``.
- ``BedrockOpenAILLM`` — OpenAI-compatible models via Bedrock's OpenAI-compatible endpoint
  (``https://bedrock-runtime.<region>.amazonaws.com/openai/v1``).

Both authenticate with a Bedrock **API key (bearer token)**: constructor ``api_key`` or the
``AWS_BEARER_TOKEN_BEDROCK`` env var. Region: ``aws_region`` or ``AWS_REGION`` env var.
``model`` is a Bedrock model id / inference-profile id or ARN.
"""

from __future__ import annotations

import json
import os
import re

from harness_evals.llm.anthropic import AnthropicLLM
from harness_evals.llm.openai import OpenAILLM


class BedrockAnthropicLLM(AnthropicLLM):
    """Anthropic Claude accessed through **AWS Bedrock**. Requires ``pip install harness-evals[llm]``.

    Reuses :class:`AnthropicLLM`'s request logic (``generate`` / ``generate_json`` with the
    Anthropic ``output_config`` structured-output path and token-usage recording); only the
    underlying client and auth differ.

    **Auth is Bedrock API key (bearer token) only** — via ``api_key`` or the
    ``AWS_BEARER_TOKEN_BEDROCK`` env var. It does **not** use ``ANTHROPIC_API_KEY``, and it does
    not support AWS IAM/SigV4 credentials (that path needs the ``anthropic[bedrock]``/boto3
    extra, which is intentionally not required). Region: ``aws_region`` or ``AWS_REGION``.
    """

    def __init__(
        self,
        model: str = "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        *,
        aws_region: str | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> None:
        try:
            import anthropic  # noqa: F811
        except ImportError as e:
            raise ImportError("Install anthropic: pip install harness-evals[llm]") from e

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k

        # Bearer-only: require the Bedrock API key up front. This keeps auth to the bearer
        # token (no AWS IAM/SigV4, which would need boto3) and fails fast with a clear message
        # rather than deferring to a confusing call-time error.
        bearer = api_key or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        if not bearer:
            raise ValueError("No Bedrock API key: pass api_key= or set AWS_BEARER_TOKEN_BEDROCK")
        region = aws_region or os.environ.get("AWS_REGION")
        client_kwargs: dict = {"api_key": bearer}
        if region:
            client_kwargs["aws_region"] = region
        self._client = anthropic.AsyncAnthropicBedrock(**client_kwargs)


_REASONING_RE = re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json_object(text: str) -> dict:
    """Best-effort extraction of a single JSON object from model text.

    Handles: clean JSON, markdown-fenced JSON, and JSON preceded by reasoning/prose
    (as gpt-oss models on Bedrock emit).
    """
    cleaned = _REASONING_RE.sub("", text).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    fence = _FENCE_RE.search(cleaned)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    # Last resort: scan for the first balanced {...} object and parse it.
    start = cleaned.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = cleaned.find("{", start + 1)

    raise json.JSONDecodeError("No JSON object found in Bedrock OpenAI response", text, 0)


class BedrockOpenAILLM(OpenAILLM):
    """OpenAI-compatible model accessed through AWS Bedrock. Requires ``pip install harness-evals[llm]``.

    Reuses :class:`OpenAILLM`'s ``generate`` (chat completions) and token-usage recording;
    overrides only the client construction (Bedrock base URL + bearer auth) and ``generate_json``
    (prompt-appended schema + robust extraction, since Bedrock doesn't enforce json_schema).
    """

    def __init__(
        self,
        model: str = "openai.gpt-oss-120b-1:0",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        *,
        aws_region: str | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
    ) -> None:
        try:
            import openai  # noqa: F811
        except ImportError as e:
            raise ImportError("Install openai: pip install harness-evals[llm]") from e

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

        key = api_key or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        if not key:
            # Require the Bedrock bearer explicitly. If we passed api_key=None (or omitted it),
            # AsyncOpenAI would silently fall back to OPENAI_API_KEY and send a direct-OpenAI key
            # as a Bedrock bearer, producing a confusing 401 at call time. Bedrock's
            # OpenAI-compatible endpoint is bearer-only, so fail fast with a clear message.
            raise ValueError("No Bedrock API key: pass api_key= or set AWS_BEARER_TOKEN_BEDROCK")
        region = aws_region or os.environ.get("AWS_REGION") or "us-east-1"
        base_url = f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1"
        self._client = openai.AsyncOpenAI(api_key=key, base_url=base_url)

    async def generate_json(self, prompt: str, schema: dict, **kwargs: object) -> dict:
        instruction = (
            f"{prompt}\n\nRespond with ONLY a single JSON object matching this schema "
            f"(no markdown, no commentary, no <reasoning> tags):\n{json.dumps(schema)}"
        )
        text = await self.generate(instruction, **kwargs)
        return _extract_json_object(text)
