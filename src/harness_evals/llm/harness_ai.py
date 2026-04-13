"""Harness AI Service LLM provider.

Routes LLM calls through the Harness AI Service gateway, which proxies to
OpenAI/Anthropic/etc with a unified API. Auth via HS256 JWT.

Requires ``PyJWT`` and ``requests``::

    pip install harness-evals[harness]

Environment variables:
    HARNESS_AI_SERVICE_URL: Gateway base URL (default: http://localhost:8001)
    HARNESS_AI_SERVICE_SECRET: HS256 signing secret for JWT auth
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time

from harness_evals.llm.base import BaseLLM


def _generate_jwt(secret: bytes, *, service_name: str = "STO") -> str:
    """Generate a Harness AI service JWT (HS256).

    ``service_name`` sets both the ``sub`` and ``name`` claims. The Harness AI
    gateway validates the client identity against registered service names.
    """
    try:
        import jwt
    except ImportError as e:
        raise ImportError("Install PyJWT: pip install harness-evals[harness]") from e

    now = time.time()
    claims = {
        "iss": "Harness Inc",
        "sub": service_name,
        "iat": int(now),
        "nbf": int(now - 60),
        "exp": int(now + 3600),
        "type": "SERVICE",
        "name": service_name,
    }
    return jwt.encode(claims, secret, algorithm="HS256")


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _extract_json(text: str) -> dict:
    """Parse JSON from LLM text, stripping markdown fences if present."""
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = _JSON_FENCE_RE.search(text)
    if m:
        return json.loads(m.group(1).strip())

    raise json.JSONDecodeError("No valid JSON found in LLM response", text, 0)


class HarnessAILLM(BaseLLM):
    """LLM backed by the Harness AI chat gateway.

    API key resolution: constructor ``secret`` > ``HARNESS_AI_SERVICE_SECRET`` env var.
    URL resolution: constructor ``base_url`` > ``HARNESS_AI_SERVICE_URL`` env var.
    """

    def __init__(
        self,
        base_url: str | None = None,
        secret: str | None = None,
        model: str = "gpt-4o",
        provider: str = "openai",
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
    ) -> None:
        try:
            import jwt  # noqa: F401
            import requests  # noqa: F401
        except ImportError as e:
            raise ImportError("Install dependencies: pip install harness-evals[harness]") from e

        self.base_url = (base_url or os.environ.get("HARNESS_AI_SERVICE_URL", "http://localhost:8001")).rstrip("/")
        secret_str = secret or os.environ.get("HARNESS_AI_SERVICE_SECRET", "")
        if not secret_str:
            raise ValueError("No secret: pass secret= or set HARNESS_AI_SERVICE_SECRET")
        self.secret = secret_str.encode() if isinstance(secret_str, str) else secret_str
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def _call_chat(self, prompt: str) -> dict:
        """Send a chat request to the Harness AI gateway and return the parsed response."""
        import requests

        token = _generate_jwt(self.secret)
        resp = requests.post(
            f"{self.base_url}/chat",
            json={
                "provider": self.provider,
                "model_name": self.model,
                "message": prompt,
                "model_parameters": {
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                    "top_p": 0,
                    "top_k": 0,
                },
                "examples": [],
            },
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            timeout=120,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"Harness AI Service returned {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        if data.get("blocked"):
            raise RuntimeError("Harness AI Service blocked the response")

        return data

    async def generate(self, prompt: str, **kwargs: object) -> str:
        data = await asyncio.to_thread(self._call_chat, prompt)
        return data.get("text", "")

    async def generate_json(self, prompt: str, schema: dict, **kwargs: object) -> dict:
        data = await asyncio.to_thread(self._call_chat, prompt)
        text = data.get("text", "{}")
        return _extract_json(text)
