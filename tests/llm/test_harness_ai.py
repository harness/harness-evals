"""Tests for the Harness AI Service LLM provider."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from harness_evals.llm.harness_ai import HarnessAILLM, _extract_json, _generate_jwt


@pytest.mark.unit
class TestGenerateJwt:
    def test_returns_string(self):
        token = _generate_jwt(b"test-secret")
        assert isinstance(token, str)
        assert len(token) > 0

    def test_decodable(self):
        import jwt

        token = _generate_jwt(b"my-secret")
        payload = jwt.decode(token, b"my-secret", algorithms=["HS256"])
        assert payload["iss"] == "Harness Inc"
        assert payload["sub"] == "harness-evals"
        assert payload["type"] == "SERVICE"

    def test_custom_service_name(self):
        import jwt

        token = _generate_jwt(b"my-secret", service_name="CustomApp")
        payload = jwt.decode(token, b"my-secret", algorithms=["HS256"])
        assert payload["sub"] == "CustomApp"
        assert payload["name"] == "CustomApp"


@pytest.mark.unit
class TestExtractJson:
    def test_plain_json(self):
        assert _extract_json('{"score": 8, "reasoning": "ok"}') == {"score": 8, "reasoning": "ok"}

    def test_json_with_whitespace(self):
        assert _extract_json('  \n {"score": 5} \n  ') == {"score": 5}

    def test_json_in_markdown_fence(self):
        text = '```json\n{"score": 7, "reasoning": "good"}\n```'
        assert _extract_json(text) == {"score": 7, "reasoning": "good"}

    def test_json_in_bare_fence(self):
        text = '```\n{"score": 3}\n```'
        assert _extract_json(text) == {"score": 3}

    def test_json_fence_with_surrounding_text(self):
        text = 'Here is my evaluation:\n```json\n{"score": 9, "reasoning": "excellent"}\n```\nEnd.'
        assert _extract_json(text) == {"score": 9, "reasoning": "excellent"}

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("this is not json at all")

    def test_empty_string_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("")


@pytest.mark.unit
class TestHarnessAILLMInit:
    def test_missing_secret_raises(self):
        with patch.dict("os.environ", {}, clear=True), pytest.raises(ValueError, match="No secret"):
            HarnessAILLM(secret="")

    def test_env_secret(self):
        with patch.dict("os.environ", {"HARNESS_AI_SERVICE_SECRET": "s3cret"}):
            llm = HarnessAILLM()
            assert llm.secret == b"s3cret"

    def test_explicit_params(self):
        llm = HarnessAILLM(
            base_url="http://example.com:8001",
            secret="my-secret",
            model="gpt-4.1",
            provider="openai",
        )
        assert llm.base_url == "http://example.com:8001"
        assert llm.model == "gpt-4.1"
        assert llm.provider == "openai"

    def test_service_name_default(self):
        llm = HarnessAILLM(secret="test")
        assert llm.service_name == "harness-evals"

    def test_service_name_custom(self):
        llm = HarnessAILLM(secret="test", service_name="my-app")
        assert llm.service_name == "my-app"


def _mock_httpx_response(payload: dict, status_code: int = 200) -> httpx.Response:
    """Build a mock httpx.Response with a canned JSON body."""
    content = json.dumps(payload).encode()
    return httpx.Response(status_code=status_code, content=content)


@pytest.mark.unit
class TestHarnessAILLMGenerate:
    def _make_llm(self) -> HarnessAILLM:
        return HarnessAILLM(base_url="http://test:8001", secret="test-secret")

    async def test_generate_success(self):
        llm = self._make_llm()
        resp = _mock_httpx_response({"text": "Hello world", "blocked": False})
        mock_client = AsyncMock()
        mock_client.post.return_value = resp

        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await llm.generate("Say hello")
        assert result == "Hello world"

    async def test_generate_json_success(self):
        llm = self._make_llm()
        resp = _mock_httpx_response({"text": '{"reasoning": "good", "score": 9}', "blocked": False})
        mock_client = AsyncMock()
        mock_client.post.return_value = resp

        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await llm.generate_json("Rate this", schema={})
        assert result == {"reasoning": "good", "score": 9}

    async def test_generate_json_appends_schema(self):
        llm = self._make_llm()
        resp = _mock_httpx_response({"text": '{"reasoning": "ok", "score": 5}', "blocked": False})
        mock_client = AsyncMock()
        mock_client.post.return_value = resp

        schema = {"type": "object", "properties": {"score": {"type": "integer"}}}
        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await llm.generate_json("Rate this", schema=schema)

        sent_body = mock_client.post.call_args[1]["json"]
        assert "schema" in sent_body["message"].lower()

    async def test_generate_json_empty_schema_no_append(self):
        llm = self._make_llm()
        resp = _mock_httpx_response({"text": '{"reasoning": "ok", "score": 5}', "blocked": False})
        mock_client = AsyncMock()
        mock_client.post.return_value = resp

        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            await llm.generate_json("Rate this", schema={})

        sent_body = mock_client.post.call_args[1]["json"]
        assert sent_body["message"] == "Rate this"

    async def test_generate_json_markdown_fence(self):
        llm = self._make_llm()
        resp = _mock_httpx_response({"text": '```json\n{"reasoning": "fenced", "score": 7}\n```', "blocked": False})
        mock_client = AsyncMock()
        mock_client.post.return_value = resp

        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await llm.generate_json("Rate this", schema={})
        assert result == {"reasoning": "fenced", "score": 7}

    async def test_generate_json_malformed_raises(self):
        llm = self._make_llm()
        resp = _mock_httpx_response({"text": "I cannot evaluate this.", "blocked": False})
        mock_client = AsyncMock()
        mock_client.post.return_value = resp

        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            with pytest.raises(json.JSONDecodeError):
                await llm.generate_json("Rate this", schema={})

    async def test_http_error_raises(self):
        llm = self._make_llm()
        resp = _mock_httpx_response({"error": "fail"}, status_code=500)
        mock_client = AsyncMock()
        mock_client.post.return_value = resp

        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            with pytest.raises(RuntimeError, match="500"):
                await llm.generate("test")

    async def test_blocked_response_raises(self):
        llm = self._make_llm()
        resp = _mock_httpx_response({"text": "", "blocked": True})
        mock_client = AsyncMock()
        mock_client.post.return_value = resp

        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            with pytest.raises(RuntimeError, match="blocked"):
                await llm.generate("test")
