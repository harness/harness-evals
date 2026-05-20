"""Tests for WebhookMetric."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score
from harness_evals.metrics.deterministic.webhook import WebhookMetric


def _mock_httpx_response(payload: dict, status_code: int = 200) -> httpx.Response:
    """Build a mock httpx.Response with a canned JSON body."""
    content = json.dumps(payload).encode()
    return httpx.Response(status_code=status_code, content=content)


@pytest.mark.unit
class TestWebhookMetricInit:
    def test_no_response_mode_raises(self):
        with pytest.raises(ValueError, match="Must specify exactly one response handling mode"):
            WebhookMetric(url="http://example.com/validate")

    def test_multiple_response_modes_raises(self):
        with pytest.raises(ValueError, match="Cannot specify multiple response handling modes"):
            WebhookMetric(
                url="http://example.com/validate",
                response_key="valid",
                use_status_code=True,
            )

        with pytest.raises(ValueError, match="Cannot specify multiple response handling modes"):
            WebhookMetric(
                url="http://example.com/validate",
                response_fn=lambda r: Score(name="test", value=1.0, threshold=1.0),
                use_status_code=True,
            )

    def test_score_key_without_response_key_raises(self):
        with pytest.raises(ValueError, match="score_key requires response_key"):
            WebhookMetric(
                url="http://example.com/validate",
                use_status_code=True,
                score_key="confidence",
            )

    def test_valid_response_key_mode(self):
        metric = WebhookMetric(url="http://example.com/validate", response_key="is_valid")
        assert metric.response_key == "is_valid"

    def test_valid_status_code_mode(self):
        metric = WebhookMetric(url="http://example.com/validate", use_status_code=True)
        assert metric.use_status_code is True

    def test_valid_response_fn_mode(self):
        def fn(r: httpx.Response) -> Score:
            return Score(name="test", value=1.0, threshold=1.0)

        metric = WebhookMetric(url="http://example.com/validate", response_fn=fn)
        assert metric.response_fn is fn


@pytest.mark.unit
class TestWebhookMetricResponseKey:
    @pytest.mark.asyncio
    async def test_response_key_pass(self):
        ec = EvalCase(input="test input", output="test output", expected="test expected")
        metric = WebhookMetric(url="http://example.com/validate", response_key="is_valid")

        resp = _mock_httpx_response({"is_valid": True})
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.request.return_value = resp
            mock_cls.return_value = mock_client

            score = await metric.a_measure(ec)
            assert score.passed
            assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_response_key_fail(self):
        ec = EvalCase(input="test input", output="test output", expected="test expected")
        metric = WebhookMetric(url="http://example.com/validate", response_key="is_valid")

        resp = _mock_httpx_response({"is_valid": False})
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.request.return_value = resp
            mock_cls.return_value = mock_client

            score = await metric.a_measure(ec)
            assert not score.passed
            assert score.value == 0.0

    @pytest.mark.asyncio
    async def test_response_key_missing(self):
        ec = EvalCase(input="test input", output="test output", expected="test expected")
        metric = WebhookMetric(url="http://example.com/validate", response_key="is_valid")

        resp = _mock_httpx_response({"other_key": True})
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.request.return_value = resp
            mock_cls.return_value = mock_client

            score = await metric.a_measure(ec)
            assert not score.passed
            assert "missing key" in score.reason.lower()

    @pytest.mark.asyncio
    async def test_response_key_not_boolean(self):
        ec = EvalCase(input="test input", output="test output", expected="test expected")
        metric = WebhookMetric(url="http://example.com/validate", response_key="is_valid")

        resp = _mock_httpx_response({"is_valid": "yes"})
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.request.return_value = resp
            mock_cls.return_value = mock_client

            score = await metric.a_measure(ec)
            assert not score.passed
            assert "not boolean" in score.reason.lower()


@pytest.mark.unit
class TestWebhookMetricScoreKey:
    @pytest.mark.asyncio
    async def test_score_key_with_response_key(self):
        ec = EvalCase(input="test input", output="test output", expected="test expected")
        metric = WebhookMetric(
            url="http://example.com/validate",
            response_key="is_valid",
            score_key="confidence",
            threshold=0.7,
        )

        resp = _mock_httpx_response({"is_valid": True, "confidence": 0.85})
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.request.return_value = resp
            mock_cls.return_value = mock_client

            score = await metric.a_measure(ec)
            assert score.value == 0.85
            assert score.passed

    @pytest.mark.asyncio
    async def test_score_key_clamped(self):
        ec = EvalCase(input="test input", output="test output", expected="test expected")
        metric = WebhookMetric(
            url="http://example.com/validate",
            response_key="is_valid",
            score_key="confidence",
        )

        resp = _mock_httpx_response({"is_valid": True, "confidence": 1.5})
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.request.return_value = resp
            mock_cls.return_value = mock_client

            score = await metric.a_measure(ec)
            assert score.value == 1.0

    @pytest.mark.asyncio
    async def test_score_key_invalid_type(self):
        ec = EvalCase(input="test input", output="test output", expected="test expected")
        metric = WebhookMetric(
            url="http://example.com/validate",
            response_key="is_valid",
            score_key="confidence",
        )

        resp = _mock_httpx_response({"is_valid": True, "confidence": "high"})
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.request.return_value = resp
            mock_cls.return_value = mock_client

            score = await metric.a_measure(ec)
            assert score.value == 0.0
            assert "Invalid score value" in score.reason


@pytest.mark.unit
class TestWebhookMetricStatusCode:
    @pytest.mark.asyncio
    async def test_status_code_success(self):
        ec = EvalCase(input="test input", output="test output", expected="test expected")
        metric = WebhookMetric(url="http://example.com/validate", use_status_code=True)

        resp = _mock_httpx_response({}, status_code=200)
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.request.return_value = resp
            mock_cls.return_value = mock_client

            score = await metric.a_measure(ec)
            assert score.passed
            assert score.value == 1.0
            assert score.reason is None

    @pytest.mark.asyncio
    async def test_status_code_failure(self):
        ec = EvalCase(input="test input", output="test output", expected="test expected")
        metric = WebhookMetric(url="http://example.com/validate", use_status_code=True)

        resp = _mock_httpx_response({}, status_code=400)
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.request.return_value = resp
            mock_cls.return_value = mock_client

            score = await metric.a_measure(ec)
            assert not score.passed
            assert score.value == 0.0
            assert "400" in score.reason


@pytest.mark.unit
class TestWebhookMetricResponseFn:
    @pytest.mark.asyncio
    async def test_response_fn_custom_logic(self):
        ec = EvalCase(input="test input", output="test output", expected="test expected")

        def custom_parser(response: httpx.Response) -> Score:
            data = response.json()
            value = float(data.get("custom_score", 0))
            return Score(
                name="webhook",
                value=value,
                threshold=1.0,
                reason=data.get("message"),
            )

        metric = WebhookMetric(url="http://example.com/validate", response_fn=custom_parser)

        resp = _mock_httpx_response({"custom_score": 0.75, "message": "Good job"})
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.request.return_value = resp
            mock_cls.return_value = mock_client

            score = await metric.a_measure(ec)
            assert score.value == 0.75
            assert score.reason == "Good job"


@pytest.mark.unit
class TestWebhookMetricPayload:
    @pytest.mark.asyncio
    async def test_default_payload(self):
        ec = EvalCase(
            input="test input",
            output="test output",
            expected="test expected",
            context=["ctx1", "ctx2"],
        )
        metric = WebhookMetric(url="http://example.com/validate", use_status_code=True)

        resp = _mock_httpx_response({}, status_code=200)
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.request.return_value = resp
            mock_cls.return_value = mock_client

            await metric.a_measure(ec)

            call_kwargs = mock_client.request.call_args[1]
            assert call_kwargs["json"]["input"] == "test input"
            assert call_kwargs["json"]["output"] == "test output"
            assert call_kwargs["json"]["expected"] == "test expected"
            assert call_kwargs["json"]["context"] == ["ctx1", "ctx2"]

    @pytest.mark.asyncio
    async def test_default_payload_without_optional_fields(self):
        ec = EvalCase(input="test input", output="test output")
        metric = WebhookMetric(url="http://example.com/validate", use_status_code=True)

        resp = _mock_httpx_response({}, status_code=200)
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.request.return_value = resp
            mock_cls.return_value = mock_client

            await metric.a_measure(ec)

            call_kwargs = mock_client.request.call_args[1]
            assert call_kwargs["json"]["input"] == "test input"
            assert call_kwargs["json"]["output"] == "test output"
            assert "expected" not in call_kwargs["json"]
            assert "context" not in call_kwargs["json"]

    @pytest.mark.asyncio
    async def test_custom_payload_fn(self):
        ec = EvalCase(input="test input", output="test output", expected="test expected")

        def custom_payload(eval_case: EvalCase) -> dict:
            return {
                "query": eval_case.input,
                "response": eval_case.output,
                "metadata": {"custom": "data"},
            }

        metric = WebhookMetric(
            url="http://example.com/validate",
            payload_fn=custom_payload,
            use_status_code=True,
        )

        resp = _mock_httpx_response({}, status_code=200)
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.request.return_value = resp
            mock_cls.return_value = mock_client

            await metric.a_measure(ec)

            call_kwargs = mock_client.request.call_args[1]
            assert call_kwargs["json"]["query"] == "test input"
            assert call_kwargs["json"]["response"] == "test output"
            assert call_kwargs["json"]["metadata"]["custom"] == "data"


@pytest.mark.unit
class TestWebhookMetricErrorHandling:
    @pytest.mark.asyncio
    async def test_http_error_not_status_code_mode(self):
        ec = EvalCase(input="test input", output="test output", expected="test expected")
        metric = WebhookMetric(url="http://example.com/validate", response_key="is_valid")

        resp = _mock_httpx_response({}, status_code=500)
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.request.return_value = resp
            mock_cls.return_value = mock_client

            score = await metric.a_measure(ec)
            assert not score.passed
            assert score.value == 0.0
            assert "500" in score.reason

    @pytest.mark.asyncio
    async def test_timeout(self):
        ec = EvalCase(input="test input", output="test output", expected="test expected")
        metric = WebhookMetric(url="http://example.com/validate", response_key="is_valid", timeout=1.0)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.request.side_effect = httpx.TimeoutException("Timeout")
            mock_cls.return_value = mock_client

            score = await metric.a_measure(ec)
            assert not score.passed
            assert score.value == 0.0
            assert "timed out" in score.reason.lower()

    @pytest.mark.asyncio
    async def test_connection_error(self):
        ec = EvalCase(input="test input", output="test output", expected="test expected")
        metric = WebhookMetric(url="http://example.com/validate", response_key="is_valid")

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.request.side_effect = httpx.ConnectError("Connection refused")
            mock_cls.return_value = mock_client

            score = await metric.a_measure(ec)
            assert not score.passed
            assert score.value == 0.0
            assert "failed" in score.reason.lower()

    @pytest.mark.asyncio
    async def test_invalid_json_response(self):
        ec = EvalCase(input="test input", output="test output", expected="test expected")
        metric = WebhookMetric(url="http://example.com/validate", response_key="is_valid")

        resp = httpx.Response(status_code=200, content=b"not json")
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.request.return_value = resp
            mock_cls.return_value = mock_client

            score = await metric.a_measure(ec)
            assert not score.passed
            assert score.value == 0.0
            assert "Invalid response format" in score.reason


@pytest.mark.unit
class TestWebhookMetricSyncMeasure:
    def test_sync_measure_calls_async(self):
        ec = EvalCase(input="test input", output="test output", expected="test expected")
        metric = WebhookMetric(url="http://example.com/validate", use_status_code=True)

        resp = _mock_httpx_response({}, status_code=200)
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.request.return_value = resp
            mock_cls.return_value = mock_client

            score = metric.measure(ec)
            assert score.passed
            assert score.value == 1.0


@pytest.mark.unit
class TestWebhookMetricCustomHeaders:
    @pytest.mark.asyncio
    async def test_custom_headers(self):
        ec = EvalCase(input="test input", output="test output", expected="test expected")
        metric = WebhookMetric(
            url="http://example.com/validate",
            headers={"Authorization": "Bearer token123", "X-Custom": "value"},
            use_status_code=True,
        )

        resp = _mock_httpx_response({}, status_code=200)
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.request.return_value = resp
            mock_cls.return_value = mock_client

            await metric.a_measure(ec)

            call_kwargs = mock_client.request.call_args[1]
            assert call_kwargs["headers"]["Authorization"] == "Bearer token123"
            assert call_kwargs["headers"]["X-Custom"] == "value"


@pytest.mark.unit
class TestWebhookMetricCustomMethod:
    @pytest.mark.asyncio
    async def test_custom_method(self):
        ec = EvalCase(input="test input", output="test output", expected="test expected")
        metric = WebhookMetric(url="http://example.com/validate", method="PUT", use_status_code=True)

        resp = _mock_httpx_response({}, status_code=200)
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.request.return_value = resp
            mock_cls.return_value = mock_client

            await metric.a_measure(ec)

            call_args = mock_client.request.call_args[1]
            assert call_args["method"] == "PUT"
