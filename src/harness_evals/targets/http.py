"""HttpTarget — POST to a deployed agent and extract structured output."""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
import time
from dataclasses import dataclass, field
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.golden import Golden
from harness_evals.errors import TargetInvocationError
from harness_evals.plugins import register_target
from harness_evals.targets.auth import AuthConfig, NoAuth
from harness_evals.targets.base import BaseTarget
from harness_evals.targets.http_diagnostics import (
    CAPTURE_LIMIT,
    HttpCall,
    HttpDiagnostic,
    HttpExchange,
    apply_missing_selector,
    apply_parse_failure,
    apply_status_failure,
    apply_transport_failure,
    attach_case_diagnostic,
    build_exchange,
    current_request_trace,
    header_items,
    output_match,
    publish_diagnostic,
    read_urllib_error_body,
    request_trace,
    status_code_of,
)
from harness_evals.targets.templating import (
    render_headers_with_provenance,
    render_request_body,
    render_request_body_with_provenance,
)
from harness_evals.targets.trajectory import normalize_trajectory_fields, synthesize_messages
from harness_evals.utils.path import extract_path

logger = logging.getLogger(__name__)


@register_target("http")
@dataclass
class HttpTarget(BaseTarget):
    """POST to a deployed agent endpoint and map the response to an EvalCase.

    The agent's internals are opaque — this grades the *shipped system*
    end-to-end. All JSONPath extraction uses the existing ``extract_path``
    utility (backed by ``jsonpath-ng``).
    """

    url: str
    method: str = "POST"
    auth: AuthConfig = field(default_factory=NoAuth)
    headers: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 60.0
    verify_tls: bool = True

    body_template: dict | None = None

    retries: int = 2
    backoff_s: float = 0.5

    output_path: str = "$.output"
    tool_calls_path: str | None = None
    context_path: str | None = None
    messages_path: str | None = None
    token_count_path: str | None = None
    input_tokens_path: str | None = None
    output_tokens_path: str | None = None
    cost_usd_path: str | None = None
    retry_count_path: str | None = None
    confidence_path: str | None = None
    latency_ms_path: str | None = None

    # strict_onboarding turns a selector that matches nothing into an error.
    # The default keeps the legacy empty-string output. record_diagnostics
    # retains a redacted HttpDiagnostic on the target, the exception, and
    # EvalCase.metadata["http_diagnostic"].
    strict_onboarding: bool = False
    record_diagnostics: bool = False

    _async_client: object | None = field(default=None, init=False, repr=False)
    last_diagnostic: HttpDiagnostic | None = field(default=None, init=False, repr=False)

    async def __aenter__(self) -> HttpTarget:
        if httpx is None:
            raise ImportError("httpx is required for async context manager usage: pip install harness-evals[harness]")
        verify: bool | ssl.SSLContext = self.verify_tls
        if not self.verify_tls:
            logger.warning("TLS verification disabled for %s — do not use in production", self.url)
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            verify = ctx
        self._async_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout_s),
            verify=verify,
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._async_client is not None:
            await self._async_client.aclose()  # type: ignore[union-attr]
            self._async_client = None

    async def ainvoke(self, golden: Golden) -> EvalCase:
        if not self.record_diagnostics:
            self.last_diagnostic = None
        body_payload, body_secrets = render_request_body_with_provenance(self.body_template, golden)
        body = json.dumps(body_payload, ensure_ascii=False).encode("utf-8")
        headers, header_uses, header_secrets = render_headers_with_provenance(self.headers, golden)
        secrets = [*body_secrets, *header_secrets]

        with request_trace(header_uses, secrets):
            if self._async_client is not None:
                call = await self._execute_async(body, headers)
            else:
                logger.debug("HttpTarget used without context manager — falling back to sync thread pool")
                call = await asyncio.to_thread(self._execute_with_retries, body, headers)

        if call.error is not None:
            raise self._failure(call, None)

        output, matched = output_match(call.body, self.output_path, call.content_type)
        if self.strict_onboarding and not matched:
            call.error = apply_missing_selector(call.exchange, self.output_path)
            raise self._failure(call, None)

        kwargs = self._extract_optional_fields(call.body)
        extracted_context = kwargs.pop("context", None)
        # A reported trajectory (messages_path) is authoritative; otherwise
        # assemble a best-effort trace from the input, any captured tool calls,
        # and the output so agent/trajectory metrics have something to grade.
        if kwargs.get("messages") is None:
            kwargs["messages"] = synthesize_messages(golden.input, output, kwargs.get("tool_calls"))
        if self.latency_ms_path is None:
            kwargs["latency_ms"] = call.latency_ms
        else:
            extracted_latency = extract_path(call.body, self.latency_ms_path)
            kwargs["latency_ms"] = float(extracted_latency) if extracted_latency is not None else call.latency_ms

        eval_case = EvalCase.from_golden(golden, output=output, **kwargs)
        if extracted_context is not None:
            eval_case.context = extracted_context
        attach_case_diagnostic(eval_case, publish_diagnostic(self, call.exchange, output))
        return eval_case

    def _failure(self, call: HttpCall, extracted: object) -> TargetInvocationError:
        diagnostic = publish_diagnostic(self, call.exchange, extracted)
        return TargetInvocationError(
            f"HttpTarget invocation failed: {call.error}",
            latency_ms=call.latency_ms,
            diagnostic=diagnostic,
        )

    def _open_exchange(
        self,
        body: bytes,
        user_headers: dict[str, str],
        header_uses: dict | None,
        env_secrets: list[str] | None,
        *,
        accept: str,
    ) -> tuple[object, dict[str, str]]:
        exchange = build_exchange(
            method=self.method,
            url=self.url,
            accept=accept,
            user_headers=user_headers,
            header_uses=header_uses,
            body_text=body.decode("utf-8"),
            auth=self.auth,
            env_secrets=env_secrets,
        )
        # Final header map, including auth. Values stay intact on the wire;
        # redaction happens only when a diagnostic record is built.
        request_headers = {header.name: header.value for header in exchange.headers}
        return exchange, request_headers

    async def _execute_async(self, body: bytes, user_headers: dict[str, str]) -> HttpCall:
        """Async HTTP call using httpx with connection pooling and retries."""
        client = self._async_client
        assert client is not None

        header_uses, env_secrets = current_request_trace()
        exchange, headers = self._open_exchange(body, user_headers, header_uses, env_secrets, accept="application/json")
        last_error: str | None = None
        last_latency_ms = 0.0
        attempts = 1 + self.retries

        for attempt in range(attempts):
            if attempt > 0:
                await asyncio.sleep(self.backoff_s * (2 ** (attempt - 1)))

            t0 = time.perf_counter()
            try:
                response = await client.request(  # type: ignore[union-attr]
                    exchange.method,
                    exchange.url,
                    content=body,
                    headers=headers,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000
                status = status_code_of(response)
                # Read the status before raise_for_status so a non-2xx body is kept
                # even when the client only exposes it on the response object.
                if status is not None and status >= 400:
                    text = _response_text(response)
                    message = apply_status_failure(exchange, status, text, header_items(response.headers))
                    last_latency_ms = elapsed_ms
                    last_error = message
                    if 400 <= status < 500:
                        logger.warning("HttpTarget got non-retryable %d: %s", status, message)
                        return HttpCall(None, "", elapsed_ms, message, exchange)
                    logger.warning("HttpTarget attempt %d/%d failed: %s", attempt + 1, attempts, message)
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                raw = _response_text(response)
                _note_success(exchange, response, raw)
                parsed, error = _parse_success(exchange, raw, content_type)
                if error is not None:
                    return HttpCall(None, content_type, elapsed_ms, error, exchange)
                return HttpCall(parsed, content_type, elapsed_ms, None, exchange)
            except httpx.HTTPStatusError as exc:  # type: ignore[union-attr]
                elapsed_ms = (time.perf_counter() - t0) * 1000
                status = exc.response.status_code
                text = _response_text(exc.response)
                message = apply_status_failure(exchange, status, text, header_items(exc.response.headers), exc)
                last_latency_ms = elapsed_ms
                last_error = message
                # 4xx responses are client errors — retrying can never succeed,
                # so fail fast instead of hammering the endpoint with backoff.
                if 400 <= status < 500:
                    logger.warning("HttpTarget got non-retryable %d: %s", status, message)
                    return HttpCall(None, "", elapsed_ms, message, exchange)
                logger.warning("HttpTarget attempt %d/%d failed: %s", attempt + 1, attempts, message)
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                message = apply_transport_failure(exchange, exc)
                last_latency_ms = elapsed_ms
                last_error = message
                logger.warning("HttpTarget attempt %d/%d failed: %s", attempt + 1, attempts, message)

        return HttpCall(None, "", last_latency_ms, last_error, exchange)

    def _build_request_body(self, golden: Golden) -> bytes:
        payload = render_request_body(self.body_template, golden)
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def _execute_with_retries(self, body: bytes, user_headers: dict[str, str]) -> HttpCall:
        """Synchronous HTTP call with retry logic."""
        header_uses, env_secrets = current_request_trace()
        exchange, headers = self._open_exchange(body, user_headers, header_uses, env_secrets, accept="application/json")

        ssl_ctx: ssl.SSLContext | None = None
        if not self.verify_tls:
            logger.warning("TLS verification disabled for %s — do not use in production", self.url)
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE

        last_error: str | None = None
        last_attempt_latency_ms = 0.0
        attempts = 1 + self.retries

        for attempt in range(attempts):
            if attempt > 0:
                time.sleep(self.backoff_s * (2 ** (attempt - 1)))

            request = Request(exchange.url, data=body, headers=headers, method=exchange.method)
            t0 = time.perf_counter()
            try:
                with urlopen(request, timeout=self.timeout_s, context=ssl_ctx) as response:
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    raw = response.read().decode("utf-8")
                    content_type = _get_content_type(response)
                    _note_success(exchange, response, raw)
                    parsed, error = _parse_success(exchange, raw, content_type)
                    if error is not None:
                        return HttpCall(None, content_type, elapsed_ms, error, exchange)
                    return HttpCall(parsed, content_type, elapsed_ms, None, exchange)
            except HTTPError as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                message = apply_status_failure(
                    exchange,
                    exc.code,
                    read_urllib_error_body(exc),
                    header_items(getattr(exc, "headers", None)),
                    exc,
                )
                last_attempt_latency_ms = elapsed_ms
                last_error = message
                # 4xx responses are client errors — retrying can never succeed.
                if 400 <= exc.code < 500:
                    logger.warning("HttpTarget got non-retryable %d: %s", exc.code, message)
                    return HttpCall(None, "", elapsed_ms, message, exchange)
                logger.warning("HttpTarget attempt %d/%d failed: %s", attempt + 1, attempts, message)
            except (URLError, TimeoutError, OSError) as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                message = apply_transport_failure(exchange, exc)
                last_attempt_latency_ms = elapsed_ms
                last_error = message
                logger.warning("HttpTarget attempt %d/%d failed: %s", attempt + 1, attempts, message)

        return HttpCall(None, "", last_attempt_latency_ms, last_error, exchange)

    def _extract_output(self, response_body: object, content_type: str) -> str | dict | list:
        if self.output_path == "$" and "text/" in content_type:
            return str(response_body) if not isinstance(response_body, str) else response_body

        if response_body is None:
            return ""

        result = extract_path(response_body, self.output_path)
        return result if result is not None else ""

    def _extract_optional_fields(self, response_body: object) -> dict:
        kwargs: dict = {}
        if response_body is None:
            return kwargs

        _extract_field(kwargs, response_body, "tool_calls", self.tool_calls_path)
        _extract_field(kwargs, response_body, "context", self.context_path)
        _extract_field(kwargs, response_body, "messages", self.messages_path)
        _extract_float(kwargs, response_body, "token_count", self.token_count_path, int)
        _extract_float(kwargs, response_body, "input_tokens", self.input_tokens_path, int)
        _extract_float(kwargs, response_body, "output_tokens", self.output_tokens_path, int)
        _extract_float(kwargs, response_body, "cost_usd", self.cost_usd_path, float)
        _extract_float(kwargs, response_body, "retry_count", self.retry_count_path, int)
        _extract_float(kwargs, response_body, "confidence", self.confidence_path, float)

        normalize_trajectory_fields(kwargs, self.messages_path)
        return kwargs


def _note_success(exchange: HttpExchange, response: object, raw: str) -> None:
    """A later success must not keep the previous attempt's failure category."""
    status = status_code_of(response)
    if status is not None:
        exchange.status = status
    exchange.response_headers = header_items(getattr(response, "headers", None))
    exchange.response_body_text = raw[:CAPTURE_LIMIT]
    exchange.category = None
    exchange.transport_cause = None
    exchange.error_message = None


def _response_text(response: object) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    return ""


def _parse_success(exchange: HttpExchange, raw: str, content_type: str) -> tuple[object, str | None]:
    """Parse a 2xx body. Invalid JSON with a JSON content type is a parse failure, not a retry."""
    exchange.response_body_text = raw[:CAPTURE_LIMIT]
    try:
        return _parse_response(raw, content_type), None
    except json.JSONDecodeError as exc:
        return None, apply_parse_failure(exchange, exc, raw)


def _get_content_type(response: object) -> str:
    headers = getattr(response, "headers", None)
    if headers is None:
        return ""
    return str(headers.get("Content-Type", "")).lower()


def _parse_response(raw: str, content_type: str) -> object:
    """Parse response as JSON if possible, otherwise return raw text."""
    if "json" in content_type:
        return json.loads(raw)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def _extract_field(kwargs: dict, body: object, key: str, path: str | None) -> None:
    if path is None:
        return
    val = extract_path(body, path)
    if val is not None:
        kwargs[key] = val


def _extract_float(kwargs: dict, body: object, key: str, path: str | None, cast: type) -> None:
    if path is None:
        return
    val = extract_path(body, path)
    if val is None:
        return
    # The path *was* configured and resolved to a value, so a failed cast means
    # the value is the wrong type (non-numeric string, dict) or out of range
    # (json.loads accepts Infinity/NaN → int(inf) raises OverflowError). Warn so
    # a misconfigured path is distinguishable from "the endpoint reported nothing".
    try:
        kwargs[key] = cast(val)
    except (TypeError, ValueError, OverflowError):
        logger.warning(
            "%s path %r resolved to %r, which is not a valid %s; dropping it",
            key,
            path,
            val,
            cast.__name__,
        )
