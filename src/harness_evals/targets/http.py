"""HttpTarget — POST to a deployed agent and extract structured output."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import logging
import ssl
import time
from dataclasses import dataclass, field
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.golden import Golden
from harness_evals.plugins import register_target
from harness_evals.targets.auth import AuthConfig, NoAuth
from harness_evals.targets.base import BaseTarget
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

    input_path: str = "$.input"
    body_template: dict | None = None

    retries: int = 2
    backoff_s: float = 0.5

    output_path: str = "$.output"
    tool_calls_path: str | None = None
    context_path: str | None = None
    messages_path: str | None = None
    token_count_path: str | None = None
    cost_usd_path: str | None = None
    retry_count_path: str | None = None
    confidence_path: str | None = None
    latency_ms_path: str | None = None

    def __post_init__(self) -> None:
        _validate_input_path(self.input_path)

    async def ainvoke(self, golden: Golden) -> EvalCase:
        body = self._build_request_body(golden)
        response_body, content_type, latency_ms, error = await asyncio.to_thread(self._execute_with_retries, body)

        if error is not None:
            return EvalCase.from_golden(
                golden,
                output="",
                latency_ms=latency_ms,
                metadata_extra={"http_error": error},
            )

        output = self._extract_output(response_body, content_type)
        kwargs = self._extract_optional_fields(response_body)
        extracted_context = kwargs.pop("context", None)
        if self.latency_ms_path is None:
            kwargs["latency_ms"] = latency_ms
        else:
            extracted_latency = extract_path(response_body, self.latency_ms_path)
            kwargs["latency_ms"] = float(extracted_latency) if extracted_latency is not None else latency_ms

        eval_case = EvalCase.from_golden(golden, output=output, **kwargs)
        if extracted_context is not None:
            eval_case.context = extracted_context
        return eval_case

    def _build_request_body(self, golden: Golden) -> bytes:
        if self.body_template is not None:
            payload = copy.deepcopy(self.body_template)
            _set_by_path(payload, self.input_path, golden.input)
        else:
            payload = {"input": golden.input}
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def _execute_with_retries(self, body: bytes) -> tuple[object, str, float, str | None]:
        """Synchronous HTTP call with retry logic. Returns (parsed_body, content_type, latency_ms, error)."""
        headers = {"Content-Type": "application/json", "Accept": "application/json", **self.headers}
        params: dict[str, str] = {}
        self.auth.apply(headers, params)

        url = self.url
        if params:
            sep = "&" if "?" in url else "?"
            url = url + sep + urlencode(params)

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

            request = Request(url, data=body, headers=headers, method=self.method)
            t0 = time.perf_counter()
            try:
                with urlopen(request, timeout=self.timeout_s, context=ssl_ctx) as response:
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    raw = response.read().decode("utf-8")
                    content_type = _get_content_type(response)
                    parsed = _parse_response(raw, content_type)
                    return parsed, content_type, elapsed_ms, None
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                last_attempt_latency_ms = elapsed_ms
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("HttpTarget attempt %d/%d failed: %s", attempt + 1, attempts, last_error)

        return None, "", last_attempt_latency_ms, last_error

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
        _extract_float(kwargs, response_body, "cost_usd", self.cost_usd_path, float)
        _extract_float(kwargs, response_body, "retry_count", self.retry_count_path, int)
        _extract_float(kwargs, response_body, "confidence", self.confidence_path, float)

        return kwargs


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


def _validate_input_path(path: str) -> None:
    if "[" in path or "]" in path:
        raise ValueError("input_path supports dot notation only; array indices and bracket syntax are not supported")


def _set_by_path(obj: dict, path: str, value: object) -> None:
    """Set a value in a nested dict using dot-separated path notation.

    Supports paths like ``$.input``, ``$.data.query``, or bare ``input``.
    Does not handle array indices or JSONPath wildcards/filters.
    """
    if path.startswith("$."):
        path = path[2:]
    elif path.startswith("$"):
        path = path[1:]

    parts = path.split(".")
    current = obj
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


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
    if val is not None:
        with contextlib.suppress(TypeError, ValueError):
            kwargs[key] = cast(val)
