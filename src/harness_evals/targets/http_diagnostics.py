"""Redacted HTTP exchange diagnostics for HttpTarget and StreamingHttpTarget.

Failure categories are closed sets. A status that is not listed is reported
with its numeric code and a bounded body, and is not given a likely cause.
Likely-cause text is always labeled advisory.

The public record is opt-in (``record_diagnostics=True``). Credential headers
and environment-sourced values are redacted in that record and in the error
text that retains a response body. Environment values shorter than
``MIN_SUBSTRING_SECRET_LENGTH`` are redacted only when an entire field equals
them, so a short value cannot wipe unrelated text. Credential headers are
redacted in full regardless of length.
"""

from __future__ import annotations

import errno
import json
import socket
import ssl
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from harness_evals.env import resolve_env_value
from harness_evals.targets.auth import ApiKeyAuth, AuthConfig, BasicAuth, BearerAuth
from harness_evals.targets.templating import PlaceholderUse
from harness_evals.utils.path import extract_path

# Retained for display after redaction. Capture is larger so a secret that
# straddles the display cut is still replaced before the tail is dropped.
DISPLAY_LIMIT = 2048
CAPTURE_LIMIT = 16384
MIN_SUBSTRING_SECRET_LENGTH = 4
REDACTED = "[REDACTED]"
HTTP_DIAGNOSTIC_METADATA_KEY = "http_diagnostic"

# Task-local so concurrent invokes do not mix header provenance, and so the
# execute methods keep the (body, headers) shape existing tests patch.
_REQUEST_TRACE: ContextVar[tuple[dict[str, tuple[PlaceholderUse, ...]] | None, list[str] | None]] = ContextVar(
    "harness_evals_http_request_trace",
    default=(None, None),
)

# Closed sets. Anything else (404, 408, 429, 402, …) stays unclassified.
_AUTHENTICATION_STATUSES = frozenset({401, 403, 407})
_REQUEST_STATUSES = frozenset({400, 405, 409, 415, 422})

_CREDENTIAL_HEADER_NAMES = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "proxy-authenticate",
        "cookie",
        "set-cookie",
        "api-key",
        "x-api-key",
        "apikey",
        "x-apikey",
    }
)

_TIMEOUT_TYPE_NAMES = frozenset(
    {"TimeoutError", "timeout", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout", "TimeoutException"}
)


class HttpFailureCategory(str, Enum):
    """Closed failure classes. Absent means the failure was not classified."""

    AUTHENTICATION = "authentication"
    REQUEST = "request"
    TRANSPORT = "transport"
    SERVER = "server"
    RESPONSE_PARSE = "response_parse"
    MISSING_RESPONSE_PATH = "missing_response_path"


class TransportCause(str, Enum):
    """Transport causes identified from the exception, not from an HTTP status."""

    DNS = "dns"
    CONNECTION_REFUSED = "connection_refused"
    TLS = "tls"
    TIMEOUT = "timeout"


_CATEGORY_ADVICE = {
    HttpFailureCategory.AUTHENTICATION: (
        "the status is in the authentication set (401, 403, 407); credentials may be missing or rejected"
    ),
    HttpFailureCategory.REQUEST: (
        "the status is in the request/body set (400, 405, 409, 415, 422); "
        "the method, content type, or body may have been rejected"
    ),
    HttpFailureCategory.SERVER: "the status is a 5xx response produced by the remote server",
    HttpFailureCategory.RESPONSE_PARSE: (
        "the response status was successful but the body was not the expected representation"
    ),
    HttpFailureCategory.MISSING_RESPONSE_PATH: "the response selector matched no value",
}

_TRANSPORT_ADVICE = {
    TransportCause.DNS: "the host name could not be resolved",
    TransportCause.CONNECTION_REFUSED: "the connection was refused",
    TransportCause.TLS: "the TLS handshake or certificate check failed",
    TransportCause.TIMEOUT: "the request timed out",
}


@dataclass(frozen=True)
class HeaderProvenance:
    """Where a final request header value came from.

    ``kind`` is one of ``default`` (target-supplied Content-Type/Accept),
    ``literal`` (no placeholder), ``golden`` (``{{input}}`` / ``{{metadata}}``),
    ``env`` (``{{env.VAR}}``), or ``auth`` (AuthConfig.apply). A mixed header
    has one entry per placeholder.
    """

    kind: str
    expression: str | None = None
    env_var: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {"kind": self.kind, "expression": self.expression, "env_var": self.env_var}


@dataclass(frozen=True)
class HeaderDiagnostic:
    """One request header as retained on a redacted diagnostic record."""

    name: str
    value: str
    provenance: tuple[HeaderProvenance, ...]
    redacted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "redacted": self.redacted,
            "provenance": [item.to_dict() for item in self.provenance],
        }


@dataclass(frozen=True)
class HttpDiagnostic:
    """Opt-in redacted view of one HTTP attempt.

    ``body`` is the rendered, serialized request body. ``extracted_value`` is a
    redacted string form of the selector result, or ``None`` when extraction
    did not run. ``category`` is ``None`` when the failure was not in a closed
    set (for example HTTP 404) and on success.
    """

    method: str
    url: str
    body: str | None
    headers: tuple[HeaderDiagnostic, ...]
    response_status: int | None
    response_headers: tuple[tuple[str, str], ...]
    response_body: str | None
    extracted_value: str | None
    category: str | None = None
    transport_cause: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "url": self.url,
            "body": self.body,
            "headers": [header.to_dict() for header in self.headers],
            "response_status": self.response_status,
            "response_headers": [{"name": name, "value": value} for name, value in self.response_headers],
            "response_body": self.response_body,
            "extracted_value": self.extracted_value,
            "category": self.category,
            "transport_cause": self.transport_cause,
        }


@dataclass
class HttpExchange:
    """Mutable facts for the attempt currently in flight. Not part of the public API."""

    method: str
    url: str
    body_text: str
    headers: list[HeaderDiagnostic]
    secrets: list[str]
    status: int | None = None
    response_headers: list[tuple[str, str]] = field(default_factory=list)
    response_body_text: str | None = None
    category: HttpFailureCategory | None = None
    transport_cause: TransportCause | None = None
    error_message: str | None = None


@dataclass
class HttpCall:
    """Private result of one execute call, including the exchange used to explain it."""

    body: object
    content_type: str
    latency_ms: float
    error: str | None
    exchange: HttpExchange


@contextmanager
def request_trace(
    header_uses: dict[str, tuple[PlaceholderUse, ...]] | None,
    env_secrets: list[str] | None,
):
    """Expose template provenance to the HTTP execute methods for this task."""
    token = _REQUEST_TRACE.set((header_uses, env_secrets))
    try:
        yield
    finally:
        _REQUEST_TRACE.reset(token)


def current_request_trace() -> tuple[dict[str, tuple[PlaceholderUse, ...]] | None, list[str] | None]:
    return _REQUEST_TRACE.get()


def coerce_http_call(result: HttpCall | tuple) -> HttpCall:
    """Accept either the real ``HttpCall`` or the legacy four-tuple test doubles return."""
    if isinstance(result, HttpCall):
        return result
    raw, content_type, latency, error = result
    return HttpCall(
        body=raw,
        content_type=content_type or "",
        latency_ms=float(latency or 0.0),
        error=error,
        exchange=HttpExchange(method="", url="", body_text="", headers=[], secrets=[]),
    )


def classify_status(status: int) -> HttpFailureCategory | None:
    """Map a status onto a closed category, or ``None`` when it is not in one."""
    if status in _AUTHENTICATION_STATUSES:
        return HttpFailureCategory.AUTHENTICATION
    if status in _REQUEST_STATUSES:
        return HttpFailureCategory.REQUEST
    if 500 <= status <= 599:
        return HttpFailureCategory.SERVER
    return None


def classify_transport(exc: BaseException) -> TransportCause | None:
    """Identify DNS, connection refusal, TLS, or timeout without guessing other I/O errors."""
    seen: set[int] = set()
    pending: list[BaseException] = [exc]
    texts: list[str] = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        typed = _classify_exception_type(current)
        if typed is not None:
            return typed
        texts.append(str(current))
        cause = current.__cause__
        if isinstance(cause, BaseException):
            pending.append(cause)
        reason = getattr(current, "reason", None)
        if isinstance(reason, BaseException):
            pending.append(reason)
        elif isinstance(reason, str):
            texts.append(reason)
    for text in texts:
        classified = _classify_transport_text(text)
        if classified is not None:
            return classified
    return None


def is_credential_header(name: str) -> bool:
    """True for Authorization, proxy auth, Cookie/Set-Cookie, and API-key header names."""
    compact = name.lower().replace("_", "-")
    if compact in _CREDENTIAL_HEADER_NAMES:
        return True
    return compact.endswith("-api-key") or compact.endswith("-apikey")


def describe_failure(
    *,
    status: int | None = None,
    category: HttpFailureCategory | None = None,
    transport_cause: TransportCause | None = None,
    response_body: str | None = None,
    secrets: list[str] | None = None,
    exc: BaseException | None = None,
    selector: str | None = None,
) -> str:
    """Facts first (status, body, exception). Likely cause is included only when classified."""
    parts: list[str] = []
    if status is not None:
        parts.append(f"HTTP {status}")
    if category is not None:
        parts.append(f"category={category.value}")
    if transport_cause is not None:
        parts.append(f"transport={transport_cause.value}")
    advice = _TRANSPORT_ADVICE.get(transport_cause) if transport_cause is not None else None
    if advice is None and category is not None:
        advice = _CATEGORY_ADVICE.get(category)
    if advice is not None:
        parts.append(f"advisory likely cause: {advice}")
    if selector is not None:
        parts.append(f"selector: {selector}")
    if status is not None or response_body is not None:
        shown = "" if response_body is None else response_body
        parts.append(f"response body: {bound_text(redact_text(shown, secrets or []))}")
    if exc is not None:
        # Exception text can echo the URL, including a query-string credential.
        parts.append(redact_text(f"{type(exc).__name__}: {exc}", secrets or []))
    return "; ".join(parts)


def apply_status_failure(
    exchange: HttpExchange,
    status: int,
    response_body: str,
    response_headers: list[tuple[str, str]],
    exc: BaseException | None = None,
) -> str:
    """Record a non-2xx response and return the redacted error text."""
    exchange.status = status
    exchange.response_body_text = _capture(response_body)
    exchange.response_headers = response_headers
    exchange.category = classify_status(status)
    exchange.transport_cause = None
    message = describe_failure(
        status=status,
        category=exchange.category,
        response_body=exchange.response_body_text,
        secrets=exchange.secrets,
        exc=exc,
    )
    exchange.error_message = message
    return message


def apply_transport_failure(exchange: HttpExchange, exc: BaseException) -> str:
    """Record a transport exception. Unrecognized I/O errors stay unclassified."""
    cause = classify_transport(exc)
    exchange.transport_cause = cause
    exchange.category = HttpFailureCategory.TRANSPORT if cause is not None else None
    message = describe_failure(
        category=exchange.category,
        transport_cause=cause,
        secrets=exchange.secrets,
        exc=exc,
    )
    exchange.error_message = message
    return message


def apply_parse_failure(exchange: HttpExchange, exc: BaseException, response_body: str) -> str:
    """Record a 2xx body that was not the expected representation."""
    exchange.response_body_text = _capture(response_body)
    exchange.category = HttpFailureCategory.RESPONSE_PARSE
    message = describe_failure(
        status=exchange.status,
        category=exchange.category,
        response_body=exchange.response_body_text,
        secrets=exchange.secrets,
        exc=exc,
    )
    exchange.error_message = message
    return message


def apply_missing_selector(exchange: HttpExchange, selector: str) -> str:
    """Record a selector that resolved to nothing. Used only in strict onboarding mode."""
    exchange.category = HttpFailureCategory.MISSING_RESPONSE_PATH
    message = describe_failure(
        status=exchange.status,
        category=exchange.category,
        response_body=exchange.response_body_text,
        secrets=exchange.secrets,
        selector=selector,
    )
    exchange.error_message = message
    return message


def build_exchange(
    *,
    method: str,
    url: str,
    accept: str,
    user_headers: dict[str, str],
    header_uses: dict[str, tuple[PlaceholderUse, ...]] | None,
    body_text: str,
    auth: AuthConfig,
    env_secrets: list[str] | None,
) -> HttpExchange:
    """Assemble the final URL and headers, then remember which values are secrets."""
    headers = {"Content-Type": "application/json", "Accept": accept}
    provenance: dict[str, tuple[HeaderProvenance, ...]] = {
        "Content-Type": (HeaderProvenance(kind="default"),),
        "Accept": (HeaderProvenance(kind="default"),),
    }
    uses = header_uses or {}
    for name, value in user_headers.items():
        headers[name] = value
        placeholder_uses = uses.get(name)
        if placeholder_uses:
            provenance[name] = tuple(
                HeaderProvenance(kind=use.kind, expression=use.expression, env_var=use.env_var)
                for use in placeholder_uses
            )
        else:
            provenance[name] = (HeaderProvenance(kind="literal"),)
    before = dict(headers)
    params: dict[str, str] = {}
    auth.apply(headers, params)
    auth_expression = _auth_expression(auth)
    for name, value in headers.items():
        if before.get(name) != value:
            provenance[name] = (HeaderProvenance(kind="auth", expression=auth_expression),)
    header_rows = [
        HeaderDiagnostic(
            name=name,
            value=value,
            provenance=provenance.get(name, (HeaderProvenance(kind="literal"),)),
            redacted=False,
        )
        for name, value in headers.items()
    ]
    secrets = _dedupe([*(env_secrets or []), *_auth_secret_values(auth)])
    return HttpExchange(
        method=method,
        url=_append_query(url, params),
        body_text=body_text,
        headers=header_rows,
        secrets=secrets,
    )


def build_http_diagnostic(exchange: HttpExchange, extracted: object) -> HttpDiagnostic:
    """Copy the exchange into a redacted public record."""
    secrets = exchange.secrets
    headers = tuple(_redact_request_header(header, secrets) for header in exchange.headers)
    response_headers = tuple(
        (name, _redact_header_value(name, value, secrets)[0]) for name, value in exchange.response_headers
    )
    return HttpDiagnostic(
        method=exchange.method,
        url=redact_url(exchange.url, secrets),
        body=redact_serialized_body(exchange.body_text, secrets),
        headers=headers,
        response_status=exchange.status,
        response_headers=response_headers,
        response_body=(
            None
            if exchange.response_body_text is None
            else bound_text(redact_text(exchange.response_body_text, secrets))
        ),
        extracted_value=format_extracted(extracted, secrets),
        category=exchange.category.value if exchange.category is not None else None,
        transport_cause=exchange.transport_cause.value if exchange.transport_cause is not None else None,
    )


def publish_diagnostic(target: Any, exchange: HttpExchange, extracted: object) -> HttpDiagnostic | None:
    """Store the redacted record on the target when the caller opted in.

    ``last_diagnostic`` is last-write-wins if the same instance is invoked
    concurrently. The per-call copy returned here is what callers attach to
    the exception or the eval case.
    """
    if not getattr(target, "record_diagnostics", False):
        target.last_diagnostic = None
        return None
    diagnostic = build_http_diagnostic(exchange, extracted)
    target.last_diagnostic = diagnostic
    return diagnostic


def attach_case_diagnostic(eval_case: Any, diagnostic: HttpDiagnostic | None) -> None:
    if diagnostic is None:
        return
    if eval_case.metadata is None:
        eval_case.metadata = {}
    eval_case.metadata[HTTP_DIAGNOSTIC_METADATA_KEY] = diagnostic.to_dict()


def output_match(body: object, output_path: str, content_type: str) -> tuple[object, bool]:
    """Return ``(output, matched)`` using the legacy empty-string fallback.

    ``matched`` is false only when a selector was applied and resolved to
    nothing. Callers still return the empty string unless strict onboarding
    is on. JSON null is indistinguishable from a miss because ``extract_path``
    returns ``None`` for both.
    """
    if output_path == "$" and "text/" in content_type:
        text = body if isinstance(body, str) else str(body)
        return text, True
    if body is None:
        return "", False
    result = extract_path(body, output_path)
    if result is None:
        return "", False
    return result, True


def header_items(headers: object) -> list[tuple[str, str]]:
    items = getattr(headers, "items", None)
    if not callable(items):
        return []
    return [(str(key), str(value)) for key, value in items()]


def status_code_of(response: object) -> int | None:
    for attr in ("status_code", "status"):
        value = getattr(response, attr, None)
        if isinstance(value, int):
            return value
    return None


def read_urllib_error_body(exc: object) -> str:
    read = getattr(exc, "read", None)
    if not callable(read):
        return ""
    try:
        raw = read()
    except Exception:
        return ""
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def bound_text(text: str, limit: int = DISPLAY_LIMIT) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}...[truncated {omitted} chars]"


def redact_text(text: str, secrets: list[str]) -> str:
    """Replace known secret strings. Short secrets match only the entire text."""
    if not text or not secrets:
        return text
    redacted = text
    for secret in sorted({item for item in secrets if item}, key=len, reverse=True):
        if redacted == secret:
            return REDACTED
        if len(secret) >= MIN_SUBSTRING_SECRET_LENGTH and secret in redacted:
            redacted = redacted.replace(secret, REDACTED)
    return redacted


def redact_serialized_body(body_text: str, secrets: list[str]) -> str:
    captured = _capture(body_text)
    try:
        parsed = json.loads(captured)
    except (json.JSONDecodeError, ValueError):
        return bound_text(redact_text(captured, secrets))
    rendered = json.dumps(_redact_json(parsed, secrets), ensure_ascii=False)
    return bound_text(redact_text(rendered, secrets))


def redact_url(url: str, secrets: list[str]) -> str:
    """Drop userinfo, redact credential query values, then scrub secret substrings."""
    split = urlsplit(url)
    netloc = split.netloc
    if split.username is not None or split.password is not None:
        host = split.hostname or ""
        netloc = f"{host}:{split.port}" if split.port is not None else host
    query = []
    for key, value in parse_qsl(split.query, keep_blank_values=True):
        if is_credential_header(key):
            query.append((key, REDACTED))
        else:
            query.append((key, redact_text(value, secrets)))
    rebuilt = urlunsplit(
        (split.scheme, netloc, split.path, urlencode(query, quote_via=_quote_diagnostic), split.fragment)
    )
    return redact_text(rebuilt, secrets)


def format_extracted(value: object, secrets: list[str]) -> str | None:
    if value is None:
        return None
    redacted = _redact_json(value, secrets)
    text = redacted if isinstance(redacted, str) else json.dumps(redacted, ensure_ascii=False)
    return bound_text(redact_text(text, secrets))


def _capture(text: str) -> str:
    if len(text) <= CAPTURE_LIMIT:
        return text
    return text[:CAPTURE_LIMIT]


def _redact_json(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, str):
        return redact_text(value, secrets)
    if isinstance(value, dict):
        return {str(key): _redact_json(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_json(item, secrets) for item in value]
    return value


def _redact_request_header(header: HeaderDiagnostic, secrets: list[str]) -> HeaderDiagnostic:
    value, redacted = _redact_header_value(header.name, header.value, secrets)
    env_sourced = any(item.kind == "env" for item in header.provenance)
    if env_sourced and value != header.value:
        redacted = True
    return HeaderDiagnostic(name=header.name, value=value, provenance=header.provenance, redacted=redacted)


def _redact_header_value(name: str, value: str, secrets: list[str]) -> tuple[str, bool]:
    if is_credential_header(name):
        return REDACTED, True
    redacted_value = redact_text(value, secrets)
    return redacted_value, redacted_value != value


def _quote_diagnostic(text: str, safe: str, encoding: str | None, errors: str | None) -> str:
    # Keep the redaction marker readable in diagnostic URLs.
    return quote(text, safe=f"{safe}[]", encoding=encoding or "utf-8", errors=errors or "strict")


def _append_query(url: str, params: dict[str, str]) -> str:
    if not params:
        return url
    separator = "&" if "?" in url else "?"
    return url + separator + urlencode(params)


def _auth_expression(auth: AuthConfig) -> str:
    if isinstance(auth, BearerAuth):
        return "bearer"
    if isinstance(auth, ApiKeyAuth):
        return "api_key"
    if isinstance(auth, BasicAuth):
        return "basic"
    return type(auth).__name__


def _auth_secret_values(auth: AuthConfig) -> list[str]:
    """Credential material AuthConfig will place on the request, including ``${VAR}`` values."""
    try:
        if isinstance(auth, BearerAuth):
            values = [resolve_env_value(auth.token)]
        elif isinstance(auth, ApiKeyAuth):
            values = [resolve_env_value(auth.key)]
        elif isinstance(auth, BasicAuth):
            values = [resolve_env_value(auth.password)]
        else:
            return []
    except ValueError:
        return []
    return [value for value in values if value]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _classify_exception_type(exc: BaseException) -> TransportCause | None:
    if isinstance(exc, TimeoutError) or type(exc).__name__ in _TIMEOUT_TYPE_NAMES:
        return TransportCause.TIMEOUT
    if isinstance(exc, ssl.SSLError):
        return TransportCause.TLS
    if isinstance(exc, ConnectionRefusedError):
        return TransportCause.CONNECTION_REFUSED
    if isinstance(exc, OSError) and exc.errno == errno.ECONNREFUSED:
        return TransportCause.CONNECTION_REFUSED
    if isinstance(exc, OSError) and exc.errno == errno.ETIMEDOUT:
        return TransportCause.TIMEOUT
    # socket.gaierror is an OSError subclass. Check it after the errno cases above
    # so a refused connection is not relabeled as DNS.
    if isinstance(exc, socket.gaierror):
        return TransportCause.DNS
    return None


def _classify_transport_text(text: str) -> TransportCause | None:
    lowered = text.lower()
    if "timed out" in lowered or "timeout" in lowered:
        return TransportCause.TIMEOUT
    if "certificate" in lowered or "tls" in lowered or "ssl" in lowered:
        return TransportCause.TLS
    if "connection refused" in lowered:
        return TransportCause.CONNECTION_REFUSED
    if (
        "name or service not known" in lowered
        or "nodename nor servname" in lowered
        or "getaddrinfo" in lowered
        or "temporary failure in name resolution" in lowered
        or "name resolution" in lowered
    ):
        return TransportCause.DNS
    return None
