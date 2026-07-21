"""Unit tests for HarnessOTELEvalCaseSource."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness_evals.importers.harness_otel import (
    HarnessOTELEvalCaseSource,
    _parse_result_rows,
    _parse_trace_id,
)
from harness_evals.refs import ResourceRef, resolve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(**kwargs) -> HarnessOTELEvalCaseSource:
    defaults = dict(
        api_key="pat.acct.xxx.yyy",
        account_id="acct123",
        org_id="MyOrg",
        project_id="MyProject",
    )
    defaults.update(kwargs)
    return HarnessOTELEvalCaseSource(**defaults)


def _qs_response(columns: list[str], rows: list[list]) -> dict:
    """Build a minimal QueryService JSON response."""
    return {
        "result": {
            "columns": [{"name": c} for c in columns],
            "rows": [{"values": r} for r in rows],
        }
    }


# ---------------------------------------------------------------------------
# _parse_trace_id
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_parse_trace_id_bare():
    assert _parse_trace_id("abc123") == "abc123"


@pytest.mark.unit
def test_parse_trace_id_prefixed():
    assert _parse_trace_id("trace/abc123") == "abc123"


# ---------------------------------------------------------------------------
# _parse_result_rows
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_parse_result_rows_empty():
    assert _parse_result_rows({}) == []


@pytest.mark.unit
def test_parse_result_rows_no_rows():
    resp = {"result": {"columns": [{"name": "trace_id"}], "rows": []}}
    assert _parse_result_rows(resp) == []


@pytest.mark.unit
def test_parse_result_rows_basic():
    resp = _qs_response(["trace_id", "duration_ms"], [["t1", 500], ["t2", 300]])
    rows = _parse_result_rows(resp)
    assert len(rows) == 2
    assert rows[0] == {"trace_id": "t1", "duration_ms": 500}
    assert rows[1] == {"trace_id": "t2", "duration_ms": 300}


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_constructor_missing_api_key(monkeypatch):
    monkeypatch.delenv("HARNESS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="api_key"):
        HarnessOTELEvalCaseSource(account_id="x")


@pytest.mark.unit
def test_constructor_missing_account_id(monkeypatch):
    monkeypatch.delenv("HARNESS_ACCOUNT_ID", raising=False)
    with pytest.raises(ValueError, match="account_id"):
        HarnessOTELEvalCaseSource(api_key="pat.x.y.z")


@pytest.mark.unit
def test_constructor_strips_trailing_slash():
    src = _make_source(base_url="https://app.harness.io/")
    assert not src._base_url.endswith("/")


# ---------------------------------------------------------------------------
# _build_url / _build_headers
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_build_url_contains_account_and_identifiers():
    src = _make_source()
    url = src._build_url()
    assert "query-service/grpc/" in url
    assert "executeQuery" in url
    assert "accountIdentifier=acct123" in url
    assert "orgIdentifier=MyOrg" in url
    assert "projectIdentifier=MyProject" in url


@pytest.mark.unit
def test_build_headers():
    src = _make_source()
    headers = src._build_headers()
    assert headers["x-api-key"] == "pat.acct.xxx.yyy"
    assert headers["harness-account"] == "acct123"
    assert headers["x-tenant-id"] == "default"
    assert headers["content-type"] == "application/json"


# ---------------------------------------------------------------------------
# _execute_hql — HTTP layer mocked
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_hql_success():
    src = _make_source()
    expected = {"result": {"columns": [], "rows": []}}

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = expected

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await src._execute_hql("find event x")

    assert result == expected


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_hql_401_raises_permission_error():
    src = _make_source()

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized"

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(PermissionError, match="401"):
            await src._execute_hql("find event x")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_hql_500_raises_runtime_error():
    src = _make_source()

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(RuntimeError, match="500"):
            await src._execute_hql("find event x")


# ---------------------------------------------------------------------------
# fetch — end-to-end with mock HTTP
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_fetch_single_trace_builds_eval_case():
    src = _make_source()

    spans_response = _qs_response(
        ["trace_id", "span_id", "parent_span_id", "name",
         "start_timestamp", "duration_ms", "attributes"],
        [
            ["trace-1", "span-root", None, "invoke_agent support-agent",
             "2026-01-01T00:00:00", 2000, '{"gen_ai.operation.name": "invoke_agent"}'],
            ["trace-1", "span-llm", "span-root", "chat gpt-4o",
             "2026-01-01T00:00:01", 1000,
             '{"gen_ai.operation.name": "chat",'
             ' "gen_ai.input.messages": "[{\\"role\\": \\"user\\", \\"parts\\": [{\\"type\\": \\"text\\", \\"text\\": \\"Hello\\"}]}]",'
             ' "gen_ai.output.messages": "[{\\"role\\": \\"assistant\\", \\"parts\\": [{\\"type\\": \\"text\\", \\"text\\": \\"Hi there\\"}]}]",'
             ' "gen_ai.usage.input_tokens": 10, "gen_ai.usage.output_tokens": 5}'],
        ],
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = spans_response

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("httpx.AsyncClient", return_value=mock_client):
        cases = await src.fetch(resolve("harness://trace/trace-1"))

    assert len(cases) == 1
    ec = cases[0]
    assert ec.input == "Hello"
    assert ec.output == "Hi there"
    assert ec.token_count == 15


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fetch_empty_trace_returns_empty_eval_case():
    src = _make_source()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"result": {"columns": [], "rows": []}}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("httpx.AsyncClient", return_value=mock_client):
        cases = await src.fetch(resolve("harness://trace-1"))

    assert len(cases) == 1
    assert cases[0].input == ""
    assert cases[0].output == ""


# ---------------------------------------------------------------------------
# fetch_traces — concurrent, error isolation
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_fetch_traces_isolates_errors():
    src = _make_source()

    call_count = 0

    async def fake_execute_hql(hql: str) -> dict:
        nonlocal call_count
        call_count += 1
        if "trace-bad" in hql:
            raise RuntimeError("QueryService error")
        return {"result": {"columns": [], "rows": []}}

    src._execute_hql = fake_execute_hql  # type: ignore[method-assign]

    cases = await src.fetch_traces(["trace-ok", "trace-bad"])

    assert len(cases) == 2
    assert call_count == 2
    # Second case should carry the error in metadata, not raise
    assert "error" in (cases[1].metadata or {})
    assert cases[1].input == ""


# ---------------------------------------------------------------------------
# list_traces
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
async def test_list_traces_returns_rows():
    src = _make_source()

    list_response = _qs_response(
        ["trace_id", "name", "duration_ms"],
        [["t1", "support-agent", 3000], ["t2", "docs-agent", 1200]],
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = list_response

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("httpx.AsyncClient", return_value=mock_client):
        traces = await src.list_traces(limit=10)

    assert len(traces) == 2
    assert traces[0]["trace_id"] == "t1"
    assert traces[1]["name"] == "docs-agent"
