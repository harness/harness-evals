"""Harness OTEL eval-case importer — fetch traces from Harness QueryService.

Calls the QueryService gRPC-web endpoint exposed through the Harness gateway,
then converts the returned spans into EvalCase objects using the same logic as
OTELEvalCaseSource.

Requires: pip install harness-evals[harness]

Environment variables (all optional if passed to constructor):
    HARNESS_BASE_URL   — e.g. https://app.harness.io  (default)
    HARNESS_API_KEY    — PAT or SAT (pat.account.xxx.yyy or sat.account.xxx)
    HARNESS_ACCOUNT_ID — Harness account identifier
    HARNESS_ORG_ID     — Org identifier
    HARNESS_PROJECT_ID — Project identifier
"""

from __future__ import annotations

import json
import os
from typing import Any

from harness_evals.core.eval_case import EvalCase
from harness_evals.importers.base import BaseEvalCaseSource
from harness_evals.importers.otel import _build_conversation_eval_case
from harness_evals.plugins import register_eval_case_source
from harness_evals.refs import ResourceRef

_GATEWAY_PATH = (
    "/gateway/query-service/grpc/"
    "io.harness.platform.query.service.api.v1.QueryServiceGrpc/executeQuery"
)

# HQL that fetches all spans for a single trace, ordered chronologically.
# Column selection matches what online_eval_service._fetch_spans() selects so
# the same _build_conversation_eval_case() logic works on both paths.
_TRACE_HQL = (
    'find event "genai:agent_trace"'
    " | filter trace_id = '{trace_id}'"
    " | select {{ trace_id, span_id, parent_span_id, name,"
    "   start_timestamp, duration_ms, status_code,"
    "   input_tokens, output_tokens, model, tool_name,"
    "   service_name, attributes }}"
    " | order_by start_timestamp asc"
    " | limit 1000"
)


@register_eval_case_source("harness")
class HarnessOTELEvalCaseSource(BaseEvalCaseSource):
    """Fetch EvalCases from production traces stored in the Harness platform.

    Queries the Harness QueryService (ClickHouse backend) via the gRPC-web
    gateway endpoint using a PAT or SAT for auth.

    Usage::

        from harness_evals.refs import resolve

        source = HarnessOTELEvalCaseSource(
            api_key="pat.acctId.xxx.yyy",
            account_id="acctId",
            org_id="MyOrg",
            project_id="MyProject",
        )

        # Single trace by ID
        cases = await source.fetch(resolve("harness://trace/abc123"))

        # Multiple traces
        cases = await source.fetch_traces(["abc123", "def456"])

    ``ref.id`` is interpreted as ``trace/<trace_id>`` or just ``<trace_id>``.
    """

    name = "harness"

    def __init__(
        self,
        api_key: str | None = None,
        account_id: str | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        try:
            import httpx  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Install dependencies: pip install harness-evals[harness]"
            ) from e

        self._api_key = api_key or os.environ.get("HARNESS_API_KEY", "")
        self._account_id = account_id or os.environ.get("HARNESS_ACCOUNT_ID", "")
        self._org_id = org_id or os.environ.get("HARNESS_ORG_ID", "")
        self._project_id = project_id or os.environ.get("HARNESS_PROJECT_ID", "")
        raw_base = base_url or os.environ.get("HARNESS_BASE_URL", "https://app.harness.io")
        self._base_url = raw_base.rstrip("/")
        self._timeout = timeout

        if not self._api_key:
            raise ValueError("No API key: pass api_key= or set HARNESS_API_KEY")
        if not self._account_id:
            raise ValueError("No account ID: pass account_id= or set HARNESS_ACCOUNT_ID")

    # ------------------------------------------------------------------
    # BaseEvalCaseSource
    # ------------------------------------------------------------------

    async def fetch(self, ref: ResourceRef) -> list[EvalCase]:
        """Fetch a single trace and return it as a one-element list.

        ``ref.id`` must be ``trace/<trace_id>`` or just ``<trace_id>``.
        """
        trace_id = _parse_trace_id(ref.id)
        spans = await self._fetch_spans(trace_id)
        return [_build_conversation_eval_case(spans)]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    async def fetch_traces(self, trace_ids: list[str]) -> list[EvalCase]:
        """Fetch multiple traces concurrently and return one EvalCase per trace."""
        import asyncio

        results = await asyncio.gather(
            *[self._fetch_and_build(tid) for tid in trace_ids],
            return_exceptions=True,
        )
        cases: list[EvalCase] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Return an empty EvalCase rather than dropping the item so
                # the caller can still detect the failure by index.
                cases.append(
                    EvalCase(
                        input="",
                        output="",
                        metadata={"trace_id": trace_ids[i], "error": str(result)},
                    )
                )
            else:
                cases.append(result)  # type: ignore[arg-type]
        return cases

    async def list_traces(
        self,
        *,
        limit: int = 50,
        lookback_hours: int | None = None,
        lookback_days: int = 7,
        extra_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """List recent root-level agent traces for this org/project.

        Returns a list of dicts with keys: trace_id, name, status_code,
        duration_ms, start_timestamp, model, input_tokens, output_tokens.

        ``lookback_hours`` takes precedence over ``lookback_days`` when both
        are supplied.  Use ``lookback_hours=4`` to fetch the last 4 hours.

        ``extra_filter`` is appended as an additional HQL ``and`` clause, e.g.
        ``"service_name != 'my-service'"`` or ``"model = 'gpt-4o'"``.
        """
        extra_clause = f" and {extra_filter}" if extra_filter else ""
        lookback_expr = f"ago({lookback_hours}h)" if lookback_hours is not None else f"ago({lookback_days}d)"
        hql = (
            'find event "genai:agent_trace"'
            f" | filter start_timestamp >= {lookback_expr}"
            " and start_timestamp < now()"
            " and parent_span_id = ''"
            f" and account_id = '{self._account_id}'"
            f" and org_id = '{self._org_id}'"
            f" and project_id = '{self._project_id}'"
            f"{extra_clause}"
            " | group_by trace_id"
            " | select { trace_id,"
            "   min(service_name) as name,"
            "   min(status_code) as status_code,"
            "   min(duration_ms) as duration_ms,"
            "   min(start_timestamp) as start_timestamp,"
            "   sum(input_tokens) as input_tokens,"
            "   sum(output_tokens) as output_tokens,"
            "   min(model) as model }"
            " | order_by start_timestamp desc"
            f" | limit {limit}"
        )
        raw = await self._execute_hql(hql)
        return _parse_result_rows(raw)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _fetch_and_build(self, trace_id: str) -> EvalCase:
        spans = await self._fetch_spans(trace_id)
        return _build_conversation_eval_case(spans)

    async def _fetch_spans(self, trace_id: str) -> list[dict[str, Any]]:
        hql = _TRACE_HQL.format(trace_id=trace_id)
        raw = await self._execute_hql(hql)
        spans = _parse_result_rows(raw)
        # Normalise: attributes is a JSON string from ClickHouse → parse to dict
        for span in spans:
            attrs = span.get("attributes")
            if isinstance(attrs, str):
                try:
                    span["attributes"] = json.loads(attrs)
                except (ValueError, TypeError):
                    span["attributes"] = {}
            elif not isinstance(attrs, dict):
                span["attributes"] = {}
        return spans

    async def _execute_hql(self, hql: str) -> dict[str, Any]:
        """POST the HQL to the QueryService gRPC-web gateway and return raw JSON."""
        import httpx

        url = self._build_url()
        headers = self._build_headers()
        body = {"queryString": hql, "params": {"query_params": []}}

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, headers=headers, json=body)

        if resp.status_code == 401:
            raise PermissionError(
                "QueryService returned 401 — check api_key and account_id"
            )
        if resp.status_code == 403:
            raise PermissionError(
                "QueryService returned 403 — token lacks access to this org/project"
            )
        if resp.status_code != 200:
            raise RuntimeError(
                f"QueryService returned {resp.status_code}: {resp.text[:300]}"
            )

        return resp.json()  # type: ignore[no-any-return]

    def _build_url(self) -> str:
        params = (
            f"routingId={self._account_id}"
            f"&accountIdentifier={self._account_id}"
            f"&orgIdentifier={self._org_id}"
            f"&projectIdentifier={self._project_id}"
        )
        return f"{self._base_url}{_GATEWAY_PATH}?{params}"

    def _build_headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "harness-account": self._account_id,
            "x-tenant-id": "default",
            "content-type": "application/json",
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _parse_trace_id(ref_id: str) -> str:
    """Extract trace_id from ``trace/<id>`` or bare ``<id>``."""
    if ref_id.startswith("trace/"):
        return ref_id[len("trace/"):]
    return ref_id


def _parse_result_rows(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert QueryService JSON response into a list of row dicts."""
    result = response.get("result") or {}
    columns = [col.get("name", "") for col in result.get("columns", [])]
    rows = result.get("rows", [])
    if not columns or not rows:
        return []

    records: list[dict[str, Any]] = []
    for row in rows:
        values = row.get("values", [])
        record: dict[str, Any] = {}
        for i, col_name in enumerate(columns):
            record[col_name] = values[i] if i < len(values) else None
        records.append(record)
    return records
