"""HarnessSink — POST run items and scores back to the Harness AI Evals API.

This sink is activated when the CLI runs as a subprocess of the ai-evals API
server. It buffers per-item results during evaluation and batch-POSTs them
to ``POST /v1/orgs/{org}/projects/{project}/runs/{run_id}/items`` on
``finalize()``.

Required environment variables (set by cli_executor.py):
  HARNESS_BASE_URL  — ai-evals API base URL (e.g. http://localhost:8080/api)
  HARNESS_API_KEY   — auth token for the callback
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink

logger = logging.getLogger(__name__)


class HarnessSink(BaseSink):
    """Posts eval results back to the Harness AI Evals API as run items."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        run_id: str,
        org: str,
        project: str,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._run_id = run_id
        self._org = org
        self._project = project
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
        self._items_buffer: list[dict[str, Any]] = []

    def write(self, scores: list[Score], eval_case: EvalCase) -> None:
        """Buffer a run item with its scores for later batch POST."""
        # Extract dataset_item_id from score metadata (set by EvaluationEngine)
        dataset_item_id = ""
        if scores and scores[0].metadata:
            dataset_item_id = scores[0].metadata.get("dataset_item_id", "")

        # Determine item status from scores
        all_passed = all(s.passed for s in scores) if scores else True
        has_error = any(
            s.metadata and s.metadata.get("error") == "target_failed"
            for s in scores
        )
        status = "error" if has_error else ("passed" if all_passed else "failed")

        # Build dataset_item_snapshot from eval_case
        snapshot: dict[str, Any] = {"input": eval_case.input}
        if eval_case.expected is not None:
            snapshot["expected"] = eval_case.expected
        if eval_case.tags:
            snapshot["tags"] = eval_case.tags
        if eval_case.metadata:
            snapshot["metadata"] = eval_case.metadata

        # Build output dict
        output: dict[str, Any] = {"output": eval_case.output}
        if eval_case.latency_ms is not None:
            output["latency_ms"] = eval_case.latency_ms
        if eval_case.cost_usd is not None:
            output["cost_usd"] = eval_case.cost_usd

        self._items_buffer.append({
            "dataset_item_id": str(dataset_item_id),
            "dataset_item_snapshot": snapshot,
            "output": output,
            "status": status,
            "scores": [
                {
                    "score_name": s.name,
                    "value": s.value,
                    "threshold": s.threshold,
                    "eval_id": (s.metadata or {}).get("eval_id"),
                    "reason": s.reason,
                }
                for s in scores
            ],
        })

    def finalize(self) -> None:
        """Batch POST all buffered items to the API."""
        if not self._items_buffer:
            return

        url = f"/v1/orgs/{self._org}/projects/{self._project}/runs/{self._run_id}/items"
        try:
            resp = self._client.post(url, json={"items": self._items_buffer})
            resp.raise_for_status()
            logger.info(
                "HarnessSink: posted %d run items for run %s",
                len(self._items_buffer),
                self._run_id,
            )
        except httpx.HTTPStatusError as e:
            logger.error(
                "HarnessSink: failed to post run items: %s %s",
                e.response.status_code,
                e.response.text[:500],
            )
            raise
        except httpx.HTTPError as e:
            logger.error("HarnessSink: HTTP error posting run items: %s", e)
            raise

    def shutdown(self) -> None:
        """Close the HTTP client."""
        self._client.close()
