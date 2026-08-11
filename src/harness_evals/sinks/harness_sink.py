"""Harness AI Evals sink — push evaluation scores into a Harness run."""

from __future__ import annotations

import os

try:
    import httpx
except ImportError as _err:
    raise ImportError(
        "HarnessSink requires the httpx package. Install with: pip install harness-evals[harness]"
    ) from _err

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink


class HarnessSink(BaseSink):
    """Push evaluation scores into a Harness AI Evals run.

    Each ``EvalCase`` must carry two metadata keys:

    - ``harness_run_id``: the run to write into
    - ``harness_dataset_item_id``: the dataset item this case corresponds to

    Credentials resolve via constructor params > environment variables:

    - ``HARNESS_ACCOUNT_ID``
    - ``HARNESS_API_KEY``
    - ``HARNESS_ORG_ID`` (default: ``"default"``)
    - ``HARNESS_PROJECT_ID``
    - ``HARNESS_BASE_URL`` (default: ``"https://app.harness.io"``)

    Example::

        from harness_evals.sinks.harness_sink import HarnessSink

        sink = HarnessSink()
        evaluate(cases, metrics=[...], sinks=[sink])
        sink.finalize()
    """

    def __init__(
        self,
        account_id: str | None = None,
        api_key: str | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._account_id = account_id or os.environ["HARNESS_ACCOUNT_ID"]
        self._api_key = api_key or os.environ["HARNESS_API_KEY"]
        self._org_id = org_id or os.environ.get("HARNESS_ORG_ID", "default")
        self._project_id = project_id or os.environ["HARNESS_PROJECT_ID"]
        _base: str = base_url if base_url is not None else os.environ.get("HARNESS_BASE_URL", "https://app.harness.io")
        self._base_url = _base.rstrip("/") + "/gateway/ai-evals/api/v1"

        self._client = httpx.Client(
            headers={
                "Harness-Account": self._account_id,
                "x-api-key": self._api_key,
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        self._completed_runs: set[str] = set()

    def write(self, scores: list[Score], eval_case: EvalCase) -> None:
        meta = eval_case.metadata or {}
        run_id = meta.get("harness_run_id")
        dataset_item_id = meta.get("harness_dataset_item_id")

        if not run_id:
            raise ValueError(
                "HarnessSink requires 'harness_run_id' in eval_case.metadata"
            )
        if not dataset_item_id:
            raise ValueError(
                "HarnessSink requires 'harness_dataset_item_id' in eval_case.metadata"
            )

        url = (
            f"{self._base_url}/orgs/{self._org_id}/projects/{self._project_id}"
            f"/runs/{run_id}/items"
        )

        payload = {
            "items": [
                {
                    "dataset_item_id": dataset_item_id,
                    "output": {"agent_response": eval_case.output_as_str()},
                    "status": "success" if all(s.passed for s in scores) else "failed",
                    "scores": [
                        {
                            "score_name": s.name,
                            "value": s.value,
                            "passed": s.passed,
                            "reason": s.reason,
                            "metadata": s.metadata,
                        }
                        for s in scores
                    ],
                }
            ]
        }

        response = self._client.post(url, json=payload)
        response.raise_for_status()

        self._completed_runs.add(run_id)

    def finalize(self) -> None:
        """Mark all runs that received items as completed."""
        for run_id in self._completed_runs:
            url = (
                f"{self._base_url}/orgs/{self._org_id}/projects/{self._project_id}"
                f"/runs/{run_id}"
            )
            self._client.patch(url, json={"status": "completed"}).raise_for_status()

    def shutdown(self) -> None:
        """Close the HTTP client."""
        self._client.close()
