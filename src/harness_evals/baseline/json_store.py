"""JsonBaselineStore — file-based baseline persistence using JSON."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness_evals.baseline.store import BaselineStore
from harness_evals.core.score import Score


def _score_to_dict(score: Score) -> dict[str, Any]:
    """Serialize a Score to a JSON-safe dict."""
    d: dict[str, Any] = {
        "name": score.name,
        "value": score.value,
        "threshold": score.threshold,
    }
    if score.reason is not None:
        d["reason"] = score.reason
    if score.metadata is not None:
        d["metadata"] = score.metadata
    d["created_at"] = score.created_at.isoformat()
    return d


def _score_from_dict(d: dict[str, Any]) -> Score:
    """Deserialize a Score from a dict."""
    created_at = datetime.fromisoformat(d["created_at"]) if "created_at" in d else datetime.now(timezone.utc)
    return Score(
        name=d["name"],
        value=d["value"],
        threshold=d["threshold"],
        reason=d.get("reason"),
        metadata=d.get("metadata"),
        created_at=created_at,
    )


class JsonBaselineStore(BaselineStore):
    """File-based baseline store using one JSON file per run.

    Layout::

        {baseline_dir}/
            {run_id}.json    # one file per saved run
            latest.json      # pointer: {"run_id": "..."}

    Each run file contains::

        {
            "run_id": "...",
            "saved_at": "ISO-8601",
            "scores": {
                "metric_name": [<score_dict>, ...],
                ...
            }
        }
    """

    def __init__(self, baseline_dir: str = ".harness-evals/baselines") -> None:
        self.baseline_dir = Path(baseline_dir)

    def save(self, run_id: str, scores: dict[str, list[Score]]) -> None:
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "run_id": run_id,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "scores": {name: [_score_to_dict(s) for s in score_list] for name, score_list in scores.items()},
        }

        run_path = self.baseline_dir / f"{run_id}.json"
        run_path.write_text(json.dumps(payload, indent=2) + "\n")

        latest_path = self.baseline_dir / "latest.json"
        latest_path.write_text(json.dumps({"run_id": run_id}) + "\n")

    def load(self, run_id: str | None = None) -> dict[str, list[Score]]:
        if run_id is None:
            run_id = self._latest_run_id()

        run_path = self.baseline_dir / f"{run_id}.json"
        if not run_path.exists():
            raise FileNotFoundError(f"Baseline run not found: {run_path}")

        payload = json.loads(run_path.read_text())
        return {name: [_score_from_dict(d) for d in score_dicts] for name, score_dicts in payload["scores"].items()}

    def list_runs(self) -> list[str]:
        if not self.baseline_dir.exists():
            return []
        runs: list[tuple[float, str]] = []
        for p in self.baseline_dir.glob("*.json"):
            if p.name == "latest.json":
                continue
            runs.append((p.stat().st_mtime, p.stem))
        runs.sort()
        return [run_id for _, run_id in runs]

    def _latest_run_id(self) -> str:
        latest_path = self.baseline_dir / "latest.json"
        if not latest_path.exists():
            raise FileNotFoundError(f"No baselines saved yet (missing {latest_path})")
        data = json.loads(latest_path.read_text())
        return data["run_id"]
