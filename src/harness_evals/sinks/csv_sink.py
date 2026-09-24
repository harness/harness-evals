"""CSV sink — long-form scores or conversation-eval pivot layout."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink

_LONG_FIELDNAMES = [
    "input",
    "metric",
    "value",
    "threshold",
    "passed",
    "reason",
    "created_at",
]

CONVERSATION_PIVOT_METRICS = [
    "outcome_goal_accuracy",
    "conversation_resolution",
    "tool_use",
    "tool_argument_match",
    "hallucination",
    "pii",
    "prompt_injection",
    "role_violation",
    "runner_v3_usage_budget",
    "sse_events_match",
]

# Write-flow conversation evals use qpe plugins with harness_* metric names.
WRITE_CONVERSATION_PIVOT_METRICS = [
    "outcome_goal_accuracy",
    "conversation_resolution",
    "tool_use",
    "tool_argument_match",
    "harness_hallucination",
    "pii",
    "prompt_injection",
    "harness_role_violation",
    "runner_v3_usage_budget",
    "sse_events_match",
]

CONVERSATION_PIVOT_FIELDNAMES = [
    "golden_id",
    "scenario",
    *CONVERSATION_PIVOT_METRICS,
    "total_duration_ms",
    "total_cost_usd",
    "total_tool_calls",
    "total_turns",
]


def conversation_pivot_fieldnames(metrics: list[str]) -> list[str]:
    return [
        "golden_id",
        "scenario",
        *metrics,
        "total_duration_ms",
        "total_cost_usd",
        "total_tool_calls",
        "total_turns",
    ]


def _format_pivot_value(value: Any) -> str:
    if value is None:
        return "?"
    number = float(value)
    if number == int(number):
        return str(int(number))
    return f"{number:.4f}".rstrip("0").rstrip(".")


def _format_labeled_value(value: Any, label: str) -> str:
    return f"{label}: {_format_pivot_value(value)}"


def _observed_usage(scores: list[Score]) -> dict[str, Any]:
    for score in scores:
        if score.name != "runner_v3_usage_budget":
            continue
        metadata = score.metadata or {}
        observed = metadata.get("observed") or {}
        return observed if isinstance(observed, dict) else {}
    return {}


def _golden_id(eval_case: EvalCase) -> str:
    metadata = eval_case.metadata or {}
    golden_id = metadata.get("golden_id") or metadata.get("id")
    return str(golden_id) if golden_id is not None else ""


def _scenario_text(eval_case: EvalCase) -> str:
    metadata = eval_case.metadata or {}
    scenario = metadata.get("scenario")
    if scenario is not None and str(scenario).strip():
        return str(scenario)
    return str(eval_case.input)


def _run_failed(scores: list[Score], eval_case: EvalCase) -> bool:
    """True when the target/simulator failed before a meaningful eval ran."""
    metadata = eval_case.metadata or {}
    if metadata.get("simulate_error"):
        return True
    return any((score.metadata or {}).get("target_error") for score in scores)


class CsvSink(BaseSink):
    """Write scores to CSV.

    * ``format: long`` (default) — one row per (eval_case, metric); appends across writes.
    * ``format: conversation_pivot`` — one row per eval case with metric columns
      labeled ``{label}: <value>`` (e.g. ``grok: 0.75``). Overwrites on ``finalize()``.
    """

    def __init__(
        self,
        path: str,
        *,
        format: str = "long",
        label: str = "run",
        pivot_metrics: list[str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.format = format
        self.label = label
        self._pivot_metrics = list(pivot_metrics or CONVERSATION_PIVOT_METRICS)
        self._pivot_fieldnames = conversation_pivot_fieldnames(self._pivot_metrics)
        self._pivot_rows: list[dict[str, str]] = []

    def write(self, scores: list[Score], eval_case: EvalCase) -> None:
        if self.format == "conversation_pivot":
            self._pivot_rows.append(self._build_pivot_row(scores, eval_case))
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.path.exists() and self.path.stat().st_size > 0

        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_LONG_FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            input_preview = str(eval_case.input)[:120]
            for score in scores:
                writer.writerow(
                    {
                        "input": input_preview,
                        "metric": score.name,
                        "value": f"{score.value:.4f}",
                        "threshold": f"{score.threshold:.4f}",
                        "passed": score.passed,
                        "reason": score.reason or "",
                        "created_at": score.created_at.isoformat(),
                    }
                )

    def _build_pivot_row(self, scores: list[Score], eval_case: EvalCase) -> dict[str, str]:
        score_map = {score.name: score for score in scores}
        observed = _observed_usage(scores)
        failed = _run_failed(scores, eval_case)
        row: dict[str, str] = {
            "golden_id": _golden_id(eval_case),
            "scenario": _scenario_text(eval_case),
        }
        for metric in self._pivot_metrics:
            value = score_map.get(metric)
            metric_value = None if failed else (value.value if value is not None else None)
            row[metric] = _format_labeled_value(metric_value, self.label)
        # Prefer runner_v3 observed duration; fall back to EvalCase.latency_ms.
        duration_ms = observed.get("duration_ms")
        if duration_ms is None:
            duration_ms = eval_case.latency_ms
        row["total_duration_ms"] = _format_labeled_value(duration_ms, self.label)
        row["total_cost_usd"] = _format_labeled_value(observed.get("cost_usd"), self.label)
        row["total_tool_calls"] = _format_labeled_value(
            observed.get("tool_count"), self.label
        )
        row["total_turns"] = _format_labeled_value(observed.get("num_turns"), self.label)
        return row

    def finalize(self) -> None:
        if self.format != "conversation_pivot" or not self._pivot_rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._pivot_fieldnames)
            writer.writeheader()
            writer.writerows(self._pivot_rows)
        print(
            f"Wrote {len(self._pivot_rows)} row(s) to {self.path.resolve()}",
            file=sys.stderr,
        )
        self._pivot_rows.clear()
