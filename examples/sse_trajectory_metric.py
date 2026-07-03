"""Example custom metric: surface captured SSE trajectory events.

The default sinks (stdout, json) only print scores — not
``EvalCase.metadata``. This metric bridges that gap: it reads the SSE events
captured by ``StreamingHttpTarget`` (via its ``capture_events`` option) and
reports them through the score's ``reason`` (shown by the stdout sink) and
``metadata`` (written by the json sink).

It is intentionally generic: it summarizes whatever events were captured and,
if a ``tool_call`` event is present, lists the tool names found in it. Adapt the
event names to whatever your streaming service emits.

Load it from a YAML eval config via a ``plugins:`` entry, then reference it as a
metric ``kind``:

    plugins:
      - examples.sse_trajectory_metric
    metrics:
      - {kind: sse_trajectory, threshold: 1.0}

Because ``examples`` is not an installed package, run from the repo root with it
on the import path:

    PYTHONPATH=. poetry run harness-evals run examples/streaming-http.eval.yaml
"""

from __future__ import annotations

from typing import Any

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.plugins import register_metric


@register_metric("sse_trajectory")
class SseTrajectoryMetric(BaseMetric):
    """Summarize captured SSE events into a score.

    Reads ``EvalCase.metadata["sse_events"]`` (populated by
    ``StreamingHttpTarget``'s ``capture_events``) and reports per-event counts
    plus any tool names found in ``tool_call`` events. Scores 1.0 when at least
    one event was captured, else 0.0 — so a run that never streamed the expected
    events fails loudly instead of silently passing.
    """

    def __init__(self, threshold: float = 1.0, **kwargs: object) -> None:
        super().__init__(name="sse_trajectory", dimension=Dimension.TRAJECTORY, threshold=threshold, **kwargs)

    def measure(self, eval_case: EvalCase) -> Score:
        sse_events: dict[str, list[Any]] = eval_case.meta("sse_events") or {}
        if not sse_events:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="No SSE events captured (check target.capture_events and that the endpoint streamed).",
            )

        counts = {name: len(payloads) for name, payloads in sse_events.items()}
        tool_calls = _tool_names(sse_events.get("tool_call", []))

        reason = "captured: " + ", ".join(f"{name}={n}" for name, n in sorted(counts.items()))
        if tool_calls:
            reason += f" | tools: {', '.join(tool_calls)}"

        return Score(
            name=self.name,
            value=1.0,
            threshold=self.threshold,
            reason=reason,
            metadata={"event_counts": counts, "tool_calls": tool_calls},
        )


def _tool_names(events: list[Any]) -> list[str]:
    """Extract tool names from captured ``tool_call`` payloads (``{"name": ...}``)."""
    names: list[str] = []
    for payload in events:
        if isinstance(payload, dict) and payload.get("name"):
            names.append(str(payload["name"]))
    return names
