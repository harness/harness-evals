"""Example custom metric: grade MULTIPLE SSE events (not just the primary output).

``StreamingHttpTarget`` selects a single primary output via ``output_event`` /
``output_path`` (graded by normal metrics like ``contains``). But a streamed run
often has several interesting events — tool requests, tool results, review
elicitations, usage — that you also want to assert on. This metric grades across
the whole captured stream.

It reads ``EvalCase.metadata["sse_events"]`` (populated by the target; capture
ALL events by leaving ``capture_events`` unset) and runs a list of per-event
``checks``. The score is the fraction of checks that passed, gated by the
metric's ``threshold`` (e.g. ``threshold: 0.8`` => at least 80% must pass).

Checks come from two places and are merged (global first, then per-row):

  * ``params.checks`` in the eval config — GLOBAL, applied to every dataset row.
    Put input-agnostic policy here (e.g. "a tool request was made", "usage was
    reported", "the changeset creates a table").
  * ``golden.metadata["sse_checks"]`` — PER-ROW, one set per dataset line. Put
    input-specific expectations here (this row's table name, schema id, columns)
    so a dataset with many rows keeps its expectations in the data, not the
    config. The metadata key is configurable via ``params.row_checks_key``.

The check schema is identical in both places, so the metric stays generic — it
never hardcodes any event name or value.

Config (note the ``params:`` wrapper — that's how the runner forwards kwargs):

    plugins:
      - examples.sse_events_match_metric
    metrics:
      - kind: sse_events_match
        threshold: 0.8
        params:
          checks:
            # assert the changeset YAML contains a substring
            - {event: elicitation_yaml, path: $.content.yaml, contains: createTable, occurrence: last}
            # assert a substring equal to the golden's `expected`
            - {event: elicitation_yaml, path: $.content.yaml, contains_expected: true}
            # assert several fields on the same selected payload/item
            - event: assistant_tool_request
              path: $.v[*]
              match:
                - {path: $.name, contains: mcp__harness__harness_get}
                - {path: $.arguments.resource_type, equals: database_schema}
            # assert an event was emitted at all
            - {event: assistant_tool_request, exists: true}
            - {event: model_usage, exists: true}

Check keys:
    event            (required) SSE event name to look at.
    exists           bool — pass if the event was (not) captured. Standalone check.
    path             JSONPath applied to each payload before matching (optional;
                     when omitted the whole payload is matched).
    contains         substring that must appear in the extracted value.
    contains_expected  bool — use the golden's ``expected`` as the substring.
    equals           value the extracted value must equal.
    match            list of nested checks that must all pass on the same selected
                     payload/item. Useful when correlating fields from one tool
                     call (e.g. name + arguments.resource_type) instead of
                     allowing separate checks to match different tool calls.
    occurrence       which captured payload(s) of the event to test:
                     ``any`` (default), ``first``, or ``last``.

Run from the repo root so ``examples`` is importable:

    PYTHONPATH=. poetry run harness-evals run examples/streaming-http.harness.local.eval.yaml
"""

from __future__ import annotations

import logging
from typing import Any

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.logging_config import compact_json
from harness_evals.plugins import register_metric
from harness_evals.utils.path import extract_path

_logger = logging.getLogger(__name__)


@register_metric("sse_events_match")
class SseEventsMatchMetric(BaseMetric):
    """Grade multiple SSE events with per-event checks; score = fraction passed."""

    def __init__(
        self,
        checks: list[dict[str, Any]] | None = None,
        threshold: float = 1.0,
        row_checks_key: str = "sse_checks",
        **kwargs: object,
    ) -> None:
        super().__init__(name="sse_events_match", dimension=Dimension.TRAJECTORY, threshold=threshold, **kwargs)
        # Global checks from the eval config — applied to every row.
        self.checks = checks or []
        # Golden metadata key holding per-row checks (input-specific expectations
        # that vary per dataset row). Merged on top of the global checks.
        self.row_checks_key = row_checks_key

    def measure(self, eval_case: EvalCase) -> Score:
        row_checks = eval_case.meta(self.row_checks_key) or []
        if not isinstance(row_checks, list):
            row_checks = []
        checks = [*self.checks, *row_checks]

        if not checks:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"No checks configured (set params.checks or golden metadata[{self.row_checks_key!r}]).",
            )

        sse_events: dict[str, list[Any]] = eval_case.meta("sse_events") or {}
        results = [_run_check(check, sse_events, eval_case) for check in checks]

        passed = sum(1 for r in results if r["passed"])
        total = len(results)
        value = passed / total if total else 0.0

        failures = [r for r in results if not r["passed"]]
        reason = f"{passed}/{total} event checks passed"
        if failures:
            reason += " | failed: " + ", ".join(r["label"] for r in failures)
            for failure in failures:
                _logger.debug(
                    "sse_events_match check failed: %s — %s",
                    failure["label"],
                    failure.get("detail"),
                )

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=reason,
            metadata={"checks": results},
        )


def _run_check(check: dict[str, Any], sse_events: dict[str, list[Any]], eval_case: EvalCase) -> dict[str, Any]:
    event = check.get("event")
    label = _label(check)
    if not event:
        return {"passed": False, "label": label, "detail": "check missing 'event'"}

    payloads = sse_events.get(event, [])

    # Presence-only check.
    if "exists" in check:
        want = bool(check["exists"])
        present = len(payloads) > 0
        return {"passed": present == want, "label": label, "detail": f"present={present} want={want}"}

    if not payloads:
        return {"passed": False, "label": label, "detail": "event not captured"}

    candidates = _select(payloads, check.get("occurrence", "any"))

    expected_substr = str(eval_case.expected) if check.get("contains_expected") else check.get("contains")
    equals = check.get("equals")
    path = check.get("path")
    nested_match = check.get("match")
    actual_values: list[Any] = []

    for payload in candidates:
        value = extract_path(payload, path) if path else payload
        values = value if isinstance(value, list) else [value]
        actual_values.extend(values)
        if nested_match is not None:
            for item in values:
                if _matches_all(item, nested_match, eval_case):
                    return {"passed": True, "label": label, "detail": "matched correlated fields"}
        if expected_substr is not None and any(expected_substr in str(v) for v in values if v is not None):
            return {"passed": True, "label": label, "detail": f"matched contains={expected_substr!r}"}
        if equals is not None and any(v == equals for v in values):
            return {"passed": True, "label": label, "detail": f"matched equals={equals!r}"}

    if expected_substr is None and equals is None:
        # No value assertion beyond presence -> passes because payloads exist.
        return {"passed": True, "label": label, "detail": "event present, no value assertion"}

    detail = _failure_detail(check, actual_values, eval_case)
    return {"passed": False, "label": label, "detail": detail}


def _matches_all(item: Any, checks: Any, eval_case: EvalCase) -> bool:
    if not isinstance(checks, list) or not checks:
        return False
    return all(_matches_one(item, check, eval_case) for check in checks if isinstance(check, dict))


def _matches_one(item: Any, check: dict[str, Any], eval_case: EvalCase) -> bool:
    path = check.get("path")
    value = extract_path(item, path) if path else item
    values = value if isinstance(value, list) else [value]

    expected_substr = str(eval_case.expected) if check.get("contains_expected") else check.get("contains")
    if expected_substr is not None:
        return any(expected_substr in str(v) for v in values if v is not None)

    if "equals" in check:
        return any(v == check["equals"] for v in values)

    # Presence-only nested check: the path resolved to at least one non-null value.
    return any(v is not None for v in values)


def _select(payloads: list[Any], occurrence: str) -> list[Any]:
    if occurrence == "first":
        return [payloads[0]]
    if occurrence == "last":
        return [payloads[-1]]
    return payloads  # "any"


def _failure_detail(check: dict[str, Any], actual_values: list[Any], eval_case: EvalCase) -> str:
    if not actual_values:
        return "no payload matched"

    actual_summary = compact_json(actual_values)
    if check.get("contains_expected"):
        return f"no payload matched; actual={actual_summary} expected contains=<expected>"
    if "contains" in check:
        return f"no payload matched; actual={actual_summary} expected contains={check['contains']!r}"
    if "equals" in check:
        path = check.get("path")
        path_note = f" at path {path!r}" if path else ""
        return f"no payload matched{path_note}; actual={actual_summary} expected equals={check['equals']!r}"
    if "match" in check:
        return f"no payload matched correlated fields; actual={actual_summary}"
    return f"no payload matched; actual={actual_summary}"


def _label(check: dict[str, Any]) -> str:
    event = check.get("event", "?")
    if "exists" in check:
        return f"{event}.exists={check['exists']}"
    if check.get("contains_expected"):
        return f"{event}.contains=<expected>"
    if "match" in check:
        return f"{event}.match"
    if "contains" in check:
        return f"{event}.contains={check['contains']!r}"
    if "equals" in check:
        return f"{event}.equals={check['equals']!r}"
    return f"{event}.present"
