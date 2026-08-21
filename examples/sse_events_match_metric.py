"""Example custom metric: grade MULTIPLE SSE events (not just the primary output).

``StreamingHttpTarget`` selects a single primary output via ``output_event`` /
``output_path`` (graded by normal metrics like ``contains``). But a streamed run
often has several interesting events — tool requests, tool results, review
elicitations, usage — that you also want to assert on. This metric grades across
the whole captured stream.

It reads ``EvalCase.metadata["sse_events"]`` (populated by the target; capture
ALL events by leaving ``capture_events`` unset) and runs a list of per-event
``checks``. The score is the fraction of checks that passed, gated by the
metric's ``threshold`` (e.g. ``threshold: 0.8`` => at least 80% must pass.

Checks come from two places and are merged (global first, then per-row):

  * ``params.checks`` in the eval config — GLOBAL, applied to every dataset row.
  * ``golden.metadata["sse_checks"]`` — PER-ROW, one set per dataset line.

Check keys:
    event              (required) SSE event name to look at.
    exists             bool — pass if the event was (not) captured.
    path               JSONPath applied to each payload before matching.
    contains           substring that must appear in the extracted value.
    contains_expected  bool — use the golden's ``expected`` as the substring.
    not_contains       nested only — none of the resolved values may contain this substring.
    forbidden_contains top-level — fail if *any* resolved value contains this substring.
    equals             value the extracted value must equal.
    skill_equals       skill name on ``$.arguments.skill`` (hyphen/underscore equivalent).
    match              nested checks that must all pass on the same payload/item.
    match_any          list of ``match`` lists; OR semantics on the same item.
    occurrence           ``any`` (default), ``first``, ``second``, ``third``, or ``last``.
                       Ordinals index into captured payloads of that event type (1-based names).
    events               optional list of event names — payloads are merged across all listed types
                       (use instead of ``event`` for OR-across-event-type checks).
    optional             when true, the check passes if no matching events were captured.
"""

from __future__ import annotations

import logging
from typing import Any

from examples.skill_sse_checks import skill_names_equivalent

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
        self.checks = checks or []
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
    events = check.get("events")
    label = _label(check)
    if not event and not events:
        return {"passed": False, "label": label, "detail": "check missing 'event' or 'events'"}

    if events:
        payloads: list[Any] = []
        for event_name in events:
            if isinstance(event_name, str):
                payloads.extend(sse_events.get(event_name, []))
    else:
        payloads = sse_events.get(event, [])

    if "exists" in check:
        want = bool(check["exists"])
        present = len(payloads) > 0
        return {"passed": present == want, "label": label, "detail": f"present={present} want={want}"}

    forbidden = check.get("forbidden_contains")
    if forbidden is not None:
        if not payloads:
            return {
                "passed": True,
                "label": label,
                "detail": "event not captured; forbidden substring absent",
            }
        candidates = _select(payloads, check.get("occurrence", "any"))
        missing = _missing_occurrence_detail(check, payloads, candidates)
        if missing:
            return {"passed": False, "label": label, "detail": missing}
        return _check_forbidden_contains(label, candidates, check.get("path"), str(forbidden))

    if not payloads:
        if check.get("optional"):
            return {
                "passed": True,
                "label": label,
                "detail": "optional check skipped (no matching events captured)",
            }
        return {"passed": False, "label": label, "detail": "event not captured"}

    candidates = _select(payloads, check.get("occurrence", "any"))
    missing = _missing_occurrence_detail(check, payloads, candidates)
    if missing:
        return {"passed": False, "label": label, "detail": missing}
    path = check.get("path")

    expected_substr = str(eval_case.expected) if check.get("contains_expected") else check.get("contains")
    equals = check.get("equals")
    nested_match = check.get("match")
    match_any = check.get("match_any")
    actual_values: list[Any] = []

    for payload in candidates:
        value = extract_path(payload, path) if path else payload
        values = value if isinstance(value, list) else [value]
        actual_values.extend(values)
        if match_any is not None:
            for item in values:
                if isinstance(match_any, list):
                    for alternative in match_any:
                        if _matches_all(item, alternative, eval_case):
                            return {
                                "passed": True,
                                "label": label,
                                "detail": f"matched {_describe_check_expectation(check)}",
                            }
        if nested_match is not None:
            for item in values:
                if _matches_all(item, nested_match, eval_case):
                    return {
                        "passed": True,
                        "label": label,
                        "detail": f"matched {_describe_check_expectation(check)}",
                    }
        if expected_substr is not None and any(expected_substr in str(v) for v in values if v is not None):
            return {
                "passed": True,
                "label": label,
                "detail": f"matched {_describe_check_expectation(check)}",
            }
        if equals is not None and any(v == equals for v in values):
            return {
                "passed": True,
                "label": label,
                "detail": f"matched {_describe_check_expectation(check)}",
            }

    if expected_substr is None and equals is None and nested_match is None and match_any is None:
        return {
            "passed": True,
            "label": label,
            "detail": f"event captured ({_describe_check_expectation(check)})",
        }

    detail = _failure_detail(check, actual_values, eval_case)
    return {"passed": False, "label": label, "detail": detail}


def _check_forbidden_contains(
    label: str,
    candidates: list[Any],
    path: str | None,
    forbidden: str,
) -> dict[str, Any]:
    violations: list[Any] = []
    for payload in candidates:
        value = extract_path(payload, path) if path else payload
        values = value if isinstance(value, list) else [value]
        for item in values:
            if item is not None and forbidden in str(item):
                violations.append(item)
    if violations:
        return {
            "passed": False,
            "label": label,
            "detail": (f"forbidden substring {forbidden!r} found in {compact_json(violations[:3])}"),
        }
    return {
        "passed": True,
        "label": label,
        "detail": f"no payload contains forbidden {forbidden!r}",
    }


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

    if "not_contains" in check:
        forbidden = str(check["not_contains"])
        checked = [v for v in values if v is not None]
        if not checked:
            return False
        return all(forbidden not in str(v) for v in checked)

    if "equals" in check:
        return any(v == check["equals"] for v in values)

    if "skill_equals" in check:
        expected = str(check["skill_equals"])
        for value in values:
            if value is None:
                continue
            if skill_names_equivalent(str(value), expected):
                return True
            if isinstance(value, dict):
                for key in ("skill", "skill_name", "name"):
                    nested = value.get(key)
                    if nested is not None and skill_names_equivalent(str(nested), expected):
                        return True
        return False

    return any(v is not None for v in values)


def _select(payloads: list[Any], occurrence: str) -> list[Any]:
    if occurrence == "first":
        return [payloads[0]] if payloads else []
    if occurrence == "second":
        return [payloads[1]] if len(payloads) >= 2 else []
    if occurrence == "third":
        return [payloads[2]] if len(payloads) >= 3 else []
    if occurrence == "last":
        return [payloads[-1]] if payloads else []
    return payloads


def _missing_occurrence_detail(check: dict[str, Any], payloads: list[Any], candidates: list[Any]) -> str | None:
    occurrence = check.get("occurrence", "any")
    if occurrence in (None, "any") or candidates:
        return None
    return f"no payload at occurrence={occurrence!r} (captured {len(payloads)})"


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
        return f"no payload matched {_describe_check_expectation(check)}; actual={actual_summary}"
    if "match_any" in check:
        return f"no payload matched {_describe_check_expectation(check)}; actual={actual_summary}"
    return f"no payload matched; actual={actual_summary}"


def _short_path(path: str | None) -> str:
    if not path:
        return "payload"
    short = str(path)
    if short.startswith("$."):
        short = short[2:]
    return short.replace("[*]", "")


def _format_nested_match_clause(check: dict[str, Any]) -> str:
    field = _short_path(check.get("path"))
    if check.get("contains_expected"):
        return f"{field} contains <expected>"
    if "contains" in check:
        return f"{field} contains {check['contains']!r}"
    if "not_contains" in check:
        return f"{field} not_contains {check['not_contains']!r}"
    if "equals" in check:
        return f"{field} equals {check['equals']!r}"
    if "skill_equals" in check:
        return f"{field} skill_equals {check['skill_equals']!r}"
    if check.get("path"):
        return f"{field} present"
    return "payload present"


def _describe_check_expectation(check: dict[str, Any]) -> str:
    if "exists" in check:
        return f"exists={check['exists']}"
    if "forbidden_contains" in check:
        return f"forbidden_contains={check['forbidden_contains']!r}"
    if check.get("contains_expected"):
        path = _short_path(check.get("path"))
        return f"{path} contains <expected>"
    if "contains" in check:
        path = _short_path(check.get("path"))
        return f"{path} contains {check['contains']!r}"
    if "equals" in check:
        path = _short_path(check.get("path"))
        return f"{path} equals {check['equals']!r}"
    if "match" in check:
        nested = check.get("match")
        if isinstance(nested, list) and nested:
            clauses = [_format_nested_match_clause(m) for m in nested if isinstance(m, dict)]
            if clauses:
                return "correlated fields: " + "; ".join(clauses)
        return "correlated fields (unspecified)"
    if "match_any" in check:
        alternatives = check.get("match_any")
        if isinstance(alternatives, list) and alternatives:
            rendered: list[str] = []
            for alternative in alternatives:
                if not isinstance(alternative, list):
                    continue
                clauses = [_format_nested_match_clause(m) for m in alternative if isinstance(m, dict)]
                if clauses:
                    rendered.append("(" + "; ".join(clauses) + ")")
            if rendered:
                return "any of: " + " OR ".join(rendered)
        return "any correlated fields (unspecified)"
    return "present (no field assertions)"


def _label(check: dict[str, Any]) -> str:
    event = check.get("event")
    if not event:
        events = check.get("events")
        event = "+".join(events) if isinstance(events, list) and events else "?"
    return f"{event}.{_describe_check_expectation(check)}"
