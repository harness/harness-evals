"""Tests for Skill SSE check helpers and metric extensions."""

import pytest
from examples.skill_sse_checks import (
    skill_names_equivalent,
    skill_sse_checks,
    upgrade_skill_checks_in_sse_checks,
)
from examples.sse_events_match_metric import SseEventsMatchMetric

from harness_evals.core.eval_case import EvalCase


@pytest.mark.unit
@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("hql-reference", "hql_reference", True),
        ("harness-documentation", "harness_documentation", True),
        ("debug-pipeline", "debug-pipeline", True),
        ("scs-routing", "kg-analysis", False),
    ],
)
def test_skill_names_equivalent(left: str, right: str, expected: bool) -> None:
    assert skill_names_equivalent(left, right) is expected


@pytest.mark.unit
def test_skill_sse_checks_single_skill() -> None:
    checks = skill_sse_checks(["ccm-cost-categories"])
    assert len(checks) == 2
    request = checks[0]
    assert request["match"] == [
        {"path": "$.name", "equals": "Skill"},
        {"path": "$.arguments.skill", "skill_equals": "ccm-cost-categories"},
    ]


@pytest.mark.unit
def test_skill_sse_checks_multiple_skills_use_match_any() -> None:
    checks = skill_sse_checks(["hql-reference", "kg-analysis"])
    request = checks[0]
    assert "match_any" in request
    assert len(request["match_any"]) == 2


@pytest.mark.unit
def test_sse_events_match_skill_equals_normalizes_underscores() -> None:
    metric = SseEventsMatchMetric(
        checks=[
            {
                "event": "assistant_tool_request",
                "path": "$.v[*]",
                "match": [
                    {"path": "$.name", "equals": "Skill"},
                    {"path": "$.arguments.skill", "skill_equals": "hql-reference"},
                ],
            }
        ],
        threshold=1.0,
    )
    eval_case = EvalCase(
        input="audit templates",
        output="done",
        metadata={
            "sse_events": {
                "assistant_tool_request": [
                    {"v": [{"name": "Skill", "arguments": {"skill": "hql_reference"}}]}
                ],
            }
        },
    )

    score = metric.measure(eval_case)

    assert score.passed


@pytest.mark.unit
def test_sse_events_match_match_any_accepts_any_listed_skill() -> None:
    metric = SseEventsMatchMetric(
        checks=[
            {
                "event": "assistant_tool_request",
                "path": "$.v[*]",
                "match_any": [
                    [
                        {"path": "$.name", "equals": "Skill"},
                        {"path": "$.arguments.skill", "skill_equals": "hql-reference"},
                    ],
                    [
                        {"path": "$.name", "equals": "Skill"},
                        {"path": "$.arguments.skill", "skill_equals": "kg-analysis"},
                    ],
                ],
            }
        ],
        threshold=1.0,
    )
    eval_case = EvalCase(
        input="audit templates",
        output="done",
        metadata={
            "sse_events": {
                "assistant_tool_request": [
                    {"v": [{"name": "Skill", "arguments": {"skill": "kg-analysis"}}]}
                ],
            }
        },
    )

    score = metric.measure(eval_case)

    assert score.passed


@pytest.mark.unit
def test_upgrade_skill_checks_replaces_generic_skill_blocks() -> None:
    original = [
        {"event": "assistant_tool_request", "path": "$.v[*]", "match": [{"path": "$.name", "equals": "Skill"}]},
        {"event": "assistant_tool_result", "path": "$.v[*]", "match": [{"path": "$.name", "equals": "Skill"}]},
        {"event": "assistant_message", "exists": True},
    ]
    upgraded = upgrade_skill_checks_in_sse_checks(original, ["debug-pipeline"])
    request = upgraded[0]
    assert request["match"][1]["skill_equals"] == "debug-pipeline"
    assert upgraded[-1]["event"] == "assistant_message"
