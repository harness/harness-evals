"""Group-scope admission: reviewed allowlist and diagnostics-driven validation."""

from __future__ import annotations

import pytest

from harness_evals.adapters import GroupDiagnostics, GroupIssue, GroupUsageCoverage
from harness_evals.adapters.group import GroupIssueCount
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import EvaluationMode, OnlineScope
from harness_evals.core.types import Message, ToolCall
from harness_evals.metrics import (
    ExactMatchMetric,
    MCPTraceCompletenessMetric,
    ToolArgumentMatchMetric,
    ToolCorrectnessMetric,
    ToolSelectionAccuracyMetric,
    TurnLatencyMetric,
    TurnTokenCostMetric,
)
from harness_evals.validation import (
    ExecutionKind,
    ReasonCode,
    ValidationContext,
    describe_configured_metric,
    validate_metric,
)

pytestmark = pytest.mark.unit


def _diagnostics(**overrides) -> GroupDiagnostics:
    values = dict(
        member_count=2,
        span_count=4,
        duplicate_span_count=0,
        message_count=2,
        tool_execution_count=1,
        complete_member_count=2,
        incomplete_member_count=0,
        unknown_member_count=0,
        issues=(),
        input_tokens=GroupUsageCoverage(observed=2, selected=2),
        output_tokens=GroupUsageCoverage(observed=2, selected=2),
        cost_usd=GroupUsageCoverage(observed=1, selected=1),
    )
    values.update(overrides)
    return GroupDiagnostics(**values)


def _context(diagnostics: GroupDiagnostics | None) -> ValidationContext:
    return ValidationContext(
        mode=EvaluationMode.ONLINE,
        scope=OnlineScope.GROUP,
        execution=ExecutionKind.CASE,
        available_resources=None,
        group_diagnostics=diagnostics,
    )


def _case(**extra) -> EvalCase:
    values = dict(
        input="",
        output="",
        tool_calls=[ToolCall("lookup", {"k": 1})],
        expected_tools=["lookup"],
        expected_tool_calls=[ToolCall("lookup", {"k": 1})],
        messages=[],
    )
    values.update(extra)
    return EvalCase(**values)


def _codes(result) -> set[ReasonCode]:
    return {reason.code for reason in result.reasons}


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ToolCorrectnessMetric(),
        lambda: ToolSelectionAccuracyMetric(),
        lambda: ToolArgumentMatchMetric(),
        lambda: MCPTraceCompletenessMetric([ToolCall("lookup", {"k": 1})]),
        lambda: TurnLatencyMetric(),
        lambda: TurnTokenCostMetric(),
    ],
)
def test_allowlisted_kinds_are_group_compatible(factory) -> None:
    contract = describe_configured_metric(factory())
    assert contract.group_compatible is True
    assert OnlineScope.GROUP in contract.online_scopes


def test_non_allowlisted_metric_is_not_group_compatible() -> None:
    contract = describe_configured_metric(ExactMatchMetric())
    assert contract.group_compatible is False
    assert OnlineScope.GROUP not in contract.online_scopes


def test_eligible_group_context_for_turn_latency() -> None:
    contract = describe_configured_metric(TurnLatencyMetric())
    case = _case(messages=[Message("user", "q"), Message("assistant", "a", latency_ms=120.0)])
    result = validate_metric(contract, case, context=_context(_diagnostics()))
    assert result.eligible is True
    assert result.ready


def test_unsupported_scope_for_non_group_metric() -> None:
    contract = describe_configured_metric(ExactMatchMetric())
    result = validate_metric(contract, _case(), context=_context(_diagnostics()))
    assert result.eligible is False
    assert ReasonCode.UNSUPPORTED_SCOPE in _codes(result)


def test_incomplete_capture_blocks_requires_complete_metric() -> None:
    contract = describe_configured_metric(ToolCorrectnessMetric())
    assert contract.group_requires_complete is True
    diagnostics = _diagnostics(complete_member_count=1, incomplete_member_count=1)
    result = validate_metric(contract, _case(), context=_context(diagnostics))
    incomplete = [r for r in result.reasons if r.code is ReasonCode.INCOMPLETE_CAPTURE]
    assert incomplete
    assert not any(getattr(r, "warning", False) for r in incomplete)
    assert not result.ready


def test_incomplete_capture_is_only_a_warning_when_completeness_not_required() -> None:
    contract = describe_configured_metric(TurnLatencyMetric())
    diagnostics = _diagnostics(
        complete_member_count=1,
        incomplete_member_count=1,
        issues=(GroupIssueCount(GroupIssue.AMBIGUOUS_USAGE, 1),),
    )
    result = validate_metric(contract, _case(), context=_context(diagnostics))
    assert ReasonCode.INCOMPLETE_CAPTURE not in _codes(result)
    assert ReasonCode.PARTIAL_EVIDENCE in _codes(result)


def test_missing_group_diagnostics_is_not_eligible_or_ready() -> None:
    contract = describe_configured_metric(TurnLatencyMetric())
    result = validate_metric(contract, _case(), context=_context(None))
    assert not result.ready
    assert ReasonCode.UNKNOWN_CAPTURE in _codes(result)
