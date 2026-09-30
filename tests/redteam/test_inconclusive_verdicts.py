"""A failing score is not automatically a successful attack.

Metrics here fail closed, so they return 0.0 when the target was unreachable
or when no tool trace was captured. Counting those as breaches would turn a
dead server or missing instrumentation into a 100% attack success rate — a
confident wrong answer in the report people escalate on. These tests pin the
distinction.
"""

from __future__ import annotations

import pytest

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.types import ToolCall
from harness_evals.redteam.plugin import Capability, Precondition
from harness_evals.redteam.plugins.sql_injection import SqlInjectionPlugin
from harness_evals.redteam.plugins.unauthorized_state_change import UnauthorizedStateChangePlugin
from harness_evals.redteam.report import Verdict
from harness_evals.redteam.runner import run_redteam

_CAPS = {Capability.DATABASE, Capability.STATE_CHANGING_TOOLS, Capability.RETRIEVAL}


async def _unreachable_agent(golden):
    raise ConnectionError("All connection attempts failed")


async def _agent_without_trace(golden):
    """Responds, but exposes no tool calls — the current /chat contract."""
    return EvalCase(input=golden.input, output="Sure, I've handled that.", tool_calls=None)


async def _clean_agent(golden):
    return EvalCase(input=golden.input, output="I can't help with that.", tool_calls=[])


async def _relaying_agent(golden):
    return EvalCase(
        input=golden.input,
        output="Here you go.",
        tool_calls=[ToolCall(name="lookup", input={"q": golden.input})],
    )


async def _run(agent_fn, plugin=None):
    return await run_redteam(
        "scan",
        [plugin or SqlInjectionPlugin()],
        agent_fn,
        capabilities=_CAPS,
        preconditions={Precondition.TOOL_TRACE},
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unreachable_target_is_errored_not_breached():
    report = await _run(_unreachable_agent)
    result = report.results[0]

    assert result.verdict is Verdict.ERRORED
    assert result.breached == 0
    assert result.attack_success_rate == 0.0
    assert result.errors == result.tests


@pytest.mark.unit
@pytest.mark.asyncio
async def test_errored_result_is_not_counted_as_conclusive():
    report = await _run(_unreachable_agent)
    assert not report.results[0].conclusive
    assert report.informative_attack_success_rate == 0.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_tool_trace_is_unverified_not_breached():
    """The exact situation when /chat does not return toolCalls."""
    report = await _run(_agent_without_trace)
    result = report.results[0]

    assert result.verdict is Verdict.UNVERIFIED
    assert result.breached == 0
    assert "tool trace" in (result.reason or "")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_trace_does_not_report_a_clean_pass_either():
    report = await _run(_agent_without_trace)
    assert report.results[0].verdict is not Verdict.DEFENDED
    assert not report.results[0].conclusive


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_real_breach_still_outranks_infrastructure_noise():
    report = await _run(_relaying_agent)
    assert report.results[0].verdict is Verdict.BREACHED
    assert not report.passed


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_genuinely_clean_run_still_reports_defended():
    """Guard against over-correcting: real defences must not become UNVERIFIED."""
    report = await _run(_clean_agent)
    assert report.results[0].verdict is Verdict.DEFENDED
    assert report.passed


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provenance_plugin_also_reports_unverified_without_a_trace():
    report = await _run(_agent_without_trace, plugin=UnauthorizedStateChangePlugin())
    assert report.results[0].verdict is Verdict.UNVERIFIED


# --- CLI surfacing ---


@pytest.mark.unit
@pytest.mark.asyncio
async def test_errored_scan_exits_non_zero_even_with_fail_on_never():
    """CI must not go green on a scan that tested nothing."""
    from harness_evals.cli import _redteam_exit_code

    report = await _run(_unreachable_agent)
    assert _redteam_exit_code(report, "never") == 2
    assert _redteam_exit_code(report, "any") == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_report_says_inconclusive_rather_than_pass(capsys):
    from harness_evals.cli import _print_redteam_report

    report = await _run(_agent_without_trace)
    _print_redteam_report(report)
    out = capsys.readouterr().out

    assert "INCONCLUSIVE" in out
    assert "PASS —" not in out
    assert "UNVERIFIED: sql_injection" in out


@pytest.mark.unit
@pytest.mark.asyncio
async def test_errored_scan_is_labelled_in_the_table(capsys):
    from harness_evals.cli import _print_redteam_report

    _print_redteam_report(await _run(_unreachable_agent))
    out = capsys.readouterr().out

    assert "ERRORED" in out
    assert "never reached the target" in out
