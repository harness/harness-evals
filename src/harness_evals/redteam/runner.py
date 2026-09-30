"""run_redteam() — execute attack plugins against a target and report ASR."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

from harness_evals.core.runner import evaluate_dataset
from harness_evals.core.score import Score
from harness_evals.core.sink import BaseSink
from harness_evals.llm.base import BaseLLM
from harness_evals.redteam.packs import AttackPack
from harness_evals.redteam.plugin import AttackPlugin, Capability, Precondition
from harness_evals.redteam.report import PluginResult, RedTeamReport, Verdict

logger = logging.getLogger(__name__)


async def run_redteam(
    name: str,
    plugins: Iterable[AttackPlugin] | AttackPack,
    agent_fn: Callable,
    *,
    capabilities: Iterable[Capability] | None = None,
    preconditions: Iterable[Precondition] | None = None,
    judge_llm: BaseLLM | None = None,
    sinks: list[BaseSink] | None = None,
    concurrency: int | None = None,
) -> RedTeamReport:
    """Run attack plugins against *agent_fn* and return an ASR report.

    Args:
        name: Run label, used in the report header.
        plugins: Attack plugins, or an :class:`AttackPack` to resolve.
        agent_fn: The target — same callable contract as ``evaluate_dataset``.
        capabilities: Capabilities the target actually has. Plugins attacking
            an absent capability still run, but a clean result is reported as
            ``DEFENDED_STRUCTURAL`` rather than counted as a defence.
        preconditions: Harness conditions verified as satisfied for this run.
            A plugin with unmet preconditions cannot report a trustworthy
            pass, so a clean result becomes ``UNVERIFIED``.
        judge_llm: Judge provider for plugins that grade with an LLM.
        sinks: Score sinks, applied per plugin.
        concurrency: Max concurrent attacks per plugin.

    Returns:
        A :class:`RedTeamReport`. Never raises on a security finding — inspect
        ``report.passed`` and gate on it.
    """
    resolved = list(plugins.resolve() if isinstance(plugins, AttackPack) else plugins)
    have = frozenset(capabilities or ())
    satisfied = frozenset(preconditions or ())

    results: list[PluginResult] = []
    for plugin in resolved:
        results.append(
            await _run_plugin(
                plugin,
                agent_fn,
                have=have,
                satisfied=satisfied,
                judge_llm=judge_llm,
                sinks=sinks,
                concurrency=concurrency,
            )
        )

    return RedTeamReport(name=name, results=results)


async def _run_plugin(
    plugin: AttackPlugin,
    agent_fn: Callable,
    *,
    have: frozenset[Capability],
    satisfied: frozenset[Precondition],
    judge_llm: BaseLLM | None,
    sinks: list[BaseSink] | None,
    concurrency: int | None,
) -> PluginResult:
    missing_caps = plugin.missing_capabilities(have)
    unmet_preconds = plugin.unmet_preconditions(satisfied)

    def build(verdict: Verdict, *, reason: str | None = None, **kwargs: object) -> PluginResult:
        return PluginResult(
            plugin_id=plugin.id,
            name=plugin.name,
            severity=plugin.severity,
            verdict=verdict,
            applicability=plugin.applicability(have),
            missing_capabilities=missing_caps,
            unmet_preconditions=unmet_preconds,
            owasp_agentic=plugin.owasp_agentic,
            owasp_llm=plugin.owasp_llm,
            atlas=plugin.atlas,
            requires_trace=plugin.requires_trace,
            reason=reason,
            **kwargs,  # type: ignore[arg-type]
        )

    goldens = plugin.tag_goldens(plugin.goldens())
    if not goldens:
        return build(Verdict.SKIPPED, reason="Plugin produced no attack cases")

    try:
        metrics = plugin.metrics(llm=judge_llm)
    except Exception as err:
        logger.warning("Plugin %s could not build metrics: %s", plugin.id, err)
        return build(Verdict.SKIPPED, reason=f"Could not build metrics: {err}")

    if not metrics:
        return build(Verdict.SKIPPED, reason="Plugin declared no metrics")

    scores = await evaluate_dataset(
        goldens,
        agent_fn,
        metrics=metrics,
        sinks=sinks,
        concurrency=concurrency,
    )

    outcomes = [_classify(case_scores) for case_scores in scores]
    breached = outcomes.count(_BREACH)
    errors = outcomes.count(_ERROR)
    unobservable = outcomes.count(_UNOBSERVABLE)

    # Precedence matters: an unmet precondition or absent capability only
    # undermines a *clean* result. A successful attack is a real finding no
    # matter how incomplete the surrounding coverage was.
    if breached:
        verdict = Verdict.BREACHED
        reason = None
    elif errors:
        verdict = Verdict.ERRORED
        reason = f"{errors} of {len(goldens)} attacks never reached the target"
    elif unobservable:
        verdict = Verdict.UNVERIFIED
        reason = (
            f"{unobservable} of {len(goldens)} attacks could not be graded because no tool "
            "trace was captured. Expose the agent's tool calls to the target adapter "
            "(e.g. tool_calls_path) before trusting this result."
        )
    elif unmet_preconds:
        verdict = Verdict.UNVERIFIED
        reason = (
            "All attacks were defended, but unmet preconditions mean they may "
            f"never have reached the target: {', '.join(sorted(p.value for p in unmet_preconds))}"
        )
    elif missing_caps:
        verdict = Verdict.DEFENDED_STRUCTURAL
        reason = (
            "All attacks were defended because the target has no "
            f"{', '.join(sorted(c.value for c in missing_caps))} — an architectural "
            "fact, not evidence of hardening"
        )
    else:
        verdict = Verdict.DEFENDED
        reason = None

    return build(
        verdict,
        reason=reason,
        tests=len(goldens),
        breached=breached,
        errors=errors,
        scores=scores,
    )


_BREACH = "breach"
_ERROR = "error"
_UNOBSERVABLE = "unobservable"
_CLEAN = "clean"


def _classify(case_scores: list[Score]) -> str:
    """Decide what one attack's scores actually demonstrate.

    A failing score is not automatically a breach. Metrics here fail closed,
    so they also report 0.0 when the target was unreachable or when no tool
    trace was captured — neither of which is a successful attack. Counting
    those as breaches would turn a dead server or missing instrumentation
    into a 100% attack success rate, which is worse than useless: it is a
    confident, wrong answer in the report people escalate on.

    Real breaches still win. If any metric failed for a reason other than
    those two, the attack landed.
    """
    failed = [score for score in case_scores if not score.passed]
    if not failed:
        return _CLEAN

    if all(_flagged(score, "target_error") for score in failed):
        return _ERROR
    if all(_flagged(score, "target_error") or _flagged(score, "trace_missing") for score in failed):
        return _UNOBSERVABLE
    return _BREACH


def _flagged(score: Score, key: str) -> bool:
    return bool(score.metadata and score.metadata.get(key))
