"""Tests for the red-team plugin scaffolding: ABC contract, packs, runner, report."""

from __future__ import annotations

import pytest

from harness_evals import plugins as plugin_registry
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.golden import Golden
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.errors import MissingAdapterError
from harness_evals.redteam import (
    OWASP_AGENTIC_TOP_10,
    Applicability,
    AttackPack,
    AttackPlugin,
    Capability,
    Precondition,
    RedTeamReport,
    Severity,
    Verdict,
    normalize_plugin_id,
    pack,
    run_redteam,
)
from harness_evals.redteam.report import PluginResult

# ----------------------------------------------------------------------
# Test doubles
# ----------------------------------------------------------------------


class _AlwaysFails(BaseMetric):
    """Grades every attack as a breach."""

    def __init__(self) -> None:
        super().__init__(name="always_fails", dimension=Dimension.SAFETY, threshold=1.0)

    def measure(self, eval_case: EvalCase) -> Score:
        return Score(name=self.name, value=0.0, threshold=self.threshold)


class _AlwaysPasses(BaseMetric):
    """Grades every attack as defended."""

    def __init__(self) -> None:
        super().__init__(name="always_passes", dimension=Dimension.SAFETY, threshold=1.0)

    def measure(self, eval_case: EvalCase) -> Score:
        return Score(name=self.name, value=1.0, threshold=self.threshold)


class _StubPlugin(AttackPlugin):
    id = "stub"
    name = "Stub Attack"
    severity = Severity.HIGH
    owasp_agentic = ("ASI01",)
    atlas = ("AML.T0054",)

    def __init__(self, *, metric: BaseMetric | None = None, n: int = 2) -> None:
        self._metric = metric or _AlwaysPasses()
        self._n = n

    def goldens(self) -> list[Golden]:
        return [Golden(input=f"attack {i}") for i in range(self._n)]

    def metrics(self, llm=None) -> list[BaseMetric]:
        return [self._metric]


class _NeedsDatabase(_StubPlugin):
    id = "needs_database"
    name = "Needs Database"
    requires_capabilities = frozenset({Capability.DATABASE})


class _NeedsSessionMemory(_StubPlugin):
    id = "needs_session_memory"
    name = "Needs Session Memory"
    preconditions = frozenset({Precondition.SESSION_MEMORY})


async def _agent(golden: Golden) -> EvalCase:
    return EvalCase.from_golden(golden, output="polite refusal")


# ----------------------------------------------------------------------
# Plugin ABC contract
# ----------------------------------------------------------------------


@pytest.mark.unit
def test_plugin_requires_id():
    with pytest.raises(TypeError, match="non-empty 'id'"):

        class _NoId(AttackPlugin):
            name = "No Id"

            def goldens(self):
                return []

            def metrics(self, llm=None):
                return []


@pytest.mark.unit
def test_plugin_requires_name():
    with pytest.raises(TypeError, match="non-empty 'name'"):

        class _NoName(AttackPlugin):
            id = "no_name"

            def goldens(self):
                return []

            def metrics(self, llm=None):
                return []


@pytest.mark.unit
def test_abstract_intermediate_base_is_exempt_from_metadata():
    """An abstract subclass may omit id/name — only concrete plugins need them."""

    class _Intermediate(AttackPlugin):
        def metrics(self, llm=None):
            return []

    assert _Intermediate.__abstractmethods__


@pytest.mark.unit
def test_tag_goldens_stamps_plugin_and_preserves_metadata():
    plugin = _StubPlugin()
    goldens = [Golden(input="a", metadata={"existing": 1})]

    tagged = plugin.tag_goldens(goldens)

    assert tagged[0].metadata["plugin"] == "stub"
    assert tagged[0].metadata["severity"] == "high"
    assert tagged[0].metadata["existing"] == 1


@pytest.mark.unit
def test_tag_goldens_does_not_override_explicit_severity():
    plugin = _StubPlugin()
    goldens = [Golden(input="a", metadata={"severity": "critical"})]

    assert plugin.tag_goldens(goldens)[0].metadata["severity"] == "critical"


@pytest.mark.unit
def test_applicability_reflects_capabilities():
    plugin = _NeedsDatabase()

    assert plugin.applicability(frozenset()) is Applicability.STRUCTURAL
    assert plugin.applicability(frozenset({Capability.DATABASE})) is Applicability.APPLICABLE
    assert plugin.missing_capabilities(frozenset()) == frozenset({Capability.DATABASE})


@pytest.mark.unit
def test_unmet_preconditions():
    plugin = _NeedsSessionMemory()

    assert plugin.unmet_preconditions(frozenset()) == frozenset({Precondition.SESSION_MEMORY})
    assert plugin.unmet_preconditions(frozenset({Precondition.SESSION_MEMORY})) == frozenset()


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------


@pytest.mark.unit
def test_register_and_look_up_attack_plugin():
    plugin_registry.register_attack_plugin("stub")(_StubPlugin)

    assert plugin_registry.attack_plugin("stub") is _StubPlugin
    assert "stub" in plugin_registry.registered_attack_plugins()


@pytest.mark.unit
def test_unknown_attack_plugin_raises():
    with pytest.raises(MissingAdapterError):
        plugin_registry.attack_plugin("definitely_not_registered")


@pytest.mark.unit
def test_attack_plugins_is_a_known_family():
    assert plugin_registry.ATTACK_PLUGINS in plugin_registry.FAMILIES


# ----------------------------------------------------------------------
# Packs
# ----------------------------------------------------------------------


@pytest.mark.unit
def test_owasp_pack_covers_every_scanned_plugin():
    """The pack must cover all 21 scanned plugins plus our tool-call addition."""
    assert len(OWASP_AGENTIC_TOP_10.plugin_ids) == 22
    assert len(set(OWASP_AGENTIC_TOP_10.plugin_ids)) == 22
    assert "unauthorized_state_change" in OWASP_AGENTIC_TOP_10.plugin_ids


@pytest.mark.unit
def test_promptfoo_aliases_resolve_into_the_pack():
    """Every alias target except the skipped intent plugin is in the pack."""
    from harness_evals.redteam.packs import PROMPTFOO_ALIASES

    targets = set(PROMPTFOO_ALIASES.values()) - {"custom_intent"}
    assert targets <= set(OWASP_AGENTIC_TOP_10.plugin_ids)


@pytest.mark.unit
def test_normalize_plugin_id():
    assert normalize_plugin_id("harmful:misinformation-disinformation") == "disinformation"
    assert normalize_plugin_id("hijacking") == "resource_hijacking"
    assert normalize_plugin_id("disinformation") == "disinformation"


@pytest.mark.unit
def test_pack_lookup_and_unknown_pack():
    assert pack("owasp_agentic_top_10") is OWASP_AGENTIC_TOP_10
    with pytest.raises(KeyError, match="Unknown attack pack"):
        pack("nope")


@pytest.mark.unit
def test_pack_reports_unimplemented_plugins():
    custom = AttackPack(id="p", name="P", description="", plugin_ids=("stub", "missing"))
    plugin_registry.register_attack_plugin("stub")(_StubPlugin)

    assert custom.unimplemented() == ("missing",)
    assert [type(p) for p in custom.resolve_available()] == [_StubPlugin]
    with pytest.raises(MissingAdapterError):
        custom.resolve()


# ----------------------------------------------------------------------
# Runner — verdict precedence
# ----------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_breach_is_reported_when_any_metric_fails():
    report = await run_redteam("run", [_StubPlugin(metric=_AlwaysFails(), n=3)], _agent)

    result = report.results[0]
    assert result.verdict is Verdict.BREACHED
    assert result.tests == 3
    assert result.breached == 3
    assert result.attack_success_rate == 1.0
    assert not report.passed


@pytest.mark.unit
@pytest.mark.asyncio
async def test_clean_run_with_capabilities_present_is_a_real_defence():
    report = await run_redteam(
        "run",
        [_NeedsDatabase()],
        _agent,
        capabilities=[Capability.DATABASE],
    )

    assert report.results[0].verdict is Verdict.DEFENDED
    assert report.passed


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_capability_downgrades_pass_to_structural():
    report = await run_redteam("run", [_NeedsDatabase()], _agent, capabilities=[])

    result = report.results[0]
    assert result.verdict is Verdict.DEFENDED_STRUCTURAL
    assert result.missing_capabilities == frozenset({Capability.DATABASE})
    assert "architectural" in result.reason
    # A structural pass must not be counted as a security conclusion.
    assert not result.conclusive


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unmet_precondition_downgrades_pass_to_unverified():
    report = await run_redteam("run", [_NeedsSessionMemory()], _agent, preconditions=[])

    result = report.results[0]
    assert result.verdict is Verdict.UNVERIFIED
    assert result.unmet_preconditions == frozenset({Precondition.SESSION_MEMORY})
    assert not result.conclusive


@pytest.mark.unit
@pytest.mark.asyncio
async def test_breach_outranks_unmet_precondition():
    """A successful attack is a real finding even when coverage was incomplete."""
    plugin = _NeedsSessionMemory(metric=_AlwaysFails())

    report = await run_redteam("run", [plugin], _agent, preconditions=[])

    assert report.results[0].verdict is Verdict.BREACHED


@pytest.mark.unit
@pytest.mark.asyncio
async def test_breach_outranks_missing_capability():
    plugin = _NeedsDatabase(metric=_AlwaysFails())

    report = await run_redteam("run", [plugin], _agent, capabilities=[])

    assert report.results[0].verdict is Verdict.BREACHED


@pytest.mark.unit
@pytest.mark.asyncio
async def test_plugin_with_no_goldens_is_skipped():
    report = await run_redteam("run", [_StubPlugin(n=0)], _agent)

    result = report.results[0]
    assert result.verdict is Verdict.SKIPPED
    assert result.tests == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_plugin_needing_a_judge_is_skipped_not_silently_degraded():
    class _NeedsJudge(_StubPlugin):
        id = "needs_judge"
        name = "Needs Judge"

        def metrics(self, llm=None):
            if llm is None:
                raise ValueError("requires a judge_llm")
            return [_AlwaysPasses()]

    report = await run_redteam("run", [_NeedsJudge()], _agent)

    result = report.results[0]
    assert result.verdict is Verdict.SKIPPED
    assert "requires a judge_llm" in result.reason


@pytest.mark.unit
@pytest.mark.asyncio
async def test_runner_accepts_a_pack():
    plugin_registry.register_attack_plugin("stub")(_StubPlugin)
    custom = AttackPack(id="p", name="P", description="", plugin_ids=("stub",))

    report = await run_redteam("run", custom, _agent)

    assert [r.plugin_id for r in report.results] == ["stub"]


# ----------------------------------------------------------------------
# Report aggregation
# ----------------------------------------------------------------------


def _result(plugin_id: str, verdict: Verdict, *, tests: int, breached: int, severity: Severity) -> PluginResult:
    return PluginResult(
        plugin_id=plugin_id,
        name=plugin_id,
        severity=severity,
        verdict=verdict,
        tests=tests,
        breached=breached,
    )


@pytest.mark.unit
def test_informative_asr_excludes_inconclusive_results():
    """Reproduces the scan-#2 shape: vacuous passes must not dilute the ASR."""
    report = RedTeamReport(
        name="scan",
        results=[
            _result("hijack", Verdict.BREACHED, tests=3, breached=3, severity=Severity.HIGH),
            _result("sqli", Verdict.DEFENDED_STRUCTURAL, tests=3, breached=0, severity=Severity.HIGH),
            _result("shell", Verdict.DEFENDED_STRUCTURAL, tests=3, breached=0, severity=Severity.HIGH),
            _result("memory", Verdict.UNVERIFIED, tests=3, breached=0, severity=Severity.HIGH),
        ],
    )

    # Headline ASR dilutes the finding across attacks that proved nothing.
    assert report.attack_success_rate == pytest.approx(3 / 12)
    # Informative ASR counts only the one plugin that actually measured something.
    assert report.informative_attack_success_rate == pytest.approx(1.0)


@pytest.mark.unit
def test_breached_plugins_sorted_by_severity_then_asr():
    report = RedTeamReport(
        name="scan",
        results=[
            _result("low", Verdict.BREACHED, tests=3, breached=3, severity=Severity.LOW),
            _result("high_partial", Verdict.BREACHED, tests=3, breached=1, severity=Severity.HIGH),
            _result("high_full", Verdict.BREACHED, tests=3, breached=3, severity=Severity.HIGH),
            _result("critical", Verdict.BREACHED, tests=3, breached=1, severity=Severity.CRITICAL),
        ],
    )

    assert [r.plugin_id for r in report.breached_plugins] == [
        "critical",
        "high_full",
        "high_partial",
        "low",
    ]


@pytest.mark.unit
def test_report_passes_only_with_zero_breaches():
    clean = RedTeamReport(
        name="scan",
        results=[_result("a", Verdict.DEFENDED, tests=3, breached=0, severity=Severity.HIGH)],
    )
    assert clean.passed

    clean.results.append(_result("b", Verdict.BREACHED, tests=3, breached=1, severity=Severity.LOW))
    assert not clean.passed, "a single low-severity breach must still fail the run"


@pytest.mark.unit
def test_count_by_severity_counts_only_breaches():
    report = RedTeamReport(
        name="scan",
        results=[
            _result("a", Verdict.BREACHED, tests=3, breached=3, severity=Severity.HIGH),
            _result("b", Verdict.BREACHED, tests=3, breached=1, severity=Severity.HIGH),
            _result("c", Verdict.DEFENDED, tests=3, breached=0, severity=Severity.CRITICAL),
        ],
    )

    counts = report.count_by_severity()
    assert counts[Severity.HIGH] == 2
    assert counts[Severity.CRITICAL] == 0


@pytest.mark.unit
def test_empty_report_has_zero_asr_not_a_division_error():
    report = RedTeamReport(name="scan")

    assert report.attack_success_rate == 0.0
    assert report.informative_attack_success_rate == 0.0
    assert report.passed


@pytest.mark.unit
def test_summary_labels_structural_and_unverified_distinctly():
    report = RedTeamReport(
        name="scan",
        results=[
            PluginResult(
                plugin_id="sqli",
                name="SQL Injection",
                severity=Severity.HIGH,
                verdict=Verdict.DEFENDED_STRUCTURAL,
                tests=3,
                missing_capabilities=frozenset({Capability.DATABASE}),
            ),
            PluginResult(
                plugin_id="memory",
                name="Memory Poisoning",
                severity=Severity.HIGH,
                verdict=Verdict.UNVERIFIED,
                tests=3,
                unmet_preconditions=frozenset({Precondition.SESSION_MEMORY}),
            ),
        ],
    )

    summary = report.summary()
    assert "not applicable" in summary
    assert "database" in summary
    assert "UNVERIFIED" in summary
    assert "session_memory" in summary


@pytest.mark.unit
def test_report_to_dict_is_json_friendly():
    report = RedTeamReport(
        name="scan",
        results=[
            PluginResult(
                plugin_id="sqli",
                name="SQL Injection",
                severity=Severity.HIGH,
                verdict=Verdict.DEFENDED_STRUCTURAL,
                tests=3,
                missing_capabilities=frozenset({Capability.DATABASE}),
            )
        ],
    )

    data = report.to_dict()
    assert data["results"][0]["verdict"] == "defended_structural"
    assert data["results"][0]["missing_capabilities"] == ["database"]
    assert data["results"][0]["severity"] == "high"
