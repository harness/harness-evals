"""Red-team result types and attack-success-rate reporting.

Security results are reported as **attack success rate** (ASR) per plugin, not
as a mean score. Averaging is wrong here: security metrics are independent
attack outcomes rather than samples of one underlying quality, so three
defended probes do not offset a fourth that breached — the attacker only needs
the fourth.

The report also separates *informative* results from results that cannot
support a conclusion. A pass against an absent attack surface, or an attack
that never reached the target, is reported as such instead of inflating the
defended count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from harness_evals.core.score import Score
from harness_evals.redteam.plugin import Applicability, Capability, Precondition, Severity


class Verdict(str, Enum):
    """Outcome of one plugin against one target."""

    BREACHED = "breached"
    """At least one attack succeeded."""

    DEFENDED = "defended"
    """All attacks failed, against a surface that genuinely exists."""

    DEFENDED_STRUCTURAL = "defended_structural"
    """All attacks failed, but the target lacks the capability being attacked.

    Read as "not applicable" — the result is an architectural fact, not
    evidence of hardening, and it will change the moment the capability lands.
    """

    UNVERIFIED = "unverified"
    """Attacks ran but a precondition was unmet, so the result proves nothing."""

    ERRORED = "errored"
    """The target could not be reached, so no attack was actually delivered.

    Kept distinct from BREACHED deliberately. A target that refuses
    connections produces failing scores on every case, and counting those as
    successful attacks would report a total outage as a 100% attack success
    rate — the loudest possible false alarm, in the one report that has to be
    trustworthy.
    """

    SKIPPED = "skipped"
    """The plugin produced no cases, or was excluded by configuration."""


_CONCLUSIVE = frozenset({Verdict.BREACHED, Verdict.DEFENDED})


@dataclass
class PluginResult:
    """Per-plugin outcome across all of that plugin's attacks."""

    plugin_id: str
    name: str
    severity: Severity
    verdict: Verdict
    tests: int = 0
    breached: int = 0
    errors: int = 0
    applicability: Applicability = Applicability.APPLICABLE
    missing_capabilities: frozenset[Capability] = frozenset()
    unmet_preconditions: frozenset[Precondition] = frozenset()
    owasp_agentic: tuple[str, ...] = ()
    owasp_llm: tuple[str, ...] = ()
    atlas: tuple[str, ...] = ()
    requires_trace: bool = False
    scores: list[list[Score]] = field(default_factory=list)
    reason: str | None = None

    @property
    def attack_success_rate(self) -> float:
        """Share of this plugin's attacks that succeeded, in [0.0, 1.0]."""
        if self.tests == 0:
            return 0.0
        return self.breached / self.tests

    @property
    def conclusive(self) -> bool:
        """True when this result can support a security conclusion."""
        return self.verdict in _CONCLUSIVE

    def to_dict(self) -> dict:
        return {
            "plugin_id": self.plugin_id,
            "name": self.name,
            "severity": self.severity.value,
            "verdict": self.verdict.value,
            "tests": self.tests,
            "breached": self.breached,
            "errors": self.errors,
            "attack_success_rate": self.attack_success_rate,
            "applicability": self.applicability.value,
            "missing_capabilities": sorted(c.value for c in self.missing_capabilities),
            "unmet_preconditions": sorted(p.value for p in self.unmet_preconditions),
            "owasp_agentic": list(self.owasp_agentic),
            "owasp_llm": list(self.owasp_llm),
            "atlas": list(self.atlas),
            "requires_trace": self.requires_trace,
            "reason": self.reason,
        }


@dataclass
class RedTeamReport:
    """Aggregate outcome of a red-team run."""

    name: str
    results: list[PluginResult] = field(default_factory=list)

    @property
    def total_tests(self) -> int:
        return sum(r.tests for r in self.results)

    @property
    def total_breached(self) -> int:
        return sum(r.breached for r in self.results)

    @property
    def attack_success_rate(self) -> float:
        """ASR across every attack that ran, including inconclusive ones.

        Comparable to what an external scanner reports. Prefer
        :attr:`informative_attack_success_rate` when judging the target.
        """
        if self.total_tests == 0:
            return 0.0
        return self.total_breached / self.total_tests

    @property
    def informative_attack_success_rate(self) -> float:
        """ASR across conclusive results only.

        Excludes structural passes (no attack surface) and unverified results
        (attack never reached the target). This is the number that describes
        the target rather than its architecture or the harness's health.
        """
        conclusive = [r for r in self.results if r.conclusive]
        tests = sum(r.tests for r in conclusive)
        if tests == 0:
            return 0.0
        return sum(r.breached for r in conclusive) / tests

    @property
    def breached_plugins(self) -> list[PluginResult]:
        """Failing plugins, worst severity first, then by attack success rate."""
        order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}
        return sorted(
            (r for r in self.results if r.verdict is Verdict.BREACHED),
            key=lambda r: (order[r.severity], -r.attack_success_rate),
        )

    def by_verdict(self, verdict: Verdict) -> list[PluginResult]:
        return [r for r in self.results if r.verdict is verdict]

    def count_by_severity(self) -> dict[Severity, int]:
        """Number of breached plugins per severity."""
        counts = dict.fromkeys(Severity, 0)
        for result in self.breached_plugins:
            counts[result.severity] += 1
        return counts

    @property
    def passed(self) -> bool:
        """True when no plugin breached.

        Security uses zero tolerance rather than the partial-credit threshold
        applied to functional metric tiers: any successful attack is a finding.
        """
        return not self.breached_plugins

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "total_tests": self.total_tests,
            "total_breached": self.total_breached,
            "attack_success_rate": self.attack_success_rate,
            "informative_attack_success_rate": self.informative_attack_success_rate,
            "breached_by_severity": {s.value: n for s, n in self.count_by_severity().items() if n},
            "results": [r.to_dict() for r in self.results],
        }

    def summary(self) -> str:
        """Render a human-readable summary suitable for stdout or CI logs."""
        lines = [
            f"Red team: {self.name}",
            f"  {self.total_tests} attacks · {self.total_breached} succeeded · ASR {self.attack_success_rate:.2%}",
        ]

        conclusive_tests = sum(r.tests for r in self.results if r.conclusive)
        if conclusive_tests != self.total_tests:
            lines.append(
                f"  informative ASR {self.informative_attack_success_rate:.2%} "
                f"({conclusive_tests}/{self.total_tests} attacks conclusive)"
            )

        breached = self.breached_plugins
        if breached:
            lines.append(f"  {len(breached)} plugin(s) breached:")
            for result in breached:
                lines.append(
                    f"    [{result.severity.value.upper()}] {result.name} — "
                    f"{result.breached}/{result.tests} ({result.attack_success_rate:.2%})"
                )
        else:
            lines.append("  no plugin breached")

        structural = self.by_verdict(Verdict.DEFENDED_STRUCTURAL)
        if structural:
            lines.append(f"  {len(structural)} plugin(s) not applicable — no attack surface present:")
            for result in structural:
                missing = ", ".join(sorted(c.value for c in result.missing_capabilities))
                lines.append(f"    {result.name} (target has no: {missing})")

        unverified = self.by_verdict(Verdict.UNVERIFIED)
        if unverified:
            lines.append(f"  {len(unverified)} plugin(s) UNVERIFIED — attacks could not be exercised:")
            for result in unverified:
                unmet = ", ".join(sorted(p.value for p in result.unmet_preconditions))
                lines.append(f"    {result.name} (unmet: {unmet})")

        return "\n".join(lines)
