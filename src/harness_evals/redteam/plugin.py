"""AttackPlugin — the red-team analogue of a metric-plus-dataset pair.

A red-team plugin bundles two things our eval primitives keep separate:

- an adversarial **dataset** (the attacks to send), exposed via :meth:`AttackPlugin.goldens`
- the **metrics** that decide whether an attack succeeded, via :meth:`AttackPlugin.metrics`

Plugins additionally declare *why* a result can be trusted. Two declarations
drive that, and both exist because black-box scans routinely report passes
that measured nothing:

``requires_capabilities``
    Capabilities the target must actually have for the attack to be
    meaningful. A SQL-injection attack against an agent with no database
    cannot fail, so its clean result is an architectural fact rather than a
    security property. Missing capabilities downgrade a pass to
    ``Verdict.DEFENDED_STRUCTURAL``.

``preconditions``
    Runtime conditions the *harness* must satisfy for the attack to reach its
    target at all — persistent session memory for a memory-poisoning attack,
    for instance. Unmet preconditions downgrade a pass to
    ``Verdict.UNVERIFIED`` instead of silently counting it as a defence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from harness_evals.core.golden import Golden
from harness_evals.core.metric import BaseMetric
from harness_evals.llm.base import BaseLLM


class Severity(str, Enum):
    """Impact rating for a successful attack, ordered low to critical."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Applicability(str, Enum):
    """How meaningful a plugin's result is for the target as it exists today.

    APPLICABLE: the attack surface exists; results are trustworthy.
    STRUCTURAL: the surface is absent, so a pass proves nothing.
    FUTURE: only becomes meaningful once the target gains new capabilities.
    """

    APPLICABLE = "applicable"
    STRUCTURAL = "structural"
    FUTURE = "future"


class Capability(str, Enum):
    """Target capabilities that attacks can depend on.

    Declared per plugin via ``requires_capabilities`` and supplied per run via
    ``run_redteam(capabilities=...)``. A plugin whose required capabilities are
    absent still runs (so parity with external scanners is preserved), but its
    verdict is labelled structural.
    """

    DATABASE = "database"
    SHELL = "shell"
    OUTBOUND_HTTP = "outbound_http"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    MCP = "mcp"
    STATE_CHANGING_TOOLS = "state_changing_tools"
    RETRIEVAL = "retrieval"
    USER_RECORDS = "user_records"


class Precondition(str, Enum):
    """Harness-side conditions required for an attack to be exercised at all.

    These encode the failure modes that make an external scan's green results
    meaningless: an attack that never reached the agent looks identical to an
    attack the agent repelled.
    """

    SESSION_MEMORY = "session_memory"
    """Conversation history must persist across turns within a session."""

    CLIENT_SET_SESSION_ID = "client_set_session_id"
    """The harness must be able to choose the session identifier per case."""

    TOOL_TRACE = "tool_trace"
    """Tool calls and arguments must be observable (OTel spans or in-process)."""

    POISONED_FIXTURES = "poisoned_fixtures"
    """The target must serve attacker-controlled retrieval content."""

    MANAGED_PROMPT = "managed_prompt"
    """The target must be running its production prompt, not a fallback."""


class AttackPlugin(ABC):
    """Base class for a red-team attack plugin.

    Subclasses set the metadata attributes and implement :meth:`goldens` and
    :meth:`metrics`. Keep one plugin per module, mirroring the layout used for
    metrics.
    """

    id: str
    """Stable identifier, e.g. ``"system_prompt_override"``. Used in configs."""

    name: str
    """Human-readable name for reports."""

    severity: Severity = Severity.MEDIUM

    owasp_agentic: tuple[str, ...] = ()
    """OWASP Top 10 for Agentic Applications ids, e.g. ``("ASI05",)``."""

    owasp_llm: tuple[str, ...] = ()
    """OWASP Top 10 for LLM Applications ids, e.g. ``("LLM01",)``."""

    atlas: tuple[str, ...] = ()
    """MITRE ATLAS technique ids, e.g. ``("AML.T0054",)``."""

    requires_capabilities: frozenset[Capability] = frozenset()

    preconditions: frozenset[Precondition] = frozenset()

    requires_trace: bool = False
    """True when grading needs ``EvalCase.tool_calls``, not just ``output``.

    Kept separate from ``preconditions`` so callers can filter for
    trajectory-level plugins without inspecting precondition sets.
    """

    description: str = ""

    _REQUIRED_METHODS = ("goldens", "metrics")

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Intermediate abstract bases legitimately have no id/name. ABCMeta has
        # not populated __abstractmethods__ yet at this point, so inspect the
        # methods themselves rather than relying on it.
        still_abstract = any(
            getattr(getattr(cls, method, None), "__isabstractmethod__", False) for method in cls._REQUIRED_METHODS
        )
        if still_abstract:
            return
        if not getattr(cls, "id", None):
            raise TypeError(f"{cls.__name__} must define a non-empty 'id'")
        if not getattr(cls, "name", None):
            raise TypeError(f"{cls.__name__} must define a non-empty 'name'")

    @abstractmethod
    def goldens(self) -> list[Golden]:
        """Return the adversarial cases this plugin sends to the target.

        Each Golden should carry ``metadata["plugin"] = self.id`` so results
        can be attributed back after a mixed-plugin run; :meth:`tag_goldens`
        does this for you.
        """
        ...

    @abstractmethod
    def metrics(self, llm: BaseLLM | None = None) -> list[BaseMetric]:
        """Return the metrics that grade this plugin's attacks.

        *llm* is the judge provider, supplied only when the run configures one.
        Plugins graded deterministically should ignore it; plugins that need a
        judge must raise if it is ``None`` rather than silently degrading.
        """
        ...

    def tag_goldens(self, goldens: list[Golden]) -> list[Golden]:
        """Stamp plugin attribution onto goldens, preserving existing metadata."""
        for golden in goldens:
            metadata = dict(golden.metadata or {})
            metadata["plugin"] = self.id
            metadata.setdefault("severity", self.severity.value)
            golden.metadata = metadata
        return goldens

    def applicability(self, capabilities: frozenset[Capability]) -> Applicability:
        """Classify how meaningful this plugin's result is for the given target."""
        missing = self.requires_capabilities - capabilities
        if not missing:
            return Applicability.APPLICABLE
        return Applicability.STRUCTURAL

    def missing_capabilities(self, capabilities: frozenset[Capability]) -> frozenset[Capability]:
        """Return required capabilities the target does not have."""
        return self.requires_capabilities - capabilities

    def unmet_preconditions(self, satisfied: frozenset[Precondition]) -> frozenset[Precondition]:
        """Return preconditions the harness did not satisfy for this run."""
        return self.preconditions - satisfied
