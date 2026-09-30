"""Named bundles of attack plugins.

A pack is to red-teaming what a metric set is to evals: a named, versionable
list of checks applied together. Packs hold plugin *ids* rather than classes so
a pack can be declared in YAML and resolved lazily against the registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harness_evals import plugins as plugin_registry
from harness_evals.redteam.plugin import AttackPlugin

# Promptfoo plugin id -> harness-evals attack plugin id. Lets teams migrating
# from an external scan reuse their existing plugin lists verbatim.
PROMPTFOO_ALIASES: dict[str, str] = {
    "agentic:memory-poisoning": "memory_poisoning",
    "bfla": "function_authorization_bypass",
    "bola": "object_authorization_bypass",
    "cross-session-leak": "cross_session_leak",
    "divergent-repetition": "divergent_repetition",
    "excessive-agency": "excessive_agency",
    "goal-misalignment": "goal_misalignment",
    "hallucination": "hallucination",
    "harmful:cybercrime:malicious-code": "malicious_code",
    "harmful:misinformation-disinformation": "disinformation",
    "hijacking": "resource_hijacking",
    "imitation": "entity_impersonation",
    "indirect-prompt-injection": "indirect_prompt_injection",
    "intent": "custom_intent",
    "mcp": "mcp_abuse",
    "overreliance": "overreliance",
    "rbac": "rbac_bypass",
    "shell-injection": "shell_injection",
    "sql-injection": "sql_injection",
    "ssrf": "ssrf",
    "system-prompt-override": "system_prompt_override",
    "tool-discovery": "tool_discovery",
}


def normalize_plugin_id(plugin_id: str) -> str:
    """Resolve a promptfoo-style plugin id to its harness-evals equivalent."""
    return PROMPTFOO_ALIASES.get(plugin_id, plugin_id)


@dataclass(frozen=True)
class AttackPack:
    """A named list of attack plugin ids."""

    id: str
    name: str
    description: str
    plugin_ids: tuple[str, ...] = field(default_factory=tuple)

    def resolve(self) -> list[AttackPlugin]:
        """Instantiate every plugin in this pack.

        Raises :exc:`~harness_evals.errors.MissingAdapterError` for any id that
        is not registered. Call :meth:`unimplemented` first to check.
        """
        return [plugin_registry.attack_plugin(pid)() for pid in self.plugin_ids]

    def resolve_available(self) -> list[AttackPlugin]:
        """Instantiate the plugins in this pack that are currently registered."""
        registered = plugin_registry.registered_attack_plugins()
        return [registered[pid]() for pid in self.plugin_ids if pid in registered]

    def unimplemented(self) -> tuple[str, ...]:
        """Return pack plugin ids with no registered implementation yet."""
        registered = plugin_registry.registered_attack_plugins()
        return tuple(pid for pid in self.plugin_ids if pid not in registered)


OWASP_AGENTIC_TOP_10 = AttackPack(
    id="owasp_agentic_top_10",
    name="OWASP Top 10 for Agentic Applications",
    description=(
        "Agentic threat coverage: instruction hijacking, purpose deviation, "
        "tool misuse, memory and session integrity, and unauthorized state change. "
        "Mirrors the promptfoo collection of the same name, plus "
        "'unauthorized_state_change' for tool-argument violations that "
        "response-text grading cannot see."
    ),
    plugin_ids=(
        # Instruction and purpose integrity
        "system_prompt_override",
        "resource_hijacking",
        "goal_misalignment",
        "excessive_agency",
        # Tool and action integrity
        "unauthorized_state_change",
        "tool_discovery",
        "mcp_abuse",
        # Grounding and truthfulness
        "disinformation",
        "hallucination",
        "overreliance",
        # Memory and session integrity
        "memory_poisoning",
        "cross_session_leak",
        # Injection
        "indirect_prompt_injection",
        # Authorization
        "object_authorization_bypass",
        "function_authorization_bypass",
        "rbac_bypass",
        # Technical injection surfaces
        "sql_injection",
        "shell_injection",
        "ssrf",
        # Content and model-level
        "malicious_code",
        "entity_impersonation",
        "divergent_repetition",
    ),
)


_PACKS: dict[str, AttackPack] = {
    OWASP_AGENTIC_TOP_10.id: OWASP_AGENTIC_TOP_10,
}


def pack(pack_id: str) -> AttackPack:
    """Return a pack by id."""
    if pack_id not in _PACKS:
        known = ", ".join(sorted(_PACKS))
        raise KeyError(f"Unknown attack pack {pack_id!r}. Known packs: {known}")
    return _PACKS[pack_id]


def packs() -> list[AttackPack]:
    """Return all registered packs."""
    return list(_PACKS.values())


def register_pack(new_pack: AttackPack) -> AttackPack:
    """Register a custom pack, making it addressable by id in configs."""
    _PACKS[new_pack.id] = new_pack
    return new_pack
