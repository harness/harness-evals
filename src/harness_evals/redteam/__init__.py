"""Red-team security testing — adversarial datasets plus the metrics that grade them.

Distinct from ``harness_evals.metrics.security``, which grades AI-generated
vulnerability *remediations*. This package attacks a target and reports whether
it held.
"""

from harness_evals.redteam.packs import (
    OWASP_AGENTIC_TOP_10,
    PROMPTFOO_ALIASES,
    AttackPack,
    normalize_plugin_id,
    pack,
    packs,
    register_pack,
)
from harness_evals.redteam.plugin import (
    Applicability,
    AttackPlugin,
    Capability,
    Precondition,
    Severity,
)
from harness_evals.redteam.report import PluginResult, RedTeamReport, Verdict
from harness_evals.redteam.runner import run_redteam

# Imported for its registration side effects — must come last so the modules
# above are fully initialised before plugins import from them.
from harness_evals.redteam import plugins as _plugins  # noqa: E402,F401  isort:skip

__all__ = [
    "OWASP_AGENTIC_TOP_10",
    "PROMPTFOO_ALIASES",
    "Applicability",
    "AttackPack",
    "AttackPlugin",
    "Capability",
    "PluginResult",
    "Precondition",
    "RedTeamReport",
    "Severity",
    "Verdict",
    "normalize_plugin_id",
    "pack",
    "packs",
    "register_pack",
    "run_redteam",
]
