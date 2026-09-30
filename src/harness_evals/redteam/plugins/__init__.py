"""Attack plugin implementations.

Importing this package registers every built-in plugin, so pack ids resolve
without callers needing to know which module defines what.
"""

from harness_evals.redteam.plugins.shell_injection import ShellInjectionPlugin
from harness_evals.redteam.plugins.sql_injection import SqlInjectionPlugin
from harness_evals.redteam.plugins.ssrf import SsrfPlugin
from harness_evals.redteam.plugins.system_prompt_override import SystemPromptOverridePlugin
from harness_evals.redteam.plugins.tool_discovery import ToolDiscoveryPlugin
from harness_evals.redteam.plugins.unauthorized_state_change import UnauthorizedStateChangePlugin

__all__ = [
    "ShellInjectionPlugin",
    "SqlInjectionPlugin",
    "SsrfPlugin",
    "SystemPromptOverridePlugin",
    "ToolDiscoveryPlugin",
    "UnauthorizedStateChangePlugin",
]
