"""Prompt template data model."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_PLACEHOLDER_RE = re.compile(r"(?<!\\)\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_TEMPLATE_EXPR_RE = re.compile(r"(?<!\\)\{\{\s*(.*?)\s*\}\}")
_ESCAPED_OPEN = r"\{\{"


@dataclass
class PromptTemplate:
    """A text prompt with simple ``{{var}}`` placeholders."""

    template: str
    input_variables: list[str] = field(default_factory=lambda: ["input"])
    model_hint: dict[str, Any] | None = None
    version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_placeholder_syntax(self.template)
        placeholders = set(extract_template_variables(self.template))
        declared = set(self.input_variables)
        missing = sorted(placeholders - declared)
        if missing:
            names = ", ".join(missing)
            raise ValueError(f"input_variables is missing template placeholder(s): {names}")

    def render(self, **kwargs: Any) -> str:
        """Render the template by replacing each placeholder with ``str(value)``."""

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in kwargs:
                raise KeyError(f"Missing prompt variable {name!r}")
            return str(kwargs[name])

        rendered = _PLACEHOLDER_RE.sub(replace, self.template)
        return rendered.replace(_ESCAPED_OPEN, "{{")


def extract_template_variables(template: str) -> list[str]:
    """Return placeholder names in first-seen order."""

    variables: list[str] = []
    seen: set[str] = set()
    for match in _PLACEHOLDER_RE.finditer(template):
        name = match.group(1)
        if name not in seen:
            seen.add(name)
            variables.append(name)
    return variables


def _validate_placeholder_syntax(template: str) -> None:
    for match in _TEMPLATE_EXPR_RE.finditer(template):
        if _PLACEHOLDER_RE.fullmatch(match.group(0)) is None:
            expression = match.group(1).strip()
            raise ValueError(
                f"Unsupported prompt placeholder syntax {expression!r}; use flat '{{{{var}}}}' placeholders only"
            )


def infer_input_variables(template: str) -> list[str]:
    """Infer source-loaded prompt variables from template placeholders.

    Returns [] for static prompts with no placeholders — distinct from the
    code-first PromptTemplate default of ["input"].
    """

    return extract_template_variables(template)
