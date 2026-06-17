"""Guard foundation modules against optional vendor imports."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

FOUNDATION_MODULES = [
    Path("src/harness_evals/errors.py"),
    Path("src/harness_evals/refs.py"),
    Path("src/harness_evals/plugins.py"),
]
FORBIDDEN_IMPORTS = {"langfuse", "httpx", "harness_ai", "opentelemetry", "openai", "anthropic"}


@pytest.mark.unit
@pytest.mark.parametrize("module_path", FOUNDATION_MODULES)
def test_foundation_modules_do_not_import_optional_vendors(module_path: Path) -> None:
    tree = ast.parse(module_path.read_text(), filename=str(module_path))

    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module.split(".", maxsplit=1)[0])

    assert imported_modules.isdisjoint(FORBIDDEN_IMPORTS)
