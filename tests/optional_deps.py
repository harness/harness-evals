"""Skip helpers for tests that exercise an optional dependency.

``harness-evals`` keeps ``httpx``, ``pyjwt``, ``langfuse``, the OpenTelemetry
packages, ``openai``, ``anthropic``, and ``nltk`` optional — see
``[tool.poetry.extras]`` in ``pyproject.toml``. ``src/`` respects that: every one
of those imports is guarded, and the code raises a directed ``ImportError``
("... Install them with: pip install harness-evals[otlp]") when a caller reaches
a path it cannot serve.

The tests did not respect it. A plain ``pip install -e .`` followed by ``pytest``
aborted during collection, because two test modules imported ``httpx`` at module
scope, and 80-odd further tests failed once collection was forced through. That
makes the suite unrunnable for a contributor who installed only the base package,
which for an open-source SDK is a defect in its own right.

Tests that need an extra now declare it. Use ``requires()`` for a test or class,
and ``pytest.importorskip`` at module scope when the module itself cannot be
imported without the dependency.

**These skips must never fire in CI.** ``.github/workflows/ci.yml`` installs
``--all-extras``, so a skip reported by a CI run means the install step regressed
and coverage for the affected modules is being silently under-counted.
"""

from __future__ import annotations

import importlib.util

import pytest


def missing_module(*modules: str) -> str | None:
    """Return the first of ``modules`` that cannot be imported, else ``None``.

    ``find_spec`` is used rather than a real import so that probing stays free of
    side effects and does not populate ``sys.modules`` for a package the test may
    want to patch.
    """
    for module in modules:
        # A dotted name whose *parent* is absent raises rather than returning
        # None, which matters for probes like "opentelemetry.sdk".
        try:
            found = importlib.util.find_spec(module) is not None
        except ModuleNotFoundError:
            found = False
        if not found:
            return module
    return None


def requires(*modules: str, extra: str) -> pytest.MarkDecorator:
    """Skip the decorated test or class unless every module in ``modules`` imports.

    ``extra`` names the ``pyproject.toml`` extra that provides them, so the skip
    reason tells the reader exactly which install command fixes it.
    """
    absent = missing_module(*modules)
    return pytest.mark.skipif(
        absent is not None,
        reason=f"needs optional dependency {absent!r} — pip install 'harness-evals[{extra}]'",
    )
