"""Backwards-compatibility aliases for the ``importers`` package.

The ``sources`` package has been renamed to ``importers``.  These aliases
keep existing ``from harness_evals.sources.langfuse import LangfuseSource``
and ``from harness_evals.sources.otel import OTELSource`` imports working.

**Preferred imports** (new code should use these)::

    from harness_evals.importers.langfuse import LangfuseEvalCaseSource
    from harness_evals.importers.otel import OTELEvalCaseSource
"""

import warnings

__all__ = ["LangfuseSource", "OTELSource"]

_ALIASES = {
    "LangfuseSource": ("harness_evals.importers.langfuse", "LangfuseEvalCaseSource"),
    "OTELSource": ("harness_evals.importers.otel", "OTELEvalCaseSource"),
}


def __getattr__(name: str) -> object:
    alias = _ALIASES.get(name)
    if alias is not None:
        module_path, class_name = alias
        import importlib

        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        warnings.warn(
            f"harness_evals.sources.{name} is deprecated — "
            f"use {module_path}.{class_name} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
