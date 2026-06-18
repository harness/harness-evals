"""Backwards-compatibility alias — use ``harness_evals.importers.langfuse`` instead."""

import warnings

__all__ = ["LangfuseSource"]


def __getattr__(name: str) -> object:
    if name == "LangfuseSource":
        from harness_evals.importers.langfuse import LangfuseEvalCaseSource

        warnings.warn(
            "LangfuseSource is deprecated — "
            "use harness_evals.importers.langfuse.LangfuseEvalCaseSource instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return LangfuseEvalCaseSource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
