"""Importer adapters for harness-evals.

Importers pull already-produced data from external platforms into the
harness-evals type system:

- :class:`BaseEvalCaseSource` — fetch traces/runs as ``list[EvalCase]``,
  ready for ``evaluate_cases()``.
- :class:`BaseEvalConfigSource` — translate a platform eval definition into
  an ``EvalConfig``, ready for ``run_config()``.

Built-in implementations::

    from harness_evals.importers import LangfuseEvalCaseSource  # [langfuse] extra
    from harness_evals.importers import OTELEvalCaseSource      # [otlp] extra
    from harness_evals.importers import HarnessOTELEvalCaseSource  # [harness] extra
"""

from harness_evals.importers.base import BaseEvalCaseSource, BaseEvalConfigSource

__all__ = [
    "BaseEvalCaseSource",
    "BaseEvalConfigSource",
    "LangfuseEvalCaseSource",
    "OTELEvalCaseSource",
    "HarnessOTELEvalCaseSource",
]


def __getattr__(name: str) -> object:
    if name == "LangfuseEvalCaseSource":
        from harness_evals.importers.langfuse import LangfuseEvalCaseSource

        return LangfuseEvalCaseSource
    if name == "OTELEvalCaseSource":
        from harness_evals.importers.otel import OTELEvalCaseSource

        return OTELEvalCaseSource
    if name == "HarnessOTELEvalCaseSource":
        from harness_evals.importers.harness_otel import HarnessOTELEvalCaseSource

        return HarnessOTELEvalCaseSource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
