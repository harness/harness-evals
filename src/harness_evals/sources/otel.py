"""Backwards-compatibility alias — use ``harness_evals.importers.otel`` instead.

``OTELSource`` is an alias for
:class:`~harness_evals.importers.otel.OTELEvalCaseSource`.
"""

import warnings

from harness_evals.importers.otel import OTELEvalCaseSource as _OTELEvalCaseSource


def __getattr__(name: str) -> object:
    if name == "OTELSource":
        warnings.warn(
            "harness_evals.sources.otel.OTELSource is deprecated — "
            "use harness_evals.importers.otel.OTELEvalCaseSource instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        globals()["OTELSource"] = _OTELEvalCaseSource
        return _OTELEvalCaseSource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
