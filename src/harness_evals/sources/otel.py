"""Backwards-compatibility alias — use ``harness_evals.importers.otel`` instead.

``OTELSource`` is an alias for
:class:`~harness_evals.importers.otel.OTELEvalCaseSource`.
"""

from harness_evals.importers.otel import OTELEvalCaseSource

OTELSource = OTELEvalCaseSource

__all__ = ["OTELSource"]
