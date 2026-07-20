"""Shared exception types for harness-evals."""

from __future__ import annotations

from typing import Any


class HarnessEvalsError(Exception):
    """Base exception for harness-evals runtime and configuration errors."""


class TargetInvocationError(HarnessEvalsError):
    """Raised when a target cannot produce an EvalCase."""

    def __init__(self, message: str, *, latency_ms: float | None = None) -> None:
        self.latency_ms = latency_ms
        super().__init__(message)


class MissingAdapterError(HarnessEvalsError):
    """Raised when a source, target, sink, or store adapter is not registered."""

    def __init__(self, source: str, family: str, install_hint: str) -> None:
        self.source = source
        self.family = family
        self.install_hint = install_hint
        super().__init__(
            f"Adapter {source!r} for family {family!r} is not installed. Install it with: pip install {install_hint}"
        )


class UnknownMetricError(HarnessEvalsError):
    """Raised when a metric kind cannot be resolved."""

    def __init__(self, kind: str, valid: list[str] | None = None) -> None:
        self.kind = kind
        self.valid = valid or []
        valid_suffix = f" Valid metrics: {', '.join(self.valid)}" if self.valid else ""
        super().__init__(f"Unknown metric kind {kind!r}.{valid_suffix}")


class UnmappedMetricError(HarnessEvalsError):
    """Raised when an imported platform metric has no local equivalent."""

    def __init__(self, source: str, metric: str, mapping: dict[str, Any] | None = None) -> None:
        self.source = source
        self.metric = metric
        self.mapping = mapping or {}
        super().__init__(f"Metric {metric!r} from source {source!r} has no local mapping.")


class BaselineRegressionError(HarnessEvalsError):
    """Raised when current scores regress against the configured baseline."""

    def __init__(self, result: Any) -> None:
        self.result = result
        message = result.summary() if hasattr(result, "summary") else str(result)
        super().__init__(message)
