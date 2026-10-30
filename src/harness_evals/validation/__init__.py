"""Configured metric contracts and pure admission validation.

Catalog discovery describes possibilities. Resolve a selected configuration,
then validate its context and actual evidence before choosing to run a scorer.
"""

from harness_evals.validation.contracts import describe_configured_metric, resolve_metric_contract
from harness_evals.validation.evidence import validate_metric
from harness_evals.validation.types import (
    CONTRACT_VERSION,
    ConfiguredMetricContract,
    EvidenceRequirement,
    ExecutionKind,
    ReasonCode,
    ValidationContext,
    ValidationReason,
    ValidationResult,
)

__all__ = [
    "CONTRACT_VERSION",
    "ConfiguredMetricContract",
    "EvidenceRequirement",
    "ExecutionKind",
    "ReasonCode",
    "ValidationContext",
    "ValidationReason",
    "ValidationResult",
    "describe_configured_metric",
    "resolve_metric_contract",
    "validate_metric",
]
