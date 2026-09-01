"""Spec-grounding scoring: pure arithmetic and reason rendering (HE-1)."""

from harness_evals.metrics.grounding.scoring import (
    DEFAULT_WEIGHTS,
    ClaimResult,
    ClaimStatus,
    Improvement,
    RequirementResult,
    RequirementStatus,
    Risk,
    SpecGroundingResult,
    build_metadata,
    compute_score,
    render_reason,
    validate_weights,
)

__all__ = [
    "DEFAULT_WEIGHTS",
    "ClaimResult",
    "ClaimStatus",
    "Improvement",
    "RequirementResult",
    "RequirementStatus",
    "Risk",
    "SpecGroundingResult",
    "build_metadata",
    "compute_score",
    "render_reason",
    "validate_weights",
]
