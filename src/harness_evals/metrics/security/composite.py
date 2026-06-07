"""Remediation Quality Index — weighted composite of the 7 security metrics."""

from __future__ import annotations

from harness_evals.core.score import Score

REMEDIATION_WEIGHTS: dict[str, float] = {
    "VulnerabilityCorrectnessMetric": 0.25,
    "SecurityCompletenessMetric": 0.20,
    "CodeSafetyMetric": 0.15,
    "CodeQualityMetric": 0.10,
    "ExplanationQualityMetric": 0.15,
    "RootCauseAnalysisMetric": 0.10,
    "ActionabilityMetric": 0.05,
}


def remediation_quality_index(
    scores: list[Score],
    *,
    weights: dict[str, float] | None = None,
    threshold: float = 0.5,
) -> Score:
    """Compute a weighted Remediation Quality Index from individual metric scores.

    Args:
        scores: Score objects from the 7 security remediation metrics.
        weights: Optional custom weight mapping (metric name -> weight).
                 Defaults to ``REMEDIATION_WEIGHTS``.
        threshold: Pass/fail threshold for the composite score.

    Returns:
        A composite Score with name ``"RemediationQualityIndex"``.
        Missing metrics are skipped; weights are re-normalized.
    """
    w = weights or REMEDIATION_WEIGHTS
    total = 0.0
    total_weight = 0.0
    matched = []

    for score in scores:
        weight = w.get(score.name)
        if weight is not None:
            total += score.value * weight
            total_weight += weight
            matched.append(score.name)

    value = total / total_weight if total_weight > 0 else 0.0

    return Score(
        name="RemediationQualityIndex",
        value=value,
        threshold=threshold,
        reason=f"Composite score computed from {len(matched)} of {len(w)} security metrics ({len(matched)}/{len(w)} available)",
        metadata={"matched_metrics": matched, "weights": w},
    )
