"""Brier Score metric — joint calibration and discrimination (Rabanser et al.)."""

from __future__ import annotations

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score


class BrierScoreMetric(BaseMetric):
    """Brier Score as a proper scoring rule for predictability.

    Measures both calibration and discrimination jointly:
    P_brier = 1 - (1/T) * sum((c_i - y_i)^2)

    where c_i is confidence in [0, 1] and y_i is binary outcome (0 or 1).
    A perfect predictor that assigns confidence 1.0 to all successes and 0.0
    to all failures scores 1.0.  Random guessing with 0.5 confidence scores
    0.75.

    This metric operates over multiple eval cases via ``measure_dataset()``.
    For a single eval case, returns 0.0 (not enough data).

    Reference: Rabanser et al., "Towards a Science of AI Agent Reliability"
    (Table 2, Equation 2 — R_Pred = P_brier).
    """

    def __init__(self, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(name="brier_score", threshold=threshold, **kwargs)

    def measure(self, eval_case: EvalCase) -> Score:
        return Score(
            name=self.name,
            value=0.0,
            threshold=self.threshold,
            reason="Brier score requires multiple eval cases — use measure_dataset()",
        )

    def measure_dataset(self, cases: list[EvalCase], outcomes: list[bool]) -> Score:
        """Compute Brier score over a set of eval cases with known outcomes.

        Args:
            cases: Eval cases with ``confidence`` set.
            outcomes: Whether each case was a success (True) or failure (False).
        """
        if len(cases) != len(outcomes):
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"cases ({len(cases)}) and outcomes ({len(outcomes)}) must have same length",
            )

        pairs: list[tuple[float, float]] = []
        for case, outcome in zip(cases, outcomes, strict=True):
            conf = case.confidence
            if conf is not None:
                pairs.append((conf, 1.0 if outcome else 0.0))

        if len(pairs) < 2:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"Need at least 2 cases with confidence, got {len(pairs)}",
            )

        mse = sum((c - y) ** 2 for c, y in pairs) / len(pairs)
        value = max(0.0, min(1.0, 1.0 - mse))

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=f"Brier={value:.4f} (MSE={mse:.4f}) over {len(pairs)} cases",
            metadata={"mse": mse, "n_cases": len(pairs)},
        )
