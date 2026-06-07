"""Discrimination metric — AUC-ROC over (confidence, pass/fail) pairs."""

from __future__ import annotations

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score


class DiscriminationMetric(BaseMetric):
    """AUC-ROC measuring whether confidence separates successes from failures.

    Score = AUC-ROC over (confidence, outcome) pairs. 1.0 means confidence
    perfectly separates successes and failures. 0.5 means no discrimination.

    This metric operates over multiple eval cases via ``measure_dataset()``.
    """

    def __init__(self, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(name="discrimination", dimension=Dimension.CORRECTNESS, threshold=threshold, **kwargs)

    def measure(self, eval_case: EvalCase) -> Score:
        return Score(
            name=self.name,
            value=0.0,
            threshold=self.threshold,
            reason="Cannot compute discrimination on a single case — this metric requires multiple eval cases via measure_dataset()",
        )

    def measure_dataset(self, cases: list[EvalCase], outcomes: list[bool]) -> Score:
        """Compute AUC-ROC over eval cases with known outcomes.

        Args:
            cases: Eval cases with ``confidence`` set.
            outcomes: Whether each case was a success (True) or failure (False).
        """
        if len(cases) != len(outcomes):
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"Input mismatch — cases ({len(cases)}) and outcomes ({len(outcomes)}) must have the same length",
            )

        pairs = []
        for case, outcome in zip(cases, outcomes, strict=True):
            conf = case.confidence
            if conf is not None:
                pairs.append((conf, outcome))

        if len(pairs) < 2:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"Insufficient data — need at least 2 cases with confidence scores, but only found {len(pairs)}",
            )

        positives = sum(1 for _, o in pairs if o)
        negatives = len(pairs) - positives

        if positives == 0 or negatives == 0:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="Cannot compute discrimination — need both successes and failures to calculate AUC-ROC, but only one outcome type is present",
            )

        # Compute AUC-ROC via the Wilcoxon-Mann-Whitney statistic
        # Sort by confidence descending
        sorted_pairs = sorted(pairs, key=lambda x: x[0], reverse=True)

        auc = 0.0
        tp = 0
        fp = 0
        prev_fpr = 0.0
        prev_tpr = 0.0

        for _, outcome in sorted_pairs:
            if outcome:
                tp += 1
            else:
                fp += 1
            tpr = tp / positives
            fpr = fp / negatives
            # Trapezoidal rule
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
            prev_fpr = fpr
            prev_tpr = tpr

        value = max(0.0, min(1.0, auc))

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=f"Model confidence separates successes from failures with {value * 100:.0f}% discrimination across {len(pairs)} cases (AUC-ROC = {value:.4f}, {positives} successes, {negatives} failures)",
            metadata={"auc_roc": value, "n_cases": len(pairs), "n_positive": positives, "n_negative": negatives},
        )
