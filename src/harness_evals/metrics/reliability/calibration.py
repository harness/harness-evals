"""Calibration metric — Expected Calibration Error (ECE) over a dataset."""

from __future__ import annotations

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score


class CalibrationMetric(BaseMetric):
    """Expected Calibration Error (ECE) over a set of eval cases.

    Bins eval cases by ``confidence``, compares average confidence to actual
    success rate in each bin. Score = 1 - ECE (higher is better).

    This metric operates over multiple eval cases via ``measure_dataset()``.
    For a single eval case, returns 0.0 (not enough data).

    Each eval case must have ``confidence`` set and a pre-computed pass/fail
    status (passed via metadata["passed"] or determined by the caller).
    """

    def __init__(self, n_bins: int = 10, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(name="calibration", threshold=threshold, **kwargs)
        self.n_bins = n_bins

    def measure(self, eval_case: EvalCase) -> Score:
        return Score(
            name=self.name, value=0.0, threshold=self.threshold,
            reason="Calibration requires multiple eval cases — use measure_dataset()",
        )

    def measure_dataset(self, cases: list[EvalCase], outcomes: list[bool]) -> Score:
        """Compute ECE over a set of eval cases with known outcomes.

        Args:
            cases: Eval cases with ``confidence`` set.
            outcomes: Whether each case was a success (True) or failure (False).
        """
        if len(cases) != len(outcomes):
            return Score(
                name=self.name, value=0.0, threshold=self.threshold,
                reason=f"cases ({len(cases)}) and outcomes ({len(outcomes)}) must have same length",
            )

        pairs = []
        for case, outcome in zip(cases, outcomes, strict=True):
            conf = case.confidence
            if conf is not None:
                pairs.append((conf, outcome))

        if len(pairs) < 2:
            return Score(
                name=self.name, value=0.0, threshold=self.threshold,
                reason=f"Need at least 2 cases with confidence, got {len(pairs)}",
            )

        # Compute ECE
        bin_width = 1.0 / self.n_bins
        ece = 0.0
        total = len(pairs)

        for i in range(self.n_bins):
            lo = i * bin_width
            hi = lo + bin_width
            bin_pairs = [(c, o) for c, o in pairs if lo <= c < hi or (i == self.n_bins - 1 and c == hi)]

            if not bin_pairs:
                continue

            avg_conf = sum(c for c, _ in bin_pairs) / len(bin_pairs)
            avg_acc = sum(1 for _, o in bin_pairs if o) / len(bin_pairs)
            ece += (len(bin_pairs) / total) * abs(avg_acc - avg_conf)

        value = max(0.0, 1.0 - ece)

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=f"ECE={ece:.4f} over {total} cases in {self.n_bins} bins",
            metadata={"ece": ece, "n_cases": total, "n_bins": self.n_bins},
        )
