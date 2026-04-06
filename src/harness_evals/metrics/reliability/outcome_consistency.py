from __future__ import annotations

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import Dimension, ReliabilityMetric
from harness_evals.core.score import Score


class OutcomeConsistencyMetric(ReliabilityMetric):
    """Fraction of K runs that produce the same output as the majority.

    Maps to C_out from Rabanser et al. — the pass-all-K consistency measure.
    value = (count of most common output) / K.
    A score of 1.0 means all runs produced identical output.
    """

    def __init__(self, threshold: float = 0.8, k: int = 5, **kwargs: object) -> None:
        super().__init__(name="outcome_consistency", dimension=Dimension.CORRECTNESS, threshold=threshold, k=k, **kwargs)

    def measure_runs(self, eval_case: EvalCase) -> Score:
        runs = eval_case.runs or []
        if len(runs) < 2:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"Need at least 2 runs, got {len(runs)}",
            )

        outputs = [str(run.output) for run in runs]
        most_common_count = max(outputs.count(o) for o in set(outputs))
        value = most_common_count / len(outputs)

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            metadata={"k": len(runs), "unique_outputs": len(set(outputs))},
        )
