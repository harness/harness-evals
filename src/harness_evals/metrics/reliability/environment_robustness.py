"""Environment robustness metric — accuracy ratio under structural perturbations (Rabanser et al.)."""

from __future__ import annotations

from harness_evals.core.metric import Dimension
from harness_evals.metrics.reliability.robustness_base import RobustnessMetric


class EnvironmentRobustnessMetric(RobustnessMetric):
    """Measures stability under environment perturbations (JSON reordering, schema changes, etc.).

    R_env = min(Acc_perturbed / Acc_nominal, 1)

    Targets environment-level changes: field reordering, date/time format
    shifts, naming convention changes, response wrapping, and tool interface
    parameter renaming.

    Reference: Rabanser et al., Table 2 — R_env = min(Acc_pert / Acc_0, 1).
    """

    def __init__(self, threshold: float = 0.8, **kwargs: object) -> None:
        super().__init__(name="environment_robustness", dimension=Dimension.CORRECTNESS, threshold=threshold, **kwargs)
