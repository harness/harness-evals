"""Prompt robustness metric — accuracy ratio under semantic rephrasings (Rabanser et al.)."""

from __future__ import annotations

from harness_evals.metrics.reliability.robustness_base import RobustnessMetric


class PromptRobustnessMetric(RobustnessMetric):
    """Measures invariance to semantically equivalent prompt rephrasings.

    R_prompt = min(Acc_perturbed / Acc_nominal, 1)

    Reference: Rabanser et al., Table 2 — R_prompt = min(Acc_para / Acc_0, 1).
    """

    def __init__(self, threshold: float = 0.8, **kwargs: object) -> None:
        super().__init__(name="prompt_robustness", threshold=threshold, **kwargs)
