"""FaultRobustness metric — accuracy under injected faults (Rabanser et al. R_fault)."""

from __future__ import annotations

from harness_evals.metrics.reliability.robustness_base import RobustnessMetric


class FaultRobustnessMetric(RobustnessMetric):
    """Ratio-based robustness metric for fault injection scenarios.

    Measures ``accuracy(with faults) / accuracy(nominal)``, inheriting the
    full per-case ``measure()`` and dataset-level ``measure_robustness()``
    behaviour from :class:`RobustnessMetric`.

    Typically used with :class:`~harness_evals.testing.FaultInjector` to
    generate the nominal/perturbed results.
    """

    def __init__(self, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(name="fault_robustness", threshold=threshold, **kwargs)
