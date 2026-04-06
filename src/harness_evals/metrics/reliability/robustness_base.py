"""Shared base for ratio-based robustness metrics (Rabanser et al.).

Both prompt and environment robustness share the same formula:
  R = min(Acc_perturbed / Acc_nominal, 1)

Subclasses only set ``name``.  See ADR-008 for why the dataset-level
method is ``measure_robustness()`` rather than ``measure_dataset()``.
"""

from __future__ import annotations

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score


class RobustnessMetric(BaseMetric):
    """Base class for ratio-based robustness metrics (Rabanser et al.).

    Computes R = min(Acc_perturbed / Acc_nominal, 1), which disentangles
    robustness from raw capability.

    **Per-case usage** (``measure()``): reads from ``eval_case.metadata``:
      - ``nominal_passed`` (bool): did the unperturbed run pass?
      - ``perturbed_results`` (list[bool]): did each variant pass?

    **Dataset usage** (``measure_robustness()``): computes the accuracy ratio
    across an entire benchmark.  Named differently from the predictability
    ``measure_dataset(cases, outcomes)`` because the signatures are
    incompatible — see ADR-008.

    A system that fails nominally receives a robustness score of 1.0
    because degradation cannot be attributed to perturbation.  Evaluate
    raw capability separately.
    """

    def measure(self, eval_case: EvalCase) -> Score:
        _sentinel = object()
        nominal_passed = eval_case.meta("nominal_passed", _sentinel)
        perturbed_results = eval_case.meta("perturbed_results", _sentinel)

        if nominal_passed is _sentinel or perturbed_results is _sentinel:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason='metadata must contain "nominal_passed" (bool) and "perturbed_results" (list[bool])',
            )

        if not perturbed_results:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="perturbed_results is empty",
            )

        if not nominal_passed:
            return Score(
                name=self.name,
                value=1.0,
                threshold=self.threshold,
                reason="Nominal run failed — degradation not attributable to perturbation",
                metadata={"nominal_passed": False, "n_perturbed": len(perturbed_results)},
            )

        perturbed_acc = sum(perturbed_results) / len(perturbed_results)

        return Score(
            name=self.name,
            value=min(perturbed_acc, 1.0),
            threshold=self.threshold,
            reason=(f"{sum(perturbed_results)}/{len(perturbed_results)} perturbed variants passed"),
            metadata={
                "nominal_passed": True,
                "n_perturbed": len(perturbed_results),
                "perturbed_pass_rate": perturbed_acc,
            },
        )

    def measure_robustness(
        self,
        nominal_passed: list[bool],
        perturbed_passed: list[list[bool]],
    ) -> Score:
        """Compute robustness ratio across a full dataset.

        Args:
            nominal_passed: Per-task nominal pass/fail (one bool per task).
            perturbed_passed: Per-task list of pass/fail for each perturbation variant.
        """
        if len(nominal_passed) != len(perturbed_passed):
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=(
                    f"nominal ({len(nominal_passed)}) and perturbed ({len(perturbed_passed)}) must have same length"
                ),
            )

        if not nominal_passed:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="Empty dataset",
            )

        acc_nominal = sum(nominal_passed) / len(nominal_passed)

        if acc_nominal == 0.0:
            return Score(
                name=self.name,
                value=1.0,
                threshold=self.threshold,
                reason="Nominal accuracy is 0 — degradation not attributable to perturbation",
                metadata={"acc_nominal": 0.0, "n_tasks": len(nominal_passed)},
            )

        total_perturbed = sum(len(variants) for variants in perturbed_passed)
        total_perturbed_passed = sum(sum(variants) for variants in perturbed_passed)

        if total_perturbed == 0:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="No perturbed results provided",
            )

        acc_perturbed = total_perturbed_passed / total_perturbed
        value = min(acc_perturbed / acc_nominal, 1.0)

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=(
                f"Acc(perturbed)={acc_perturbed:.4f} / Acc(nominal)={acc_nominal:.4f} "
                f"= {value:.4f} over {len(nominal_passed)} tasks"
            ),
            metadata={
                "acc_nominal": acc_nominal,
                "acc_perturbed": acc_perturbed,
                "n_tasks": len(nominal_passed),
                "n_perturbed_total": total_perturbed,
            },
        )
