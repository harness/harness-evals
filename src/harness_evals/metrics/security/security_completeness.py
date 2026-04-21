"""Security Completeness metric — edge cases handled? Defense-in-depth?"""

from __future__ import annotations

from harness_evals.core.metric import Dimension
from harness_evals.llm.base import BaseLLM
from harness_evals.metrics.llm_judge.geval import GEvalMetric
from harness_evals.metrics.llm_judge.types import RubricLevel


class SecurityCompletenessMetric(GEvalMetric):
    """Evaluates how thoroughly the fix addresses the vulnerability including edge cases.

    Weight in composite RQI: 20%.
    """

    criteria = (
        "Evaluate how thoroughly the fix addresses the vulnerability, including edge cases "
        "and defense-in-depth. A minimal single-point fix that only handles the most obvious "
        "case is partially effective. A comprehensive fix that applies multiple layers of "
        "protection and considers boundary conditions is thorough."
    )

    evaluation_steps = [
        "Assess whether the fix handles only the primary case or also considers boundary "
        "and edge conditions relevant to the vulnerability type.",
        "Evaluate whether the fix uses defense-in-depth: multiple layers of protection rather than a single check.",
        "Determine the coverage depth relative to the vulnerability category — does the fix "
        "address the full lifecycle of the vulnerable pattern (input validation, data handling, "
        "safe output) or only one point in that chain?",
        "For dependency or component vulnerabilities, check whether the upgrade path is fully "
        "specified with the target version and update mechanism.",
        "Assess whether the fix could be trivially bypassed by an attacker using common evasion techniques.",
    ]

    rubric = [
        RubricLevel(0, 3, "Minimal fix, easily bypassable, single-point only."),
        RubricLevel(4, 6, "Handles the main case but misses edge cases or lacks depth."),
        RubricLevel(7, 8, "Good coverage with some defense-in-depth."),
        RubricLevel(9, 10, "Comprehensive: multiple layers, edge cases handled, defense-in-depth."),
    ]

    def __init__(self, llm: BaseLLM, threshold: float = 0.5, **kwargs: object) -> None:
        super().__init__(llm=llm, threshold=threshold, dimension=Dimension.CORRECTNESS, **kwargs)
