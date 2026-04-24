"""Actionability metric — can a developer copy-paste and use this fix?"""

from __future__ import annotations

from harness_evals.core.metric import Dimension
from harness_evals.llm.base import BaseLLM
from harness_evals.metrics.llm_judge.geval import GEvalMetric
from harness_evals.metrics.llm_judge.types import RubricLevel


class ActionabilityMetric(GEvalMetric):
    """Evaluates how easily a developer can apply the fix.

    Weight in composite RQI: 5%.
    """

    criteria = (
        "Evaluate how easily a developer can apply this fix. Can they copy-paste the "
        "code and have it work? Are necessary commands or version changes included for "
        "dependency upgrades? Are file paths and line numbers referenced? "
        "Is the output complete (not truncated)?"
    )

    evaluation_steps = [
        "Check if the code fix is complete and ready to use — not truncated mid-line.",
        "For dependency or component vulnerabilities, check whether the necessary commands, "
        "version identifiers, or configuration changes are provided for the relevant technology stack.",
        "Check if file paths or line numbers are referenced so the developer knows where to apply the fix.",
        "Assess whether the fix includes context about what was changed and why, so the developer "
        "understands the rationale rather than applying a blind patch.",
        "Evaluate if the developer would need significant additional research to implement this fix.",
        "Assess overall readiness — could a developer apply the fix with minimal additional work, "
        "or would substantial effort be needed to make it usable?",
    ]

    rubric = [
        RubricLevel(0, 3, "Truncated, incomplete, or requires major additional work."),
        RubricLevel(4, 6, "Usable but missing commands, file refs, or some assembly required."),
        RubricLevel(7, 8, "Nearly copy-paste ready with minor gaps."),
        RubricLevel(9, 10, "Fully actionable: complete code, commands, file refs, ready to apply."),
    ]

    def __init__(self, llm: BaseLLM, threshold: float = 0.5, **kwargs: object) -> None:
        super().__init__(llm=llm, threshold=threshold, dimension=Dimension.CORRECTNESS, **kwargs)
