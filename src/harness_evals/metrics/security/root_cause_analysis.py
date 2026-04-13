"""Root Cause Analysis metric — is the root cause accurately identified?"""

from __future__ import annotations

from harness_evals.core.metric import Dimension
from harness_evals.llm.base import BaseLLM
from harness_evals.metrics.security._base import RubricLevel, SecurityRemediationMetric


class RootCauseAnalysisMetric(SecurityRemediationMetric):
    """Evaluates whether the output accurately identifies the root cause of the vulnerability.

    Weight in composite RQI: 10%.
    """

    criteria = (
        "Evaluate whether the output accurately identifies the root cause of the "
        "vulnerability. The root cause analysis should trace the origin of the issue — "
        "whether it is a code-level data flow problem, a vulnerable dependency, an exposed "
        "secret, or a misconfigured resource — with specificity and precision."
    )

    evaluation_steps = [
        "Identify the vulnerability type from the input context.",
        "Assess whether the output traces the origin of the vulnerability to its specific "
        "source — this may involve identifying a data flow path, a vulnerable component "
        "and version, an exposure mechanism, or a misconfigured resource and field.",
        "Evaluate the precision of the root cause identification — does it reference "
        "specific code locations, configuration fields, component versions, or data flow "
        "paths, or is it vague and generic?",
        "Check whether the identified root cause is consistent with the vulnerability type described in the input.",
        "For vulnerability types that do not fit standard categories (such as business logic "
        "flaws, API misuse, or access control issues), assess whether the root cause analysis "
        "still identifies the specific mechanism that creates the exposure.",
        "Assess whether the analysis provides enough detail for a developer to understand "
        "exactly why the vulnerability exists.",
    ]

    rubric = [
        RubricLevel(0, 3, "Root cause is wrong or not identified."),
        RubricLevel(4, 6, "Root cause is partially correct but vague or missing specifics."),
        RubricLevel(7, 8, "Root cause is correct with reasonable specificity."),
        RubricLevel(9, 10, "Root cause is precise: correct taint path, line refs, or exact config field."),
    ]

    def __init__(self, llm: BaseLLM, threshold: float = 0.5, **kwargs: object) -> None:
        super().__init__(llm=llm, threshold=threshold, dimension=Dimension.GROUNDEDNESS, **kwargs)
