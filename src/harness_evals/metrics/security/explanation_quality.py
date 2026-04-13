"""Explanation Quality metric — is the remediation context accurate and specific?"""

from __future__ import annotations

from harness_evals.core.metric import Dimension
from harness_evals.llm.base import BaseLLM
from harness_evals.metrics.security._base import RubricLevel, SecurityRemediationMetric


class ExplanationQualityMetric(SecurityRemediationMetric):
    """Evaluates quality of the remediation explanation text.

    Weight in composite RQI: 15%.
    """

    criteria = (
        "Evaluate the quality of the remediation context / explanation text. "
        "Is it accurate and specific to this vulnerability (not generic textbook advice)? "
        "Does it reference the correct security standards? Are the remediation steps actionable? "
        "Bonus value if it includes an attack scenario, references, and verification guidance."
    )

    evaluation_steps = [
        "Check if the explanation is SPECIFIC to the actual vulnerability in the code, "
        "or if it is generic advice that could apply to any vulnerability.",
        "Assess whether the remediation steps contain substantive, vulnerability-specific guidance "
        "or rely on generic security advice not tied to the reported issue.",
        "Check if security standard references (such as CWE, CVE, OWASP, CIS benchmarks, "
        "or NVD advisories) are present and correct for the vulnerability type.",
        "Evaluate whether remediation steps are actionable: can a developer follow them "
        "without needing to research further?",
        "Note whether an attack scenario is present — this adds value but its absence alone "
        "should not heavily penalize an otherwise specific and accurate explanation.",
        "Assess the overall specificity — does the explanation reference the actual code, "
        "provide correct standards references, and include actionable steps?",
    ]

    rubric = [
        RubricLevel(0, 3, "Generic textbook advice, wrong CWE, or filler steps."),
        RubricLevel(4, 6, "Somewhat specific but missing references or attack scenario."),
        RubricLevel(7, 8, "Specific with correct CWE, actionable steps, minor gaps."),
        RubricLevel(9, 10, "Code-specific, correct CWE, attack scenario, verification checklist."),
    ]

    def __init__(self, llm: BaseLLM, threshold: float = 0.5, **kwargs: object) -> None:
        super().__init__(llm=llm, threshold=threshold, dimension=Dimension.CORRECTNESS, **kwargs)
