"""Preset criteria for GEvalMetric covering common evaluation domains.

Usage::

    from harness_evals.metrics import GEvalMetric
    from harness_evals.metrics.llm_judge import presets

    metric = GEvalMetric(llm=llm, criteria=presets.FACTUALITY, threshold=0.8)
"""

FACTUALITY = (
    "Evaluate the factual accuracy of the output compared to the expected answer. "
    "Score 1.0 if all facts are correct, 0.0 if major factual errors are present. "
    "Consider: subset of facts (0.4), superset (0.6), identical (1.0), "
    "contradicts expected (0.0), irrelevant differences (1.0)."
)

CLOSED_QA = (
    "Assess whether the output correctly answers the question. "
    "Score 1.0 if the answer is correct, 0.0 if incorrect. "
    "Partial credit for partially correct answers."
)

SQL_EQUIVALENCE = (
    "Compare the SQL queries for semantic equivalence. "
    "Score 1.0 if they would produce the same result set on any valid database, "
    "0.0 if they produce different results. Ignore formatting differences."
)

SUMMARIZATION = (
    "Evaluate the quality of the summary. Consider: coverage of key points, "
    "conciseness, accuracy, and readability. Score 0.0-1.0."
)

TRANSLATION = (
    "Evaluate translation fidelity. Score 1.0 if the translation preserves "
    "meaning, tone, and nuance. Score 0.0 if the meaning is lost or distorted."
)

HUMOR = "Rate whether the text is funny or humorous. Score 1.0 if funny, 0.0 if not."

FEASIBILITY = (
    "Evaluate whether the proposed solution is feasible and practical. "
    "Score 1.0 if fully feasible, 0.0 if impractical or impossible."
)

SECURITY = (
    "Identify security vulnerabilities in the code. Score 1.0 if no vulnerabilities "
    "found, 0.0 if critical vulnerabilities are present. Consider: injection, "
    "authentication, authorization, data exposure, and input validation."
)

__all__ = [
    "FACTUALITY",
    "CLOSED_QA",
    "SQL_EQUIVALENCE",
    "SUMMARIZATION",
    "TRANSLATION",
    "HUMOR",
    "FEASIBILITY",
    "SECURITY",
]
