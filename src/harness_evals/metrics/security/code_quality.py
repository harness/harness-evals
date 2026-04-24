"""Code Quality metric — correct language? Compilable? Follows idioms?"""

from __future__ import annotations

from harness_evals.core.metric import Dimension
from harness_evals.llm.base import BaseLLM
from harness_evals.metrics.llm_judge.geval import GEvalMetric
from harness_evals.metrics.llm_judge.types import RubricLevel


class CodeQualityMetric(GEvalMetric):
    """Evaluates syntactic and stylistic quality of the generated code fix.

    Weight in composite RQI: 10%.
    """

    criteria = (
        "Evaluate the syntactic and stylistic quality of the generated code fix. "
        "Is it in the correct programming language for the file? Is it syntactically "
        "correct? Does it follow language idioms and best practices? Are any referenced "
        "dependencies real (not hallucinated)?"
    )

    evaluation_steps = [
        "Determine the expected programming language from the filename extension and context in the input.",
        "Check if the code fix is written in the programming language expected for the file "
        "based on its extension or the context described in the input.",
        "Assess whether the fix uses the wrong programming language entirely — this is a fundamental quality failure.",
        "Check for obvious syntax errors that would prevent compilation or interpretation.",
        "Assess code completeness — is the fix a complete, insertable unit (function, block, file), "
        "or a fragment that requires the developer to guess the surrounding context?",
        "Evaluate whether the code follows idiomatic patterns and conventions for that language.",
        "Check if any imported libraries or dependencies actually exist.",
    ]

    rubric = [
        RubricLevel(0, 2, "Wrong programming language or completely broken syntax."),
        RubricLevel(3, 5, "Correct language but has syntax issues or non-idiomatic patterns."),
        RubricLevel(6, 8, "Clean, compilable code with minor style issues."),
        RubricLevel(9, 10, "Production-quality code: correct language, idiomatic, clean."),
    ]

    def __init__(self, llm: BaseLLM, threshold: float = 0.5, **kwargs: object) -> None:
        super().__init__(llm=llm, threshold=threshold, dimension=Dimension.CORRECTNESS, **kwargs)
