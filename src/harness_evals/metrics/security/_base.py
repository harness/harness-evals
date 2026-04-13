"""Base class for security remediation LLM-judged metrics.

Extends the GEval pattern with structured evaluation steps and rubrics for
chain-of-thought reasoning — the judge follows numbered steps and scores
against explicit rubric ranges.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM


@dataclass(frozen=True)
class RubricLevel:
    """A score range with an expected outcome description."""

    min_score: int
    max_score: int
    description: str


_PROMPT_TEMPLATE = """You are an expert security engineer evaluating AI-generated vulnerability remediation.

Evaluate ONLY based on the input and output provided below. Do not infer or assume
information that is not explicitly present. If the output lacks a particular element
(code fix, explanation, references), treat it as absent rather than inferring it.

**Criteria**: {criteria}

**Evaluation steps** (follow each step in order):
{steps}

**Rubric** (use these ranges to assign your final score):
{rubric}

**Input (vulnerability context)**:
{input}

**Output (AI-generated remediation)**:
{output}

First, reason step-by-step through each evaluation step above.
Then assign a score from 0 to 10 based on the rubric. The rubric is the sole authority
for mapping your observations to a numeric score.

Respond with JSON:
{{"reasoning": "your chain-of-thought reasoning", "score": <integer 0-10>}}
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["reasoning", "score"],
    "properties": {
        "reasoning": {"type": "string"},
        "score": {"type": "integer"},
    },
}


class SecurityRemediationMetric(BaseMetric):
    """LLM-judged metric for security remediation quality.

    Subclasses configure ``criteria``, ``evaluation_steps``, and ``rubric``
    to define a specific quality dimension. The LLM scores on a 0-10 scale
    which is normalized to 0.0-1.0 for the returned Score.
    """

    criteria: str = ""
    evaluation_steps: list[str] = []
    rubric: list[RubricLevel] = []

    def __init__(
        self,
        llm: BaseLLM,
        threshold: float = 0.5,
        *,
        dimension: Dimension = Dimension.CORRECTNESS,
        **kwargs: object,
    ) -> None:
        name = self.__class__.__name__
        super().__init__(name=name, dimension=dimension, threshold=threshold, **kwargs)
        self.llm = llm

    def _build_prompt(self, eval_case: EvalCase) -> str:
        steps_text = "\n".join(f"  {i + 1}. {step}" for i, step in enumerate(self.evaluation_steps))
        rubric_text = "\n".join(f"  {r.min_score}-{r.max_score}: {r.description}" for r in self.rubric)
        return _PROMPT_TEMPLATE.format(
            criteria=self.criteria,
            steps=steps_text,
            rubric=rubric_text,
            input=eval_case.input,
            output=eval_case.output,
        )

    def measure(self, eval_case: EvalCase) -> Score:
        return _run_async(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        prompt = self._build_prompt(eval_case)
        result = await self.llm.generate_json(prompt, _RESPONSE_SCHEMA)

        raw_score = int(result.get("score", 0))
        raw_score = max(0, min(10, raw_score))
        value = raw_score / 10.0
        reasoning = result.get("reasoning", "")

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=reasoning,
            metadata={"raw_score": raw_score, "max_score": 10},
        )
