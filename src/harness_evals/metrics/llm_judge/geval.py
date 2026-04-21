"""GEval metric — LLM scores output against configurable criteria with chain-of-thought.

Supports three modes of use, in increasing structure:

1. **Free-form criteria** (default)::

       GEvalMetric(llm, criteria="Is the output accurate and helpful?")

   The judge returns a float score in 0.0-1.0.

2. **Criteria + evaluation steps**::

       GEvalMetric(
           llm,
           criteria="...",
           evaluation_steps=["Step 1", "Step 2", ...],
       )

   The judge follows a numbered chain-of-thought before scoring.

3. **Criteria + evaluation steps + score-band rubric**::

       GEvalMetric(
           llm,
           criteria="...",
           evaluation_steps=[...],
           rubric=[RubricLevel(0, 2, "..."), RubricLevel(3, 5, "..."), ...],
       )

   The judge assigns an integer score from the rubric's range; the metric
   normalizes it to 0.0-1.0 and preserves the raw score in ``Score.metadata``.

Subclasses (e.g. security remediation metrics) typically use mode 3 by
declaring class-level ``criteria``, ``evaluation_steps``, ``rubric`` attributes.
"""

from __future__ import annotations

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM
from harness_evals.metrics.llm_judge.types import RubricLevel

# Prompt sections have NO leading/trailing blank lines; sections are joined
# with "\n\n" for exactly one blank line between them (no triple newlines).

_PROMPT_HEADER = """You are an expert evaluator. Score the following output against the given criteria.

Evaluate ONLY based on the input and output provided below. Do not infer or assume
information that is not explicitly present. If the output lacks a particular element,
treat it as absent rather than inferring it.

**Criteria**: {criteria}"""

_STEPS_SECTION = """**Evaluation steps** (follow each step in order):
{steps}"""

_RUBRIC_SECTION = """**Rubric** (use these ranges to assign your final score):
{rubric}"""

_IO_SECTION_WITHOUT_EXPECTED = """**Input**: {input}

**Output**: {output}"""

_IO_SECTION_WITH_EXPECTED = """**Input**: {input}

**Output**: {output}

**Expected output**: {expected}"""

_FLOAT_SCORING_INSTRUCTION = """First, reason step-by-step about how well the output meets the criteria.
Then provide your score.

Respond with JSON:
{"reasoning": "your chain-of-thought reasoning", "score": <float between 0.0 and 1.0>}"""

_INT_SCORING_INSTRUCTION = """First, reason step-by-step through each evaluation step above.
Then assign an integer score from {min_score} to {max_score} based on the rubric. The rubric is the
sole authority for mapping your observations to a numeric score.

Respond with JSON:
{{"reasoning": "your chain-of-thought reasoning", "score": <integer {min_score}-{max_score}>}}"""

_FLOAT_SCHEMA = {
    "type": "object",
    "required": ["reasoning", "score"],
    "properties": {
        "reasoning": {"type": "string"},
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}

_INT_SCHEMA = {
    "type": "object",
    "required": ["reasoning", "score"],
    "properties": {
        "reasoning": {"type": "string"},
        "score": {"type": "integer"},
    },
}


class GEvalMetric(BaseMetric):
    """LLM-judged evaluation using configurable criteria and chain-of-thought.

    When ``rubric`` is omitted, the judge returns a float score in 0.0-1.0.
    When ``rubric`` is provided, the judge returns an integer score in the
    rubric's range (e.g. 0-10), which is normalized to 0.0-1.0 for
    ``Score.value``; the raw integer is preserved in ``Score.metadata``.

    Subclasses may declare ``criteria``, ``evaluation_steps``, ``rubric`` as
    class attributes; constructor arguments override them if given.
    """

    # Optional class-level defaults. Subclasses may override.
    criteria: str = ""
    evaluation_steps: list[str] = []
    rubric: list[RubricLevel] = []

    def __init__(
        self,
        llm: BaseLLM,
        criteria: str | None = None,
        threshold: float = 0.7,
        *,
        evaluation_steps: list[str] | None = None,
        rubric: list[RubricLevel] | None = None,
        name: str | None = None,
        dimension: Dimension = Dimension.CORRECTNESS,
        **kwargs: object,
    ) -> None:
        resolved_criteria = (
            criteria
            if criteria is not None
            else (self.__class__.criteria or "Is the response accurate, relevant, and complete?")
        )
        # Always copy so neither the caller's list nor the class-level default can be mutated
        # through the instance (and vice versa).
        resolved_steps = list(evaluation_steps if evaluation_steps is not None else self.__class__.evaluation_steps)
        resolved_rubric = list(rubric if rubric is not None else self.__class__.rubric)
        resolved_name = (
            name if name is not None else ("geval" if type(self) is GEvalMetric else self.__class__.__name__)
        )

        if resolved_rubric:
            for r in resolved_rubric:
                if not isinstance(r, RubricLevel):
                    raise TypeError(f"rubric entries must be RubricLevel instances, got {type(r).__name__}")

        super().__init__(name=resolved_name, dimension=dimension, threshold=threshold, **kwargs)
        self.llm = llm
        self.criteria = resolved_criteria
        self.evaluation_steps = resolved_steps
        self.rubric = resolved_rubric

    def _build_prompt(self, eval_case: EvalCase) -> str:
        parts = [_PROMPT_HEADER.format(criteria=self.criteria)]

        if self.evaluation_steps:
            steps_text = "\n".join(f"  {i + 1}. {step}" for i, step in enumerate(self.evaluation_steps))
            parts.append(_STEPS_SECTION.format(steps=steps_text))

        if self.rubric:
            rubric_text = "\n".join(f"  {r.min_score}-{r.max_score}: {r.description}" for r in self.rubric)
            parts.append(_RUBRIC_SECTION.format(rubric=rubric_text))

        if eval_case.expected is not None:
            parts.append(
                _IO_SECTION_WITH_EXPECTED.format(
                    input=eval_case.input,
                    output=eval_case.output,
                    expected=eval_case.expected,
                )
            )
        else:
            parts.append(
                _IO_SECTION_WITHOUT_EXPECTED.format(
                    input=eval_case.input,
                    output=eval_case.output,
                )
            )

        if self.rubric:
            min_score = min(r.min_score for r in self.rubric)
            max_score = max(r.max_score for r in self.rubric)
            parts.append(_INT_SCORING_INSTRUCTION.format(min_score=min_score, max_score=max_score))
        else:
            parts.append(_FLOAT_SCORING_INSTRUCTION)

        return "\n\n".join(parts)

    def measure(self, eval_case: EvalCase) -> Score:
        return _run_async(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        prompt = self._build_prompt(eval_case)
        schema = _INT_SCHEMA if self.rubric else _FLOAT_SCHEMA
        result = await self.llm.generate_json(prompt, schema)

        reasoning = result.get("reasoning", "")

        if self.rubric:
            min_score = min(r.min_score for r in self.rubric)
            max_score = max(r.max_score for r in self.rubric)
            raw = result.get("score", min_score)
            try:
                raw_score = int(raw)
            except (TypeError, ValueError):
                raw_score = min_score
            raw_score = max(min_score, min(max_score, raw_score))
            span = max_score - min_score
            value = (raw_score - min_score) / span if span > 0 else 1.0
            metadata = {"raw_score": raw_score, "min_score": min_score, "max_score": max_score}
        else:
            raw = result.get("score", 0.0)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = 0.0
            value = max(0.0, min(1.0, value))
            metadata = None

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=reasoning,
            metadata=metadata,
        )
