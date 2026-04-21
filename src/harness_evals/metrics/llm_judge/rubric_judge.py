"""RubricJudge metric — LLM scores output against a multi-level rubric.

Takes a flat ``{level: description}`` rubric (e.g. ``{5: "Excellent", ..., 1: "Poor"}``);
the judge selects the best-matching level. The level is normalized to 0.0-1.0 for
``Score.value`` and preserved in ``Score.metadata``.

For score-band rubrics (ranges of integer scores per band), use ``GEvalMetric`` with
``rubric: list[RubricLevel]``.
"""

from __future__ import annotations

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM

_DEFAULT_RUBRIC = {
    5: "Excellent — fully addresses the task, accurate, well-structured, no issues.",
    4: "Good — mostly addresses the task with minor issues.",
    3: "Acceptable — partially addresses the task, some notable gaps.",
    2: "Poor — significant issues, partially wrong or incomplete.",
    1: "Very poor — mostly wrong, off-topic, or barely addresses the task.",
}

_PROMPT_HEADER = """You are an expert evaluator. Score the following output using the rubric below.

Evaluate ONLY based on the input and output provided below. Do not infer or assume
information that is not explicitly present. If the output lacks a particular element,
treat it as absent rather than inferring it.
"""

_STEPS_SECTION = """
**Evaluation steps** (follow each step in order):
{steps}
"""

_IO_SECTION = """
**Input**: {input}

**Output**: {output}

{expected_section}

**Rubric** (score {min_level}-{max_level}):
{rubric_text}

Evaluate the output and select the rubric level that best matches.

Respond with JSON:
{{"reasoning": "your evaluation reasoning", "level": <integer {min_level}-{max_level}>}}
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["reasoning", "level"],
    "properties": {
        "reasoning": {"type": "string"},
        "level": {"type": "integer"},
    },
}


class RubricJudgeMetric(BaseMetric):
    """LLM-judged evaluation using a multi-level rubric.

    Scores on a 1-N scale (default 1-5), normalized to 0.0-1.0.
    Custom rubrics can be provided as ``{level: description}`` dicts.

    Optional ``evaluation_steps`` enable numbered chain-of-thought reasoning
    before the judge selects a level.
    """

    def __init__(
        self,
        llm: BaseLLM,
        rubric: dict[int, str] | None = None,
        threshold: float = 0.6,
        *,
        evaluation_steps: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(name="rubric_judge", dimension=Dimension.CORRECTNESS, threshold=threshold, **kwargs)
        self.llm = llm
        if rubric is not None and not rubric:
            raise ValueError("rubric must be a non-empty dict mapping int levels to descriptions")
        self.rubric = rubric or _DEFAULT_RUBRIC
        self.evaluation_steps = list(evaluation_steps) if evaluation_steps else []

    def _build_prompt(self, eval_case: EvalCase) -> str:
        levels = sorted(self.rubric.keys())
        min_level, max_level = levels[0], levels[-1]

        rubric_text = "\n".join(f"  {k}: {v}" for k, v in sorted(self.rubric.items(), reverse=True))
        expected_section = f"**Expected output**: {eval_case.expected}" if eval_case.expected else ""

        parts = [_PROMPT_HEADER]
        if self.evaluation_steps:
            steps_text = "\n".join(f"  {i + 1}. {step}" for i, step in enumerate(self.evaluation_steps))
            parts.append(_STEPS_SECTION.format(steps=steps_text))
        parts.append(
            _IO_SECTION.format(
                input=eval_case.input,
                output=eval_case.output,
                expected_section=expected_section,
                rubric_text=rubric_text,
                min_level=min_level,
                max_level=max_level,
            )
        )
        return "\n".join(parts)

    def measure(self, eval_case: EvalCase) -> Score:
        return _run_async(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        levels = sorted(self.rubric.keys())
        min_level, max_level = levels[0], levels[-1]

        prompt = self._build_prompt(eval_case)
        result = await self.llm.generate_json(prompt, _RESPONSE_SCHEMA)
        level = int(result.get("level", min_level))
        level = max(min_level, min(max_level, level))
        reasoning = result.get("reasoning", "")

        value = (level - min_level) / (max_level - min_level) if max_level > min_level else 1.0

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=reasoning,
            metadata={"level": level, "max_level": max_level},
        )
