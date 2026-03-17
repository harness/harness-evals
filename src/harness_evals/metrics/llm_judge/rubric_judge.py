"""RubricJudge metric — LLM scores output against a multi-level rubric."""

from __future__ import annotations

import asyncio

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM

_DEFAULT_RUBRIC = {
    5: "Excellent — fully addresses the task, accurate, well-structured, no issues.",
    4: "Good — mostly addresses the task with minor issues.",
    3: "Acceptable — partially addresses the task, some notable gaps.",
    2: "Poor — significant issues, partially wrong or incomplete.",
    1: "Very poor — mostly wrong, off-topic, or barely addresses the task.",
}

_PROMPT_TEMPLATE = """You are an expert evaluator. Score the following output using the rubric below.

**Input**: {input}

**Output**: {output}

{expected_section}

**Rubric** (score 1-{max_level}):
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
    """

    def __init__(
        self,
        llm: BaseLLM,
        rubric: dict[int, str] | None = None,
        threshold: float = 0.6,
        **kwargs: object,
    ) -> None:
        super().__init__(name="rubric_judge", threshold=threshold, **kwargs)
        self.llm = llm
        self.rubric = rubric or _DEFAULT_RUBRIC

    def measure(self, eval_case: EvalCase) -> Score:
        return asyncio.run(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        levels = sorted(self.rubric.keys())
        min_level, max_level = levels[0], levels[-1]

        rubric_text = "\n".join(f"  {k}: {v}" for k, v in sorted(self.rubric.items(), reverse=True))
        expected_section = (
            f"**Expected output**: {eval_case.expected}" if eval_case.expected else ""
        )

        prompt = _PROMPT_TEMPLATE.format(
            input=eval_case.input,
            output=eval_case.output,
            expected_section=expected_section,
            rubric_text=rubric_text,
            min_level=min_level,
            max_level=max_level,
        )

        result = await self.llm.generate_json(prompt, _RESPONSE_SCHEMA)
        level = int(result.get("level", min_level))
        level = max(min_level, min(max_level, level))
        reasoning = result.get("reasoning", "")

        # Normalize level to 0.0-1.0
        value = (level - min_level) / (max_level - min_level) if max_level > min_level else 1.0

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=reasoning,
            metadata={"level": level, "max_level": max_level},
        )
