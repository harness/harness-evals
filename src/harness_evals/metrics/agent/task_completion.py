"""TaskCompletion metric — LLM judges whether the agent completed the requested task."""

from __future__ import annotations

import asyncio

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM

_PROMPT_TEMPLATE = """You are an expert evaluator assessing whether an AI agent completed a requested task.

**Task (input)**:
{input}

**Agent output**:
{output}

{expected_section}

Evaluate task completion considering:
1. Did the agent address the core request?
2. Is the output complete, or only partially done?
3. Are there missing steps, incomplete sections, or placeholder content?
4. Did the agent refuse or fail to perform the task?

Respond with JSON:
{{"reasoning": "your analysis of task completion", "score": <float between 0.0 and 1.0 where 1.0 means fully completed and 0.0 means not attempted>}}
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["reasoning", "score"],
    "properties": {
        "reasoning": {"type": "string"},
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


class TaskCompletionMetric(BaseMetric):
    """LLM-judged evaluation of whether the agent completed the requested task.

    Considers partial completion — a task that is 80% done scores 0.8,
    not 0.0. Useful for complex multi-step agent tasks where binary
    pass/fail is too coarse.
    """

    def __init__(self, llm: BaseLLM, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(name="task_completion", threshold=threshold, **kwargs)
        self.llm = llm

    def measure(self, eval_case: EvalCase) -> Score:
        return asyncio.run(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        expected_section = (
            f"**Expected output (reference)**:\n{eval_case.expected}"
            if eval_case.expected
            else ""
        )
        prompt = _PROMPT_TEMPLATE.format(
            input=eval_case.input,
            output=eval_case.output,
            expected_section=expected_section,
        )

        result = await self.llm.generate_json(prompt, _RESPONSE_SCHEMA)
        value = float(result.get("score", 0.0))
        value = max(0.0, min(1.0, value))
        reasoning = result.get("reasoning", "")

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=reasoning,
        )
