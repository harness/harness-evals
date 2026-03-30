"""ArgumentCorrectness metric — LLM judges whether tool call arguments are correct."""

from __future__ import annotations

import json

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM

_PROMPT_TEMPLATE = """You are an expert evaluator assessing whether an AI agent's tool call arguments are correct and relevant to the task.

**Task (input)**:
{input}

**Tool calls made by the agent**:
{tool_calls_text}

Evaluate each tool call's input parameters considering:
1. Are the arguments relevant to the original task?
2. Are the argument values correct and well-formed?
3. Are there missing required arguments or extraneous irrelevant arguments?
4. Would these arguments produce the intended result?

Respond with JSON:
{{"reasoning": "your analysis of argument correctness across all tool calls", "score": <float between 0.0 and 1.0 where 1.0 means all arguments are correct and relevant and 0.0 means none are>}}
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["reasoning", "score"],
    "properties": {
        "reasoning": {"type": "string"},
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


class ArgumentCorrectnessMetric(BaseMetric):
    """LLM-judged evaluation of whether tool call arguments are correct.

    Reads ``eval_case.tool_calls`` and evaluates whether each tool call's
    input parameters correctly and relevantly address the task described
    in ``eval_case.input``.

    Use this for **single-turn agent** tool argument evaluation.
    For **multi-turn conversational** tool evaluation (selection + arguments),
    see :class:`~harness_evals.metrics.conversation.tool_use.ToolUseMetric`.
    """

    def __init__(self, llm: BaseLLM, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(name="argument_correctness", threshold=threshold, **kwargs)
        self.llm = llm

    def measure(self, eval_case: EvalCase) -> Score:
        return _run_async(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        tool_calls = eval_case.tool_calls
        if not tool_calls:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="No tool_calls present to evaluate",
            )

        tool_calls_text = json.dumps([tc.to_dict() for tc in tool_calls], indent=2)
        prompt = _PROMPT_TEMPLATE.format(
            input=eval_case.input,
            tool_calls_text=tool_calls_text,
        )

        result = await self.llm.generate_json(prompt, _RESPONSE_SCHEMA)
        value = max(0.0, min(1.0, float(result.get("score", 0.0))))
        reasoning = result.get("reasoning", "")

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=reasoning,
            metadata={"n_tool_calls": len(tool_calls)},
        )
