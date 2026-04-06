"""StepEfficiency metric — LLM judges whether agent steps were necessary or redundant."""

from __future__ import annotations

import json

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM

_PROMPT_TEMPLATE = """You are an efficiency auditor evaluating how economically an AI agent executed a task.

**Task (input)**:
{input}

**Agent messages**:
{messages_text}

**Tool calls executed**:
{tool_calls_text}

Evaluate efficiency — achieving the goal using the fewest, simplest, and most direct actions possible.

Strict evaluation rules:
1. **Zero-tolerance for unnecessary actions**: Every step must be strictly required. Superfluous, speculative, repetitive, or stylistic steps lower the score sharply.
2. **Minimal action principle**: The ideal execution performs the exact minimum number of steps needed.
3. **No speculation or enrichment**: Extra retrievals, style edits, or rephrasings reduce the score to <= 0.25.
4. **Directness**: Steps must follow a logically minimal sequence. Repetition or re-querying indicates inefficiency.
5. **Resource economy**: Using multiple LLM calls or tools when one would suffice must be penalized.

Scoring guide:
- 1.0: Perfectly efficient — only essential steps, each directly necessary
- 0.75: Strong — mostly minimal with one small redundant step
- 0.5: Moderate — noticeable extra steps or indirect methods
- 0.25: Low — multiple irrelevant or unjustified actions
- 0.0: Highly inefficient — verbose, exploratory, or wasteful execution

When uncertain, assign the lower score.

Respond with JSON:
{{"reasoning": "1-3 concise sentences describing where inefficiencies occurred", "score": <float between 0.0 and 1.0>}}
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["reasoning", "score"],
    "properties": {
        "reasoning": {"type": "string"},
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


class StepEfficiencyMetric(BaseMetric):
    """LLM-judged evaluation of whether agent steps were necessary or redundant.

    Reads ``eval_case.messages`` and ``eval_case.tool_calls`` to evaluate
    whether the agent executed the task efficiently without unnecessary
    or redundant steps.
    """

    def __init__(self, llm: BaseLLM, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(name="step_efficiency", dimension=Dimension.TRAJECTORY, threshold=threshold, **kwargs)
        self.llm = llm

    def measure(self, eval_case: EvalCase) -> Score:
        return _run_async(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        messages = eval_case.messages
        tool_calls = eval_case.tool_calls

        if not messages and not tool_calls:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="No messages or tool_calls present to evaluate step efficiency",
            )

        messages_text = (
            "\n".join(f"[{msg.role}]: {msg.content or ''}" for msg in messages) if messages else "No messages recorded"
        )

        tool_calls_list = tool_calls or []
        tool_calls_text = (
            json.dumps([tc.to_dict() for tc in tool_calls_list], indent=2)
            if tool_calls_list
            else "No tool calls recorded"
        )

        prompt = _PROMPT_TEMPLATE.format(
            input=eval_case.input,
            messages_text=messages_text,
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
            metadata={
                "n_messages": len(messages) if messages else 0,
                "n_tool_calls": len(tool_calls_list),
            },
        )
