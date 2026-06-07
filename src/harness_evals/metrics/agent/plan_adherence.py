"""PlanAdherence metric — LLM judges whether the agent followed its own plan."""

from __future__ import annotations

import json

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM

_PROMPT_TEMPLATE = """You are an adversarial plan adherence evaluator. Determine whether the agent exactly and exclusively followed its declared plan.

**Task (input)**:
{input}

**Agent messages (containing plan/reasoning)**:
{messages_text}

**Tool calls executed**:
{tool_calls_text}

You are evaluating plan obedience only — not success, correctness, or usefulness.

Evaluation rules:
1. **Step verification**: Every step in the plan must correspond to a verifiable action in the tool calls or messages. Missing or ambiguous steps count as not followed.
2. **No extraneous actions**: Any major action not in the plan lowers the score significantly.
3. **Order consistency**: Steps performed in a different order than planned must lower the score.
4. **Completeness**: If even one planned step is missing, the score must drop.

Scoring guide:
- 1.0: Perfect adherence — every planned step executed in correct order, no extra steps
- 0.75: Strong — nearly all steps in order, at most one minor deviation
- 0.5: Partial — some steps match but others missing, out of order, or replaced
- 0.25: Weak — only a few plan steps appear, multiple extraneous actions
- 0.0: No adherence — trace shows little resemblance to the plan

When in doubt, assign the lower score.

Respond with JSON:
{{"reasoning": "1-3 concise sentences citing specific matched, missing, or extra steps", "score": <float between 0.0 and 1.0>}}
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["reasoning", "score"],
    "properties": {
        "reasoning": {"type": "string"},
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


class PlanAdherenceMetric(BaseMetric):
    """LLM-judged evaluation of whether the agent followed its own plan.

    Reads ``eval_case.messages`` (for the declared plan) and
    ``eval_case.tool_calls`` (for actual execution) to compare
    planned steps vs actions taken.
    """

    def __init__(self, llm: BaseLLM, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(name="plan_adherence", dimension=Dimension.TRAJECTORY, threshold=threshold, **kwargs)
        self.llm = llm

    def measure(self, eval_case: EvalCase) -> Score:
        return _run_async(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        messages = eval_case.messages
        if not messages:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="Cannot evaluate plan adherence — no messages present in the eval case",
            )

        messages_text = "\n".join(f"[{msg.role}]: {msg.content or ''}" for msg in messages)

        tool_calls = eval_case.tool_calls or []
        tool_calls_text = (
            json.dumps([tc.to_dict() for tc in tool_calls], indent=2) if tool_calls else "No tool calls recorded"
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
                "n_messages": len(messages),
                "n_tool_calls": len(tool_calls),
            },
        )
