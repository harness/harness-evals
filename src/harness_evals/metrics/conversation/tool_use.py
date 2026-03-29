"""ToolUse metric — LLM judges tool selection and argument correctness in a conversation."""

from __future__ import annotations

import json

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM

_PROMPT_TEMPLATE = """You are an expert evaluator assessing tool usage quality within a multi-turn conversation.

**Conversation:**
{conversation_text}

**Tool calls found in conversation**:
{tool_calls_text}

Evaluate tool usage considering:
1. **Tool selection**: Were the right tools chosen for each step? Were any unnecessary tools called?
2. **Argument correctness**: Were tool arguments correct, relevant, and well-formed?
3. **Sequencing**: Were tools called in a logical order?
4. **Completeness**: Were all necessary tools called to accomplish the task?

Respond with JSON:
{{"reasoning": "your analysis of tool selection quality and argument correctness", "score": <float between 0.0 and 1.0 where 1.0 means perfect tool usage and 0.0 means completely wrong>}}
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["reasoning", "score"],
    "properties": {
        "reasoning": {"type": "string"},
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


class ToolUseMetric(BaseMetric):
    """LLM-judged evaluation of tool usage within a multi-turn conversation.

    Reads ``eval_case.messages`` (which may contain ``tool_calls`` on
    individual ``Message`` objects) to evaluate tool selection quality
    and argument correctness across the conversation.

    Use this for **multi-turn conversational** tool evaluation.
    For **single-turn agent** tool argument evaluation, see
    :class:`~harness_evals.metrics.agent.argument_correctness.ArgumentCorrectnessMetric`.
    """

    def __init__(self, llm: BaseLLM, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(name="tool_use", threshold=threshold, **kwargs)
        self.llm = llm

    def measure(self, eval_case: EvalCase) -> Score:
        return _run_async(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        messages = eval_case.messages
        if not messages or len(messages) < 2:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="messages missing or has fewer than 2 turns",
            )

        # Collect tool calls from individual messages and top-level eval_case
        all_tool_calls = []
        for msg in messages:
            if msg.tool_calls:
                all_tool_calls.extend(msg.tool_calls)

        if eval_case.tool_calls:
            all_tool_calls.extend(eval_case.tool_calls)

        if not all_tool_calls:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="No tool calls found in conversation messages",
            )

        conversation_text = "\n".join(
            f"[{msg.role}]: {msg.content or ''}" for msg in messages
        )
        tool_calls_text = json.dumps(
            [tc.to_dict() for tc in all_tool_calls], indent=2
        )

        prompt = _PROMPT_TEMPLATE.format(
            conversation_text=conversation_text,
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
                "n_turns": len(messages),
                "n_tool_calls": len(all_tool_calls),
            },
        )
