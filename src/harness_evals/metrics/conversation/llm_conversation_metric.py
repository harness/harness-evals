"""Shared base class for LLM-judged conversation metrics."""

from __future__ import annotations

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM

_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["reasoning", "score"],
    "properties": {
        "reasoning": {"type": "string"},
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


class LLMConversationMetric(BaseMetric):
    """Base for LLM-judged metrics that read ``eval_case.messages``.

    Subclasses only need to set ``name`` and provide a ``_prompt_template``
    class attribute containing a ``{conversation_text}`` placeholder.
    """

    _prompt_template: str = ""

    def __init__(self, llm: BaseLLM, threshold: float = 0.7, *, name: str, **kwargs: object) -> None:
        super().__init__(name=name, threshold=threshold, **kwargs)
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

        conversation_text = "\n".join(f"[{msg.role}]: {msg.content or ''}" for msg in messages)
        prompt = self._prompt_template.format(conversation_text=conversation_text)

        result = await self.llm.generate_json(prompt, _RESPONSE_SCHEMA)
        value = max(0.0, min(1.0, float(result.get("score", 0.0))))
        reasoning = result.get("reasoning", "")

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=reasoning,
            metadata={"n_turns": len(messages)},
        )
