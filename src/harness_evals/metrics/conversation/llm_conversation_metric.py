"""Shared base class for LLM-judged conversation metrics."""

from __future__ import annotations

import logging

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM
from harness_evals.logging_config import truncate_repr

_logger = logging.getLogger(__name__)

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

    For per-turn scoring, subclasses set ``_per_turn = True`` and provide
    a ``_per_turn_prompt_template`` with ``{context}`` and ``{response}``
    placeholders. The aggregate score is the mean of per-turn scores, with
    the breakdown stored in ``score.metadata["turn_scores"]``.
    """

    _prompt_template: str = ""
    _per_turn: bool = False
    _per_turn_prompt_template: str = ""

    def __init__(
        self,
        llm: BaseLLM,
        threshold: float = 0.7,
        *,
        name: str,
        dimension: Dimension = Dimension.CORRECTNESS,
        **kwargs: object,
    ) -> None:
        super().__init__(name=name, dimension=dimension, threshold=threshold, **kwargs)
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

        if self._per_turn:
            return await self._measure_per_turn(messages)

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

    async def _measure_per_turn(self, messages: list) -> Score:
        turn_scores: list[dict] = []
        assistant_idx = 0

        for i, msg in enumerate(messages):
            if msg.role != "assistant":
                continue

            context = "\n".join(f"[{m.role}]: {m.content or ''}" for m in messages[:i])
            if not context:
                context = "(start of conversation)"

            prompt = self._per_turn_prompt_template.format(
                context=context,
                response=msg.content or "",
            )

            result = await self.llm.generate_json(prompt, _RESPONSE_SCHEMA)
            value = max(0.0, min(1.0, float(result.get("score", 0.0))))
            reasoning = result.get("reasoning", "")

            turn_scores.append(
                {
                    "turn": assistant_idx,
                    "message_index": i,
                    "score": value,
                    "reasoning": reasoning,
                }
            )
            _logger.debug(
                "LLM conversation metric %s turn %d: score=%.2f reasoning=%s",
                self.name,
                assistant_idx,
                value,
                truncate_repr(reasoning, max_len=120),
            )
            assistant_idx += 1

        if not turn_scores:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="no assistant turns found",
            )

        aggregate = sum(t["score"] for t in turn_scores) / len(turn_scores)
        reason = f"Mean of {len(turn_scores)} turn scores"
        if aggregate < self.threshold:
            details = "; ".join(
                f"turn {t['turn']}={t['score']:.2f}: {truncate_repr(t['reasoning'], max_len=80)}" for t in turn_scores
            )
            reason = f"{reason} | {details}"
        return Score(
            name=self.name,
            value=aggregate,
            threshold=self.threshold,
            reason=reason,
            metadata={
                "turn_scores": turn_scores,
                "n_turns": len(messages),
                "n_assistant_turns": len(turn_scores),
            },
        )
