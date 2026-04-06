"""RoleAdherence metric — LLM judges whether the assistant stays in its assigned role."""

from __future__ import annotations

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM

_PROMPT_TEMPLATE = """You are an expert evaluator assessing whether a chatbot adheres to its assigned role throughout a conversation.

**Assigned chatbot role**:
{chatbot_role}

**Conversation:**
{conversation_text}

Evaluate role adherence considering:
1. Does the assistant consistently behave according to its assigned role/persona?
2. Are there any responses where the assistant breaks character or acts contrary to the role?
3. Does the assistant maintain the expected tone, knowledge boundaries, and behavioral constraints of the role?
4. Are there any out-of-character responses that deviate from the role's nature?

Respond with JSON:
{{"reasoning": "your analysis of role adherence, citing any out-of-character responses", "score": <float between 0.0 and 1.0 where 1.0 means perfect role adherence and 0.0 means complete failure to maintain the role>}}
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["reasoning", "score"],
    "properties": {
        "reasoning": {"type": "string"},
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


class RoleAdherenceMetric(BaseMetric):
    """LLM-judged evaluation of whether the assistant stays in its assigned role.

    Reads ``eval_case.messages`` for the conversation and
    ``eval_case.metadata["chatbot_role"]`` for the expected role/persona.
    Returns 0.0 if messages are missing, fewer than 2, or no chatbot_role
    is provided in metadata.
    """

    def __init__(self, llm: BaseLLM, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(name="role_adherence", dimension=Dimension.TRAJECTORY, threshold=threshold, **kwargs)
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

        chatbot_role = eval_case.meta("chatbot_role")
        if not chatbot_role:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="chatbot_role missing from metadata",
            )

        conversation_text = "\n".join(f"[{msg.role}]: {msg.content or ''}" for msg in messages)
        prompt = _PROMPT_TEMPLATE.format(
            chatbot_role=chatbot_role,
            conversation_text=conversation_text,
        )

        result = await self.llm.generate_json(prompt, _RESPONSE_SCHEMA)
        value = max(0.0, min(1.0, float(result.get("score", 0.0))))
        reasoning = result.get("reasoning", "")

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=reasoning,
            metadata={"n_turns": len(messages), "chatbot_role": chatbot_role},
        )
