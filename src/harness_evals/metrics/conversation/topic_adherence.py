"""TopicAdherence metric — LLM judges whether the conversation stays within allowed topics."""

from __future__ import annotations

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM

_PROMPT_TEMPLATE = """You are an expert evaluator assessing whether a conversation stays within the allowed topics.

**Allowed topics**:
{allowed_topics}

**Conversation:**
{conversation_text}

Evaluate topic adherence considering:
1. For each user question, is it relevant to the allowed topics?
2. When a question IS relevant, does the assistant answer it correctly?
3. When a question is NOT relevant, does the assistant appropriately refuse to answer?
4. Does the assistant ever provide answers on off-topic subjects it should have refused?

Scoring:
- Answering relevant questions correctly is good (true positive)
- Refusing off-topic questions is good (true negative)
- Answering off-topic questions is bad (false positive)
- Refusing relevant questions is bad (false negative)

Respond with JSON:
{{"reasoning": "your analysis of topic adherence, citing specific on/off-topic exchanges", "score": <float between 0.0 and 1.0 where 1.0 means perfect topic adherence and 0.0 means complete failure>}}
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["reasoning", "score"],
    "properties": {
        "reasoning": {"type": "string"},
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


class TopicAdherenceMetric(BaseMetric):
    """LLM-judged evaluation of whether the conversation stays within allowed topics.

    Takes ``allowed_topics`` in the constructor and reads
    ``eval_case.messages`` for the conversation. Returns 0.0 if messages
    are missing or fewer than 2 turns.
    """

    def __init__(
        self,
        llm: BaseLLM,
        allowed_topics: list[str],
        threshold: float = 0.7,
        **kwargs: object,
    ) -> None:
        super().__init__(name="topic_adherence", threshold=threshold, **kwargs)
        self.llm = llm
        self.allowed_topics = allowed_topics

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
        prompt = _PROMPT_TEMPLATE.format(
            allowed_topics=", ".join(self.allowed_topics),
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
            metadata={
                "n_turns": len(messages),
                "allowed_topics": self.allowed_topics,
            },
        )
