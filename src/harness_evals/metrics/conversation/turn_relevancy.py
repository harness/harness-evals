"""TurnRelevancy metric — LLM judges whether each assistant response is relevant."""

from __future__ import annotations

from harness_evals.llm.base import BaseLLM
from harness_evals.metrics.conversation.llm_conversation_metric import (
    LLMConversationMetric,
)

_PROMPT_TEMPLATE = """You are an expert evaluator assessing the relevancy of assistant responses in a multi-turn conversation.

**Conversation:**
{conversation_text}

Evaluate turn relevancy considering:
1. Is each assistant response relevant to the immediately preceding user message?
2. Does each response address what was actually asked, rather than going off-topic?
3. Are there any assistant responses that are completely irrelevant to the context?
4. Do responses maintain relevance to the broader conversation context?

Note: Vague responses to vague inputs (such as greetings) do NOT count as irrelevant.
Only penalize responses that are COMPLETELY irrelevant to the context.

Respond with JSON:
{{"reasoning": "your analysis of per-turn relevancy, citing any irrelevant responses", "score": <float between 0.0 and 1.0 where 1.0 means all responses are relevant and 0.0 means none are>}}
"""


class TurnRelevancyMetric(LLMConversationMetric):
    """LLM-judged evaluation of per-turn relevancy in a conversation.

    Reads ``eval_case.messages`` and evaluates whether each assistant
    response is relevant to the preceding context. Returns 0.0 if
    messages is missing or has fewer than 2 turns.
    """

    _prompt_template = _PROMPT_TEMPLATE

    def __init__(self, llm: BaseLLM, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(llm=llm, threshold=threshold, name="turn_relevancy", **kwargs)
