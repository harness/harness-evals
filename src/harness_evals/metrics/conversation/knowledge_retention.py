"""KnowledgeRetention metric — LLM judges whether the assistant retains info across turns."""

from __future__ import annotations

from harness_evals.core.metric import Dimension
from harness_evals.llm.base import BaseLLM
from harness_evals.metrics.conversation.llm_conversation_metric import (
    LLMConversationMetric,
)

_PROMPT_TEMPLATE = """You are an expert evaluator assessing knowledge retention in a multi-turn conversation.

**Conversation:**
{conversation_text}

Evaluate knowledge retention considering:
1. Does the assistant remember facts, preferences, and details shared by the user in earlier turns?
2. Does the assistant ask for information that was already provided?
3. Does the assistant contradict information established earlier in the conversation?
4. Does the assistant build upon previously shared context appropriately?

If the assistant forgets previously stated information, asks redundant questions about known facts, or contradicts earlier context, the score should be lowered significantly.

Respond with JSON:
{{"reasoning": "your analysis of knowledge retention, citing any instances of forgetting or contradiction", "score": <float between 0.0 and 1.0 where 1.0 means perfect retention and 0.0 means complete forgetting>}}
"""


class KnowledgeRetentionMetric(LLMConversationMetric):
    """LLM-judged evaluation of knowledge retention across conversation turns.

    Reads ``eval_case.messages`` and evaluates whether the assistant
    remembers and uses information from earlier turns. Returns 0.0
    if messages is missing or has fewer than 2 turns.
    """

    _prompt_template = _PROMPT_TEMPLATE

    def __init__(self, llm: BaseLLM, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(
            llm=llm, threshold=threshold, name="knowledge_retention", dimension=Dimension.CORRECTNESS, **kwargs
        )
