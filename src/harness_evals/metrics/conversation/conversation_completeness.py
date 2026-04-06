"""ConversationCompleteness metric — LLM judges whether all user intents were addressed."""

from __future__ import annotations

from harness_evals.core.metric import Dimension
from harness_evals.llm.base import BaseLLM
from harness_evals.metrics.conversation.llm_conversation_metric import (
    LLMConversationMetric,
)

_PROMPT_TEMPLATE = """You are an expert evaluator assessing whether a multi-turn conversation addressed all user intents.

**Conversation:**
{conversation_text}

Evaluate completeness considering:
1. Identify all distinct user intents/requests across the conversation.
2. For each intent, was it fully addressed by the assistant?
3. Were any user questions left unanswered or partially answered?
4. Did the assistant acknowledge and follow up on all user needs?

Respond with JSON:
{{"reasoning": "your analysis of which user intents were or were not addressed", "score": <float between 0.0 and 1.0 where 1.0 means all intents fully addressed and 0.0 means none were>}}
"""


class ConversationCompletenessMetric(LLMConversationMetric):
    """LLM-judged evaluation of whether the conversation addressed all user intents.

    Reads ``eval_case.messages`` and evaluates whether each user intent
    was fully addressed by the assistant. Returns 0.0 if messages is
    missing or has fewer than 2 turns.
    """

    _prompt_template = _PROMPT_TEMPLATE

    def __init__(self, llm: BaseLLM, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(
            llm=llm, threshold=threshold, name="conversation_completeness", dimension=Dimension.CORRECTNESS, **kwargs
        )
