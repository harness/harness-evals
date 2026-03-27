"""ConversationCoherence metric — LLM judges topical coherence across turns."""

from __future__ import annotations

from harness_evals.llm.base import BaseLLM
from harness_evals.metrics.conversation.llm_conversation_metric import (
    LLMConversationMetric,
)

_PROMPT_TEMPLATE = """You are an expert evaluator assessing the topical coherence of a multi-turn conversation.

**Conversation:**
{conversation_text}

Evaluate coherence considering:
1. Do responses stay on topic and address what was asked?
2. Is context maintained across turns — does the assistant remember earlier information?
3. Are there contradictions between turns?
4. Do topic transitions feel natural, or are they jarring?

Respond with JSON:
{{"reasoning": "your analysis of conversational coherence", "score": <float between 0.0 and 1.0 where 1.0 means perfectly coherent and 0.0 means completely incoherent>}}
"""


class ConversationCoherenceMetric(LLMConversationMetric):
    """LLM-judged topical coherence across conversation turns.

    Reads ``eval_case.messages`` — a list of ``Message`` objects representing
    ordered turns.  Returns 0.0 if messages is missing or has fewer than
    2 turns.
    """

    _prompt_template = _PROMPT_TEMPLATE

    def __init__(self, llm: BaseLLM, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(llm=llm, threshold=threshold, name="conversation_coherence", **kwargs)
