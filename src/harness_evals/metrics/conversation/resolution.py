"""ConversationResolution metric — LLM judges whether the conversation reached resolution."""

from __future__ import annotations

from harness_evals.core.metric import Dimension
from harness_evals.llm.base import BaseLLM
from harness_evals.metrics.conversation.llm_conversation_metric import (
    LLMConversationMetric,
)

_PROMPT_TEMPLATE = """You are an expert evaluator assessing whether a conversation reached a meaningful resolution.

**Conversation:**
{conversation_text}

Evaluate resolution considering:
1. Was the user's original question or need fully addressed?
2. Did the conversation end with a clear answer, solution, or agreed next step?
3. Were follow-up questions answered satisfactorily?
4. Is there unfinished business — open questions, unresolved ambiguity, or missing information?

Respond with JSON:
{{"reasoning": "your analysis of whether the conversation reached resolution", "score": <float between 0.0 and 1.0 where 1.0 means fully resolved and 0.0 means completely unresolved>}}
"""


class ConversationResolutionMetric(LLMConversationMetric):
    """LLM-judged evaluation of whether a conversation reached resolution.

    Reads ``eval_case.messages`` — a list of ``Message`` objects representing
    ordered turns.  Returns 0.0 if messages is missing or has fewer than
    2 turns.
    """

    _prompt_template = _PROMPT_TEMPLATE

    def __init__(self, llm: BaseLLM, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(
            llm=llm, threshold=threshold, name="conversation_resolution", dimension=Dimension.CORRECTNESS, **kwargs
        )
