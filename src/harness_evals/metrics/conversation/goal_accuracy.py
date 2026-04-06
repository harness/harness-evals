"""GoalAccuracy metric — LLM judges whether the conversation achieved its goal."""

from __future__ import annotations

from harness_evals.core.metric import Dimension
from harness_evals.llm.base import BaseLLM
from harness_evals.metrics.conversation.llm_conversation_metric import (
    LLMConversationMetric,
)

_PROMPT_TEMPLATE = """You are an expert evaluator assessing whether a multi-turn conversation achieved its goal.

**Conversation:**
{conversation_text}

Evaluate goal accuracy considering:
1. What was the user's stated or clearly implied goal?
2. Did the assistant's visible output fully and correctly achieve that goal?
3. Is the information provided factually correct and directly relevant?
4. Does the response stand on its own without requiring follow-up clarification?

Scoring guide:
- 1.0: Goal completely and correctly achieved; all required outputs visible
- 0.75: Mostly achieved; minor omissions or trivial inaccuracies
- 0.5: Partially achieved; core goal addressed but key parts missing or incorrect
- 0.25: Weak attempt; loosely related but fails to satisfy the request
- 0.0: Goal not achieved at all; irrelevant, wrong, or missing answer

When in doubt, choose the lower score.

Respond with JSON:
{{"reasoning": "1-3 factual sentences explaining what parts of the goal were or were not achieved", "score": <float between 0.0 and 1.0>}}
"""


class GoalAccuracyMetric(LLMConversationMetric):
    """LLM-judged evaluation of whether the conversation achieved its goal.

    Reads ``eval_case.messages`` and evaluates whether the stated or
    inferred goal was met. Returns 0.0 if messages is missing or has
    fewer than 2 turns.
    """

    _prompt_template = _PROMPT_TEMPLATE

    def __init__(self, llm: BaseLLM, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(llm=llm, threshold=threshold, name="goal_accuracy", dimension=Dimension.CORRECTNESS, **kwargs)
