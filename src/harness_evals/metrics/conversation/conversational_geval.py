"""ConversationalGEval — GEval adapted for multi-turn conversations with per-turn scoring."""

from __future__ import annotations

from enum import Enum

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
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

_TURN_PROMPT = """You are an expert evaluator. Score the following assistant response in the context of a multi-turn conversation.

**Criteria**: {criteria}

{steps_section}

**Conversation context**:
{context}

**Assistant response to evaluate** (turn {turn_index}):
{response}

Evaluate ONLY the indicated assistant response against the criteria, considering the conversation context.

Respond with JSON:
{{"reasoning": "your analysis", "score": <float between 0.0 and 1.0>}}"""


class MultiTurnView(str, Enum):
    FULL_CONVERSATION = "full"
    SLIDING_WINDOW = "window"


class ConversationalGEvalMetric(BaseMetric):
    """GEval adapted for multi-turn conversations with per-turn scoring.

    Scores each assistant turn against the given criteria. Returns an aggregate
    score (mean of turn scores) with per-turn breakdown in metadata.
    """

    def __init__(
        self,
        llm: BaseLLM,
        criteria: str,
        *,
        evaluation_steps: list[str] | None = None,
        view: MultiTurnView = MultiTurnView.FULL_CONVERSATION,
        window_size: int = 5,
        threshold: float = 0.7,
        name: str = "conversational_geval",
        dimension: Dimension = Dimension.CORRECTNESS,
        **kwargs: object,
    ) -> None:
        super().__init__(name=name, dimension=dimension, threshold=threshold, **kwargs)
        self.llm = llm
        self.criteria = criteria
        self.evaluation_steps = evaluation_steps
        self.view = view
        self.window_size = window_size

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

        steps_section = ""
        if self.evaluation_steps:
            numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(self.evaluation_steps))
            steps_section = f"**Evaluation steps**:\n{numbered}"

        turn_scores: list[dict] = []
        assistant_turn_idx = 0

        for i, msg in enumerate(messages):
            if msg.role != "assistant":
                continue

            context = self._get_context(messages, i)
            prompt = _TURN_PROMPT.format(
                criteria=self.criteria,
                steps_section=steps_section,
                context=context,
                turn_index=assistant_turn_idx + 1,
                response=msg.content or "",
            )

            result = await self.llm.generate_json(prompt, _RESPONSE_SCHEMA)
            value = max(0.0, min(1.0, float(result.get("score", 0.0))))
            reasoning = result.get("reasoning", "")

            turn_scores.append(
                {
                    "turn": assistant_turn_idx,
                    "message_index": i,
                    "score": value,
                    "reasoning": reasoning,
                }
            )
            assistant_turn_idx += 1

        if not turn_scores:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="no assistant turns found",
            )

        aggregate = sum(t["score"] for t in turn_scores) / len(turn_scores)

        return Score(
            name=self.name,
            value=aggregate,
            threshold=self.threshold,
            reason=f"Mean of {len(turn_scores)} turn scores",
            metadata={
                "turn_scores": turn_scores,
                "n_turns": len(messages),
                "n_assistant_turns": len(turn_scores),
            },
        )

    def _get_context(self, messages: list, current_idx: int) -> str:
        if self.view == MultiTurnView.FULL_CONVERSATION:
            context_msgs = messages[:current_idx]
        else:
            start = max(0, current_idx - self.window_size * 2)
            context_msgs = messages[start:current_idx]

        if not context_msgs:
            return "(start of conversation)"

        return "\n".join(f"[{m.role}]: {m.content or ''}" for m in context_msgs)
