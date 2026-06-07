"""PlanQuality metric — LLM judges the quality of an agent's plan/strategy."""

from __future__ import annotations

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM

_PROMPT_TEMPLATE = """You are an expert plan quality evaluator. Assess the quality of an AI agent's plan to accomplish the given task.

**Task (input)**:
{input}

**Agent messages (containing plan/strategy)**:
{messages_text}

Evaluate the plan considering:
1. **Completeness**: Does the plan fully address all requirements of the task? Missing critical subtasks should reduce the score sharply.
2. **Logical coherence**: Are steps in a clear, rational sequence? Disordered or redundant steps are penalized.
3. **Optimality**: Is the plan minimal but sufficient — no unnecessary or repetitive steps?
4. **Level of detail**: Are steps specific enough to execute without ambiguity?
5. **Alignment**: Does the plan directly target the user's stated goal?

Scoring guide:
- 1.0: Excellent — fully complete, logically ordered, optimally efficient
- 0.75: Good — covers nearly all aspects, minor gaps
- 0.5: Adequate but flawed — partially complete, some ambiguity
- 0.25: Weak — major missing steps or unclear logic
- 0.0: Inadequate — irrelevant, incoherent, or severely incomplete

When in doubt, assign the lower score.

Respond with JSON:
{{"reasoning": "1-3 sentences explaining what the plan lacks or how it could fail", "score": <float between 0.0 and 1.0>}}
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["reasoning", "score"],
    "properties": {
        "reasoning": {"type": "string"},
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


class PlanQualityMetric(BaseMetric):
    """LLM-judged evaluation of an agent's plan quality.

    Reads ``eval_case.messages`` to extract the agent's planning steps and
    evaluates completeness, logical ordering, optimality, detail, and
    alignment with the task.
    """

    def __init__(self, llm: BaseLLM, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(name="plan_quality", dimension=Dimension.TRAJECTORY, threshold=threshold, **kwargs)
        self.llm = llm

    def measure(self, eval_case: EvalCase) -> Score:
        return _run_async(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        messages = eval_case.messages
        if not messages:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="Cannot evaluate plan quality — no messages present in the eval case",
            )

        messages_text = "\n".join(f"[{msg.role}]: {msg.content or ''}" for msg in messages)
        prompt = _PROMPT_TEMPLATE.format(
            input=eval_case.input,
            messages_text=messages_text,
        )

        result = await self.llm.generate_json(prompt, _RESPONSE_SCHEMA)
        value = max(0.0, min(1.0, float(result.get("score", 0.0))))
        reasoning = result.get("reasoning", "")

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=reasoning,
            metadata={"n_messages": len(messages)},
        )
