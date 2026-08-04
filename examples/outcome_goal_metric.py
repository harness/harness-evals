"""Example custom metric: judge a conversation against its curated expected outcome.

The built-in ``goal_accuracy`` metric infers the goal only from the visible
conversation transcript — it never reads the golden's ``expected_outcome``. For
production-derived goldens we curate an environment-neutral expected outcome and
want the judge to score against it explicitly.

This metric reads:

* ``eval_case.messages`` — the full conversation trajectory, and
* the expected outcome, resolved from ``eval_case.metadata["expected_outcome"]``
  (populated by ``ConversationSimulator`` from ``golden.expected_outcome``) with
  a fallback to ``eval_case.expected``.

It asks the judge whether the assistant achieved that expected outcome, scoring
0.0-1.0. Because the outcome is environment-neutral (behavioural, not tied to a
specific production entity), it tolerates the eval environment having different
resources than production.

Config (note the ``params:`` wrapper is only needed for extra kwargs)::

    plugins:
      - examples.outcome_goal_metric
    metrics:
      - {kind: outcome_goal_accuracy, threshold: 0.7}

    judge_llm: {provider: openai, name: gpt-4o}
"""

from __future__ import annotations

import json
import logging
from typing import Any

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.core.types import Message
from harness_evals.llm.base import BaseLLM
from harness_evals.plugins import register_metric

_logger = logging.getLogger(__name__)

_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["reasoning", "score"],
    "properties": {
        "reasoning": {"type": "string"},
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}

_PROMPT_TEMPLATE = """You are an expert evaluator judging whether a multi-turn \
conversation achieved a specific expected outcome.

**Expected outcome (what success looks like):**
{expected_outcome}

**Conversation (includes elicitation questions, simulated answers, and tool calls):**
{conversation_text}

Judge how well the assistant's behaviour achieved the expected outcome.
Elicitation turns often have empty assistant message content — treat pending
questions, simulated user answers, and tool request/result events as visible
progress. The expected outcome is deliberately environment-neutral: score the
assistant's behaviour and task completion, not whether a specific pre-existing
production resource happened to be present.

Path tolerance (critical):
- Intent matchers cannot cover every question wording. When they miss, an LLM
  user-simulator answers from the agent's offered options; those answers can
  vary (e.g. AWS Account ID vs Environment buckets).
- Do NOT penalize a valid alternate workflow that still achieves the core goal
  (create/update/list/diagnose as stated). Preferring one bucketing dimension,
  filter type, or option label in the expected outcome is guidance, not a hard
  requirement.
- Penalize only when simulated answers are absurd/incoherent relative to the
  question and options, the assistant abandons the task, or the core goal
  clearly fails without an honest blocking error.

Scoring guide:
- 1.0: Core goal achieved (alternate valid paths OK)
- 0.75: Mostly achieved; minor omissions that do not block the goal
- 0.5: Partially achieved; core intent addressed but key completion missing
- 0.25: Weak attempt; loosely related but fails the outcome
- 0.0: Outcome not achieved at all, or answers/actions are absurd/incorrect

When the only uncertainty is which valid path was taken, choose the higher score.

Respond with JSON:
{{"reasoning": "1-3 factual sentences about what was or was not achieved", \
"score": <float between 0.0 and 1.0>}}
"""


@register_metric("outcome_goal_accuracy")
class OutcomeGoalAccuracyMetric(BaseMetric):
    """LLM judge that scores the conversation against the curated expected outcome.

    Unlike ``goal_accuracy``, this metric injects ``expected_outcome`` (from
    golden metadata, falling back to ``eval_case.expected``) into the judge
    prompt. Returns 0.0 when the conversation is missing or too short, and 0.0
    with an explanatory reason when no expected outcome is available.
    """

    def __init__(self, llm: BaseLLM, threshold: float = 0.7, **kwargs: object) -> None:
        super().__init__(
            name="outcome_goal_accuracy",
            dimension=Dimension.CORRECTNESS,
            threshold=threshold,
            **kwargs,
        )
        self.llm = llm

    def measure(self, eval_case: EvalCase) -> Score:
        from harness_evals._async_compat import _run_async

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

        expected_outcome = self._resolve_expected_outcome(eval_case)
        if not expected_outcome:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="no expected_outcome available on the golden",
            )

        conversation_text = _format_conversation(messages, eval_case.metadata or {})
        prompt = _PROMPT_TEMPLATE.format(
            expected_outcome=expected_outcome,
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
            metadata={"n_turns": len(messages), "expected_outcome": expected_outcome},
        )

    @staticmethod
    def _resolve_expected_outcome(eval_case: EvalCase) -> str:
        metadata = eval_case.metadata or {}
        outcome = metadata.get("expected_outcome")
        if isinstance(outcome, str) and outcome.strip():
            return outcome.strip()
        if isinstance(eval_case.expected, str) and eval_case.expected.strip():
            return eval_case.expected.strip()
        return ""


def _format_conversation(messages: list[Message], metadata: dict[str, Any]) -> str:
    lines: list[str] = []
    for msg in messages:
        lines.append(_format_message(msg))

    tool_lines = _format_tool_events(metadata)
    if tool_lines:
        lines.append("")
        lines.append("[tool_events]:")
        lines.extend(tool_lines)

    error = metadata.get("elicitation_error")
    if error:
        lines.append("")
        lines.append(f"[elicitation_error]: {error}")

    return "\n".join(lines)


def _format_message(msg: Message) -> str:
    content = (msg.content or "").strip()
    meta = msg.metadata or {}
    extras: list[str] = []

    pending = meta.get("pending_human_input") or meta.get("pending_elicitation")
    if isinstance(pending, dict):
        extras.append(_format_pending(pending))

    if meta.get("simulated"):
        elicitation_type = meta.get("elicitation_type") or "simulated"
        extras.append(f"simulated_answer type={elicitation_type}")

    if content and extras:
        return f"[{msg.role}]: {content}\n  " + "\n  ".join(extras)
    if content:
        return f"[{msg.role}]: {content}"
    if extras:
        return f"[{msg.role}]: (no text content)\n  " + "\n  ".join(extras)
    return f"[{msg.role}]: (empty)"


def _format_pending(pending: dict[str, Any]) -> str:
    pending_type = pending.get("type") or "elicitation"
    payload = pending.get("payload") if isinstance(pending.get("payload"), dict) else {}
    content = payload.get("content") if isinstance(payload.get("content"), dict) else {}

    if pending_type == "elicitation_form":
        fields = content.get("fields") or []
        rendered = []
        for field in fields:
            if not isinstance(field, dict):
                continue
            label = field.get("label") or field.get("key") or "field"
            field_type = field.get("type") or "text"
            options = field.get("options") or []
            if options:
                labels = [_option_label(opt) for opt in options]
                rendered.append(f"{label} ({field_type}: {', '.join(labels)})")
            else:
                rendered.append(f"{label} ({field_type})")
        detail = "; ".join(rendered) if rendered else "(no fields)"
        return f"pending={pending_type}: {detail}"

    question = (
        content.get("question")
        or content.get("label")
        or payload.get("title")
        or payload.get("subtitle")
        or ""
    )
    options = content.get("options") or content.get("items") or content.get("choices") or []
    if options:
        labels = [_option_label(opt) for opt in options]
        return f"pending={pending_type}: {question} options=[{', '.join(labels)}]"
    if question:
        return f"pending={pending_type}: {question}"
    return f"pending={pending_type}"


def _option_label(option: object) -> str:
    if isinstance(option, dict):
        return str(option.get("label") or option.get("value") or option.get("id") or "")
    return str(option)


def _format_tool_events(metadata: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    timeline = metadata.get("sse_timeline")
    if isinstance(timeline, list) and timeline:
        for entry in timeline:
            if not isinstance(entry, dict):
                continue
            event = entry.get("event")
            if event not in {"assistant_tool_request", "assistant_tool_result"}:
                continue
            payload = entry.get("payload")
            lines.append(f"- {event}: {_compact_tool_payload(payload)}")
        if lines:
            return lines

    events = metadata.get("sse_events")
    if not isinstance(events, dict):
        return lines
    for event_name in ("assistant_tool_request", "assistant_tool_result"):
        for payload in events.get(event_name) or []:
            lines.append(f"- {event_name}: {_compact_tool_payload(payload)}")
    return lines


def _compact_tool_payload(payload: object) -> str:
    if payload is None:
        return "(empty)"
    if isinstance(payload, str):
        return payload[:300]
    try:
        text = json.dumps(payload, ensure_ascii=False)
    except TypeError:
        text = str(payload)
    return text if len(text) <= 400 else text[:397] + "..."
