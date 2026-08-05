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
      - kind: outcome_goal_accuracy
        threshold: 0.7
        params:
          max_conversation_chars: 100000

    judge_llm: {provider: openai, name: gpt-4o}

Long agent runs embed full MCP payloads in ``tool``-role messages. The metric
compacts tool results and caps total judge text (default 100k characters) so
128k-token models are not exceeded.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.core.types import Message, ToolCall
from harness_evals.llm.base import BaseLLM
from harness_evals.plugins import register_metric

_logger = logging.getLogger(__name__)

# Keep the judge prompt well under typical 128k-token model limits. Long agent runs
# embed full MCP payloads in tool-role messages; compact those before LLM judging.
_DEFAULT_MAX_CONVERSATION_CHARS = 100_000
_MAX_USER_CONTENT_CHARS = 4_000
_MAX_ASSISTANT_CONTENT_CHARS = 2_000
_MAX_TOOL_RESULT_CHARS = 400
_MAX_TOOL_ARGS_CHARS = 200

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

    def __init__(
        self,
        llm: BaseLLM,
        threshold: float = 0.7,
        *,
        max_conversation_chars: int = _DEFAULT_MAX_CONVERSATION_CHARS,
        **kwargs: object,
    ) -> None:
        super().__init__(
            name="outcome_goal_accuracy",
            dimension=Dimension.CORRECTNESS,
            threshold=threshold,
            **kwargs,
        )
        self.llm = llm
        if max_conversation_chars <= 0:
            raise ValueError(f"max_conversation_chars must be positive, got {max_conversation_chars}")
        self.max_conversation_chars = max_conversation_chars

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

        conversation_text, judge_meta = _format_conversation(
            messages,
            eval_case.metadata or {},
            max_chars=self.max_conversation_chars,
        )
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
            metadata={
                "n_turns": len(messages),
                "expected_outcome": expected_outcome,
                **judge_meta,
            },
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


def _format_conversation(
    messages: list[Message],
    metadata: dict[str, Any],
    *,
    max_chars: int = _DEFAULT_MAX_CONVERSATION_CHARS,
) -> tuple[str, dict[str, Any]]:
    lines: list[str] = []
    skipped_empty = 0
    for msg in messages:
        formatted = _format_message(msg)
        if formatted is None:
            skipped_empty += 1
            continue
        lines.append(formatted)

    has_expanded_tool_trace = any(
        msg.role == "tool" or (msg.role == "assistant" and msg.tool_calls) for msg in messages
    )
    if not has_expanded_tool_trace:
        tool_lines = _format_tool_events(metadata)
        if tool_lines:
            lines.append("")
            lines.append("[tool_events]:")
            lines.extend(tool_lines)

    error = metadata.get("elicitation_error")
    if error:
        lines.append("")
        lines.append(f"[elicitation_error]: {error}")

    text, truncated = _fit_conversation_to_budget(lines, max_chars)
    judge_meta = {
        "judge_conversation_chars": len(text),
        "judge_skipped_empty_messages": skipped_empty,
        "judge_conversation_truncated": truncated,
    }
    return text, judge_meta


def _format_message(msg: Message) -> str | None:
    meta = msg.metadata or {}
    extras: list[str] = []

    pending = meta.get("pending_human_input") or meta.get("pending_elicitation")
    if isinstance(pending, dict):
        extras.append(_format_pending(pending))

    if meta.get("simulated"):
        elicitation_type = meta.get("elicitation_type") or "simulated"
        extras.append(f"simulated_answer type={elicitation_type}")

    if msg.tool_calls:
        tool_lines = []
        for call in msg.tool_calls:
            effective = call
            if msg.role == "tool" and call.output is None and (msg.content or "").strip():
                effective = ToolCall(name=call.name, input=call.input, output=msg.content)
            tool_lines.append(_format_tool_call_line(msg.role, effective))
        tool_block = "\n  ".join(tool_lines)
        if msg.role == "tool" and not extras:
            return tool_block
        prefix = f"[{msg.role}]"
        if extras:
            return f"{prefix}:\n  {tool_block}\n  " + "\n  ".join(extras)
        return f"{prefix}:\n  {tool_block}"

    content = _truncate_text(
        (msg.content or "").strip(),
        _max_content_chars_for_role(msg.role),
    )

    if not content and not extras:
        return None
    if content and extras:
        return f"[{msg.role}]: {content}\n  " + "\n  ".join(extras)
    if content:
        return f"[{msg.role}]: {content}"
    return f"[{msg.role}]: (no text content)\n  " + "\n  ".join(extras)


def _max_content_chars_for_role(role: str) -> int:
    if role == "user":
        return _MAX_USER_CONTENT_CHARS
    if role == "tool":
        return _MAX_TOOL_RESULT_CHARS
    return _MAX_ASSISTANT_CONTENT_CHARS


def _format_tool_call_line(role: str, tool_call: ToolCall) -> str:
    name = tool_call.name or "unknown"
    args = tool_call.input if isinstance(tool_call.input, dict) else {}
    resource_type = args.get("resource_type")
    label = f"{name}(resource_type={resource_type})" if resource_type else name

    if role == "tool" or tool_call.output is not None:
        output = tool_call.output
        if output is None and role == "tool":
            output = ""
        if isinstance(output, dict):
            output_text = _summarize_tool_output(output)
        else:
            output_text = str(output or "")
        output_text = _truncate_text(output_text, _MAX_TOOL_RESULT_CHARS)
        return f"tool_result {label} → {output_text}"

    args_text = json.dumps(args, ensure_ascii=False) if args else "{}"
    args_text = _truncate_text(args_text, _MAX_TOOL_ARGS_CHARS)
    return f"tool_request {label} args={args_text}"


def _summarize_tool_output(output: dict[str, Any]) -> str:
    for key in ("message", "error", "status", "code"):
        value = output.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    items = output.get("items")
    if isinstance(items, list):
        preview = json.dumps(items[:3], ensure_ascii=False)
        suffix = f", …+{len(items) - 3} more" if len(items) > 3 else ""
        return f"items[{len(items)}]={preview}{suffix}"
    return json.dumps(output, ensure_ascii=False)


def _truncate_text(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    if max_len <= 40:
        return text[:max_len]
    return text[: max_len - 35] + f"... [truncated, {len(text)} chars total]"


def _fit_conversation_to_budget(lines: list[str], max_chars: int) -> tuple[str, bool]:
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text, False

    head_budget = int(max_chars * 0.55)
    tail_budget = max_chars - head_budget - 80
    head: list[str] = []
    used = 0
    for line in lines:
        extra = len(line) + (1 if head else 0)
        if used + extra > head_budget:
            break
        head.append(line)
        used += extra

    tail: list[str] = []
    used = 0
    for line in reversed(lines):
        extra = len(line) + (1 if tail else 0)
        if used + extra > tail_budget:
            break
        tail.insert(0, line)
        used += extra

    omitted = max(0, len(lines) - len(head) - len(tail))
    notice = (
        f"[conversation_truncated]: omitted {omitted} middle lines to fit the "
        f"{max_chars}-character judge budget; tool results were compacted"
    )
    compact = "\n".join([*head, notice, *tail])
    if len(compact) > max_chars:
        compact = _truncate_text(compact, max_chars)
    return compact, True


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
