"""ToolUse metric — LLM judges tool selection and argument correctness in a conversation."""

from __future__ import annotations

import json
from typing import Any

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.core.types import Message, ToolCall
from harness_evals.llm.base import BaseLLM

_PROMPT_TEMPLATE = """You are an expert evaluator assessing tool usage quality within a multi-turn conversation.

**Conversation:**
{conversation_text}

**Tool calls found in conversation**:
{tool_calls_text}

Evaluate tool usage considering:
1. **Tool selection**: Were the right tools chosen for each step? Were any unnecessary tools called?
2. **Argument correctness**: Were tool arguments correct, relevant, and well-formed?
3. **Sequencing**: Were tools called in a logical order?
4. **Completeness**: Were all necessary tools called to accomplish the task?

Long conversations and tool payloads may be shortened; truncation markers are not
themselves defects. Judge only the tool usage evidence that is present.

Respond with JSON:
{{"reasoning": "your analysis of tool selection quality and argument correctness", "score": <float between 0.0 and 1.0 where 1.0 means perfect tool usage and 0.0 means completely wrong>}}
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["reasoning", "score"],
    "properties": {
        "reasoning": {"type": "string"},
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}

# Prompt budget. Judge models have a hard context window, and agent conversations
# routinely carry tool results measured in hundreds of KB, so every section that
# grows with the trace is capped before it reaches the model.
_DEFAULT_MAX_PROMPT_CHARS = 100_000
_MAX_USER_CONTENT_CHARS = 4_000
_MAX_ASSISTANT_CONTENT_CHARS = 2_000
_MAX_TOOL_RESULT_CHARS = 400
# Arguments are what this metric judges, so they get a larger cap than results.
_MAX_TOOL_ARGS_CHARS = 600


class ToolUseMetric(BaseMetric):
    """LLM-judged evaluation of tool usage within a multi-turn conversation.

    Reads ``eval_case.messages`` (which may contain ``tool_calls`` on
    individual ``Message`` objects) to evaluate tool selection quality
    and argument correctness across the conversation.

    Message content and tool payloads are compacted to fit
    ``max_prompt_chars``; when the trace is still too large, middle lines are
    elided and the omission is reported in ``score.metadata``.

    Use this for **multi-turn conversational** tool evaluation.
    For **single-turn agent** tool argument evaluation, see
    :class:`~harness_evals.metrics.agent.argument_correctness.ArgumentCorrectnessMetric`.
    """

    def __init__(
        self,
        llm: BaseLLM,
        threshold: float = 0.7,
        max_prompt_chars: int = _DEFAULT_MAX_PROMPT_CHARS,
        **kwargs: object,
    ) -> None:
        super().__init__(name="tool_use", dimension=Dimension.TRAJECTORY, threshold=threshold, **kwargs)
        if max_prompt_chars < 1:
            raise ValueError(f"max_prompt_chars must be >= 1, got {max_prompt_chars}")
        self.llm = llm
        self.max_prompt_chars = max_prompt_chars

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

        # eval_case.tool_calls and message-level tool_calls are usually the
        # same events duplicated (e.g. the conversation simulator populates
        # both from SSE), so never merge them — that would double-count.
        # But some sources (e.g. the OTel/Langfuse trace adapter) only
        # promote a subset of calls to the top level while message-level
        # carries the fuller trajectory, so picking top-level unconditionally
        # can silently drop calls. Prefer whichever list is longer.
        message_tool_calls = [tc for msg in messages for tc in (msg.tool_calls or [])]
        top_level_tool_calls = list(eval_case.tool_calls or [])
        tool_calls = (
            top_level_tool_calls if len(top_level_tool_calls) >= len(message_tool_calls) else message_tool_calls
        )

        if not tool_calls:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="No tool calls found in conversation messages",
            )

        # Split the budget between the two sections that grow with the trace.
        section_budget = max(1, self.max_prompt_chars // 2)
        conversation_text, n_elided_messages = _fit_lines([_format_message(msg) for msg in messages], section_budget)
        tool_calls_text, n_elided_tool_calls = _fit_lines(
            [_format_tool_call(i, tc) for i, tc in enumerate(tool_calls, start=1)], section_budget
        )

        prompt = _PROMPT_TEMPLATE.format(
            conversation_text=conversation_text,
            tool_calls_text=tool_calls_text,
        )

        result = await self.llm.generate_json(prompt, _RESPONSE_SCHEMA)
        value = max(0.0, min(1.0, float(result.get("score", 0.0))))
        reasoning = result.get("reasoning", "")

        metadata: dict[str, Any] = {
            "n_turns": len(messages),
            "n_tool_calls": len(tool_calls),
            "judge_prompt_chars": len(prompt),
        }
        if n_elided_messages:
            metadata["judge_elided_messages"] = n_elided_messages
        if n_elided_tool_calls:
            metadata["judge_elided_tool_calls"] = n_elided_tool_calls

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=reasoning,
            metadata=metadata,
        )


def _format_message(msg: Message) -> str:
    content = _truncate(str(msg.content or ""), _max_content_chars_for_role(msg.role))
    return f"[{msg.role}]: {content}"


def _max_content_chars_for_role(role: str) -> int:
    if role == "user":
        return _MAX_USER_CONTENT_CHARS
    if role == "tool":
        return _MAX_TOOL_RESULT_CHARS
    return _MAX_ASSISTANT_CONTENT_CHARS


def _format_tool_call(index: int, tool_call: ToolCall) -> str:
    args = _truncate(_as_text(tool_call.input) if tool_call.input else "{}", _MAX_TOOL_ARGS_CHARS)
    line = f"{index}. {tool_call.name or 'unknown'} args={args}"
    if tool_call.output is not None:
        line += f" -> {_truncate(_as_text(tool_call.output), _MAX_TOOL_RESULT_CHARS)}"
    return line


def _as_text(payload: object) -> str:
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(payload)


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    if max_len <= 40:
        return text[:max_len]
    return text[: max_len - 35] + f"... [truncated, {len(text)} chars total]"


def _fit_lines(lines: list[str], max_chars: int) -> tuple[str, int]:
    """Join ``lines``, eliding the middle ones if the result exceeds ``max_chars``.

    Returns the joined text and the number of lines omitted. The start and end of
    a trace carry the task setup and the outcome, so both ends are preserved.
    """
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text, 0

    notice_budget = 120
    head_budget = int(max(0, max_chars - notice_budget) * 0.55)
    tail_budget = max(0, max_chars - notice_budget - head_budget)

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
    if not omitted:
        return "\n".join(head + tail), 0

    notice = f"... [omitted {omitted} middle lines to fit the {max_chars}-character judge budget]"
    return "\n".join([*head, notice, *tail]), omitted
