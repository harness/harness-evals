"""ConversationResolution metric — LLM judges whether the conversation reached resolution."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from harness_evals._async_compat import _run_async
from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import BaseMetric, Dimension
from harness_evals.core.score import Score
from harness_evals.core.types import Message, ToolCall
from harness_evals.llm.base import BaseLLM

_STATUS_VALUES = frozenset(
    {
        "resolved",
        "waiting_user",
        "partial",
        "unsatisfactory",
        "blocked_error",
        "abandoned",
        "insufficient_evidence",
    }
)

# Score mapped from status so ``passed`` stays ``value >= threshold``.
_STATUS_SCORE: dict[str, float] = {
    "resolved": 1.0,
    "waiting_user": 0.85,
    "partial": 0.4,
    "unsatisfactory": 0.25,
    "blocked_error": 0.1,
    "abandoned": 0.0,
    "insufficient_evidence": 1.0,  # skipped — do not count as unresolved
}

_WAITING_USER_RE = re.compile(
    r"waiting for user to review"
    r"|please (approve|confirm|review)"
    r"|approve to continue"
    r"|review gate"
    r"|askuserquestion",
    re.IGNORECASE,
)

_PROMPT_TEMPLATE = """You are an expert evaluator assessing whether a multi-turn agent conversation reached a meaningful resolution.

{expected_outcome_section}\
**Conversation:**
{conversation_text}

**Tool evidence (name, args, results — use this to decide created vs errored):**
{tool_evidence_text}

Assign exactly one status:

- `resolved`: the latest user ask was satisfactorily addressed with a clear answer, completed action, or agreed next step the user can act on.
- `waiting_user`: the agent is correctly blocked on the user — AskUserQuestion, HITL/review-gate approval, or an explicit required clarification. This is resolution *in progress*, not failure.
- `partial`: tools and/or reply made some progress but the ask is only partly answered (missing key details, unfinished multi-part request).
- `unsatisfactory`: tools may have succeeded, but the final user-visible answer does **not** satisfactorily address the ask (wrong, evasive, off-topic, or contradicts tool evidence).
- `blocked_error`: a tool/API/runtime failure (or equivalent hard error) left the ask unresolved, and the agent did not recover or provide a clear unblock path.
- `abandoned`: the agent stopped without answering and without a clear next step for the user.
- `insufficient_evidence`: the transcript is too thin or corrupted to judge.

Rules:
1. Tool success alone is NOT resolution — the final answer must still address the ask (`unsatisfactory` if it does not).
2. Tool failure alone is NOT automatic failure — if the agent still solves the ask another way, or clearly explains the blocker and what the user must do next, use `resolved` or `waiting_user`.
3. Unresolved after tool failure (no recovery, no clear next step) → `blocked_error`.
4. HITL / AskUserQuestion / "waiting for user to review" → `waiting_user`, not `abandoned`.
5. Prefer evidence in tool results/args over the assistant's prose when they conflict.

Respond with JSON:
{{"reasoning": "brief analysis tied to the statuses above", "status": "<one status>", "score": <float 0.0-1.0 consistent with status>}}
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["reasoning", "status", "score"],
    "properties": {
        "reasoning": {"type": "string"},
        "status": {
            "type": "string",
            "enum": sorted(_STATUS_VALUES),
        },
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}

_MAX_TOOL_ARGS_CHARS = 600
_MAX_TOOL_RESULT_CHARS = 400
_MAX_MSG_CHARS = 4_000


class ConversationResolutionMetric(BaseMetric):
    """LLM-judged evaluation of whether a conversation reached resolution.

    Returns a structured ``status`` in ``score.metadata``:

    - pass (default): ``resolved``, ``waiting_user``
    - fail: ``blocked_error``, ``abandoned``, ``unsatisfactory``
    - ``partial``: fail when ``fail_partial=True`` (default), else soft-pass
    - ``insufficient_evidence``: skipped (value 1.0, ``metadata.skipped=True``)
      for empty/short transcripts — not counted as unresolved.
    """

    def __init__(
        self,
        llm: BaseLLM,
        threshold: float = 0.7,
        *,
        fail_partial: bool = True,
        use_waiting_user_heuristic: bool = True,
        pass_statuses: Iterable[str] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(
            name="conversation_resolution",
            threshold=threshold,
            dimension=Dimension.CORRECTNESS,
            **kwargs,
        )
        self.llm = llm
        self.fail_partial = fail_partial
        self.use_waiting_user_heuristic = use_waiting_user_heuristic
        self.pass_statuses = frozenset(pass_statuses or ("resolved", "waiting_user"))

    def measure(self, eval_case: EvalCase) -> Score:
        return _run_async(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        messages = eval_case.messages
        if not messages or len(messages) < 2:
            return Score(
                name=self.name,
                value=1.0,
                threshold=self.threshold,
                reason="insufficient_evidence: messages missing or fewer than 2 turns",
                metadata={
                    "status": "insufficient_evidence",
                    "skipped": True,
                    "n_turns": len(messages or []),
                },
            )

        tool_calls = _collect_tool_calls(eval_case)
        if self.use_waiting_user_heuristic and _looks_like_waiting_user(messages, tool_calls):
            return Score(
                name=self.name,
                value=_STATUS_SCORE["waiting_user"],
                threshold=self.threshold,
                reason="waiting_user heuristic: last turn asks the user / HITL review gate",
                metadata={
                    "status": "waiting_user",
                    "heuristic": True,
                    "n_turns": len(messages),
                    "n_tool_calls": len(tool_calls),
                },
            )

        expected_outcome_section = ""
        expected_outcome = eval_case.meta("expected_outcome")
        if expected_outcome:
            expected_outcome_section = (
                f"**Expected outcome:**\n{expected_outcome}\n\n"
                "Judge the conversation against this expected outcome when scoring.\n\n"
            )

        conversation_text = "\n".join(
            f"[{msg.role}]: {_truncate(str(msg.content or ''), _MAX_MSG_CHARS)}" for msg in messages
        )
        tool_evidence_text = _format_tool_evidence(tool_calls)
        prompt = _PROMPT_TEMPLATE.format(
            expected_outcome_section=expected_outcome_section,
            conversation_text=conversation_text,
            tool_evidence_text=tool_evidence_text,
        )

        result = await self.llm.generate_json(prompt, _RESPONSE_SCHEMA)
        status = _normalize_status(result.get("status"))
        reasoning = str(result.get("reasoning") or "")
        llm_score = result.get("score")

        value = _score_for_status(status, fail_partial=self.fail_partial)
        # Prefer status-derived score; keep LLM score only as metadata.
        if status == "insufficient_evidence":
            return Score(
                name=self.name,
                value=1.0,
                threshold=self.threshold,
                reason=reasoning or "insufficient_evidence",
                metadata={
                    "status": status,
                    "skipped": True,
                    "llm_score": llm_score,
                    "n_turns": len(messages),
                    "n_tool_calls": len(tool_calls),
                },
            )

        # Soft-pass for partial when configured.
        if status == "partial" and not self.fail_partial:
            value = max(value, self.threshold)

        # If judge status is in pass_statuses, ensure value clears threshold.
        if status in self.pass_statuses:
            value = max(value, self.threshold)

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=reasoning or f"status={status}",
            metadata={
                "status": status,
                "llm_score": llm_score,
                "fail_partial": self.fail_partial,
                "n_turns": len(messages),
                "n_tool_calls": len(tool_calls),
            },
        )


def _normalize_status(raw: object) -> str:
    status = str(raw or "").strip().lower()
    if status in _STATUS_VALUES:
        return status
    return "abandoned"


def _score_for_status(status: str, *, fail_partial: bool) -> float:
    if status == "partial" and not fail_partial:
        return 0.75
    return _STATUS_SCORE.get(status, 0.0)


def _collect_tool_calls(eval_case: EvalCase) -> list[ToolCall]:
    message_tool_calls = [
        tc for msg in (eval_case.messages or []) for tc in (msg.tool_calls or [])
    ]
    top_level = list(eval_case.tool_calls or [])
    return top_level if len(top_level) >= len(message_tool_calls) else message_tool_calls


def _looks_like_waiting_user(messages: list[Message], tool_calls: list[ToolCall]) -> bool:
    if tool_calls:
        last_name = (tool_calls[-1].name or "").lower()
        if last_name in {"askuserquestion", "present_form"} or "askuser" in last_name:
            return True

    for msg in reversed(messages):
        if msg.role != "assistant":
            continue
        content = msg.content or ""
        if _WAITING_USER_RE.search(content):
            return True
        for tc in msg.tool_calls or []:
            name = (tc.name or "").lower()
            if name in {"askuserquestion", "present_form"} or "askuser" in name:
                return True
        break
    return False


def _format_tool_evidence(tool_calls: list[ToolCall]) -> str:
    if not tool_calls:
        return "(no tool calls observed)"
    lines: list[str] = []
    for i, tc in enumerate(tool_calls, start=1):
        args = _truncate(_as_text(tc.input) if tc.input is not None else "{}", _MAX_TOOL_ARGS_CHARS)
        line = f"{i}. {tc.name or 'unknown'} args={args}"
        if tc.output is not None:
            line += f" -> {_truncate(_as_text(tc.output), _MAX_TOOL_RESULT_CHARS)}"
        lines.append(line)
    return "\n".join(lines)


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
