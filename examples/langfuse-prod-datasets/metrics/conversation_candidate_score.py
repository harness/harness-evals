"""LLM-assisted eval candidate score (Round 4).

Scores each conversation 0–5 for golden selection. Ten equally-weighted
criteria contribute one point each when true; the final score is
``(hits / 10) * 5``.

Threshold-based criteria are computed from canonical conversation facts.
Transcript-based criteria are judged by the LLM.
"""

from __future__ import annotations

import json
from typing import Any

from harness_evals.core.eval_case import EvalCase
from harness_evals.core.metric import Dimension
from harness_evals.core.score import Score
from harness_evals.llm.base import BaseLLM
from harness_evals.metrics.conversation.llm_conversation_metric import LLMConversationMetric
from harness_evals.plugins import register_metric

from conversation_quality import format_conversation
from conversation_signals import extract_structural_facts, format_structural_facts

PROMPT_VERSION = "conversation-candidate-score-v5"

CRITERIA = (
    "high_turns",
    "high_cost",
    "high_tool_count",
    "large_tool_output",
    "tool_failure",
    "skill_loading",
    "hitl_loop",
    "multi_turn",
    "write_flow",
    "read_only",
)

_TRANSCRIPT_CRITERIA = frozenset(
    {"tool_failure", "skill_loading", "hitl_loop", "write_flow", "read_only"}
)

_FAILURE_MARKERS = (
    '"status": "error"',
    '"status":"error"',
    "status error",
    "status=error",
    "http 4",
    "http 5",
    "exception",
    "traceback",
    " failed",
    "failure",
    "tool error",
    "does not exist",
    "validationerror",
)

_SYSTEM_PROMPT = """You evaluate production Harness AI conversations for eval golden
selection. Score how many coverage / stress criteria apply. Use structural facts as
ground truth for threshold rules. Judge transcript-only criteria from evidence.

For criterion_notes, write short plain-English phrases (under 12 words each). Include
numbers where relevant (turns, cost, tool counts). Name tools and skills only — no file
paths, stack traces, or exception class names.

hitl_loop is true only when the same HITL or approval question is asked again after
the user already answered. A single approval gate or one yaml review does NOT count."""

_PROMPT_TEMPLATE = """For eval golden selection, decide which criteria apply to this conversation.
Each criterion has equal weight toward the final score.

Structural facts (ground truth for threshold rules — do not contradict these):
{structural_facts}

Precomputed structural criteria (already applied — include in your booleans as given):
{structural_criteria}

Criteria definitions:
- high_turns: num_turns >= 10
- high_cost: total_cost_usd >= 0.25
- high_tool_count: num_tool_calls >= 8
- large_tool_output: truncated tool payload OR max tool output bytes >= 8000
- tool_failure: tool results show errors / failures / HTTP 4xx-5xx / status ERROR
- skill_loading: agent launched or relied on a skill (Skill tool, skill path, skill workflow)
- hitl_loop: the same HITL / AskUserQuestion / approval prompt is asked again after the
  user already responded (repeated question loop — not a single approval gate)
- multi_turn: num_turns >= 2
- write_flow: create / update / delete / deploy style task or write tools
- read_only: list / explain / validate without mutation (mutually exclusive with write_flow)

Return a boolean for every criterion. For precomputed structural criteria, return the
given value. Exactly one of write_flow or read_only must be true.

For each criterion in criterion_notes, write one short plain-English phrase (under 12
words). Only fill notes for hitl_loop, write_flow, and read_only — structural criteria
and tool_failure / skill_loading are formatted separately.

Examples:
- hitl_loop: "same bucket question(duplicate) asked twice after user answered"
- write_flow: "created cost category"
- read_only: "listed pipelines only"

Do not include file paths, YAML dumps, or exception type names.

Complete conversation:
{conversation_text}
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "required": [*CRITERIA, "reasoning", "evidence", "criterion_notes"],
    "properties": {
        **{name: {"type": "boolean"} for name in CRITERIA},
        "reasoning": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "criterion_notes": {
            "type": "object",
            "properties": {name: {"type": "string"} for name in CRITERIA},
        },
    },
}


def compute_structural_criteria(facts: dict[str, Any]) -> dict[str, bool]:
    """Threshold criteria from canonical metadata / tool payloads."""
    num_turns = int(facts.get("num_turns") or 0)
    total_cost = float(facts.get("total_cost_usd") or 0.0)
    num_tool_calls = int(facts.get("num_tool_calls") or 0)
    truncated = int(facts.get("truncated_tool_outputs") or 0)
    max_bytes = int(facts.get("max_tool_output_bytes") or 0)
    return {
        "high_turns": num_turns >= 10,
        "high_cost": total_cost >= 0.25,
        "high_tool_count": num_tool_calls >= 8,
        "large_tool_output": truncated > 0 or max_bytes >= 8000,
        "multi_turn": num_turns >= 2,
    }


def merge_criteria(
    structural: dict[str, bool],
    llm_result: dict[str, Any],
    facts: dict[str, Any],
) -> dict[str, bool]:
    """Merge structural facts with LLM transcript criteria."""
    write_flow = bool(llm_result.get("write_flow"))
    read_only = bool(llm_result.get("read_only"))
    if write_flow == read_only:
        tool_blob = " ".join(str(name) for name in facts.get("tool_names_sample") or [])
        write_flow = any(
            token in tool_blob
            for token in ("harness_create", "harness_update", "harness_delete")
        )
        read_only = not write_flow

    merged = dict(structural)
    merged.update(
        {
            "tool_failure": bool(llm_result.get("tool_failure")),
            "skill_loading": bool(llm_result.get("skill_loading")),
            "hitl_loop": bool(llm_result.get("hitl_loop")),
            "write_flow": write_flow,
            "read_only": read_only,
        }
    )
    return {name: bool(merged.get(name)) for name in CRITERIA}


def compute_eval_candidate_score(criteria: dict[str, bool]) -> float:
    """Equal-weight score out of 5."""
    hits = sum(1 for name in CRITERIA if criteria.get(name))
    return round((hits / len(CRITERIA)) * 5.0, 2)


def _tool_calls(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    calls = conversation.get("tool_calls") or []
    return [call for call in calls if isinstance(call, dict)]


def _output_text(output: object) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, ensure_ascii=False)
    except TypeError:
        return str(output)


def _friendly_tool_name(name: str) -> str:
    """Strip MCP prefixes for readable reasoning."""
    cleaned = name.strip()
    for prefix in ("mcp__harness__", "mcp__"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    return cleaned or "unknown_tool"


def _format_count_label(count: int, singular: str, plural: str | None = None) -> str:
    if count == 1:
        return f"1 {singular}"
    return f"{count} {plural or singular + 's'}"


def _looks_like_tool_failure(output: object) -> bool:
    text = _output_text(output).lower()
    if not text.strip():
        return False
    if any(marker in text for marker in _FAILURE_MARKERS):
        return True
    if '"status": "error"' in text or '"status":"error"' in text:
        return True
    return False


def summarize_tool_failures(conversation: dict[str, Any]) -> list[str]:
    """Tool name + failure count only — no error payloads."""
    counts: dict[str, int] = {}
    for call in _tool_calls(conversation):
        if not _looks_like_tool_failure(call.get("output")):
            continue
        tool = _friendly_tool_name(str(call.get("name") or "unknown_tool"))
        counts[tool] = counts.get(tool, 0) + 1
    parts: list[str] = []
    for tool, count in sorted(counts.items()):
        if count == 1:
            parts.append(f"{tool} failed once")
        else:
            parts.append(f"{tool} failed {count} times")
    return parts


def summarize_skill_names(conversation: dict[str, Any]) -> list[str]:
    """Distinct skill names from Skill tool invocations only."""
    names: list[str] = []
    seen: set[str] = set()
    for call in _tool_calls(conversation):
        if str(call.get("name") or "") != "Skill":
            continue
        skill_input = call.get("input")
        if not isinstance(skill_input, dict):
            continue
        skill_name = str(
            skill_input.get("skill")
            or skill_input.get("skill_name")
            or skill_input.get("name")
            or ""
        ).strip()
        if skill_name and skill_name not in seen:
            seen.add(skill_name)
            names.append(skill_name)
    return names


def skill_load_had_errors(conversation: dict[str, Any]) -> bool:
    """True when skill-related tool calls returned failures."""
    for call in _tool_calls(conversation):
        name = str(call.get("name") or "")
        if name not in {"Skill", "Read", "Glob"}:
            continue
        if not _looks_like_tool_failure(call.get("output")):
            continue
        if name == "Skill":
            return True
        blob = _output_text(call.get("output")).lower()
        if "skill" in blob or "does not exist" in blob:
            return True
    return False


def _sanitize_note(text: str, *, max_words: int = 12) -> str:
    """Plain-English clip: drop paths and long dumps."""
    cleaned = " ".join(text.split())
    for prefix in ("yes —", "no —", "yes -", "no -", "yes—", "no—"):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    for token in cleaned.split():
        if token.startswith("/") or token.startswith("http"):
            cleaned = cleaned.replace(token, "").strip()
    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words])
    return cleaned.strip(" ,—-")


def _structural_criterion_line(name: str, matched: bool, facts: dict[str, Any]) -> str:
    num_turns = int(facts.get("num_turns") or 0)
    total_cost = float(facts.get("total_cost_usd") or 0.0)
    num_tool_calls = int(facts.get("num_tool_calls") or 0)
    truncated = int(facts.get("truncated_tool_outputs") or 0)
    prefix = "yes" if matched else "no"

    if name == "high_turns":
        return f"{prefix} — {_format_count_label(num_turns, 'turn')}"
    if name == "multi_turn":
        return f"{prefix} — {_format_count_label(num_turns, 'turn')}"
    if name == "high_cost":
        return f"{prefix} — ${total_cost:.2f}"
    if name == "high_tool_count":
        return f"{prefix} — {_format_count_label(num_tool_calls, 'tool call')}"
    if name == "large_tool_output":
        if truncated:
            return f"{prefix} — {_format_count_label(truncated, 'truncated tool output')}"
        return f"{prefix} — large tool output"
    return prefix


def _deterministic_transcript_line(
    name: str,
    matched: bool,
    conversation: dict[str, Any],
) -> str:
    prefix = "yes" if matched else "no"
    if name == "tool_failure":
        parts = summarize_tool_failures(conversation)
        if not parts:
            return f"{prefix} — no failed tools detected"
        return f"{prefix} — " + "; ".join(parts)
    if name == "skill_loading":
        names = summarize_skill_names(conversation)
        if not names:
            return f"{prefix} — no skills launched"
        label = ", ".join(names)
        if skill_load_had_errors(conversation):
            label = f"{label} (with load errors)"
        return f"{prefix} — {label}"
    return ""


def build_eval_candidate_reasoning(
    criteria: dict[str, bool],
    facts: dict[str, Any],
    conversation: dict[str, Any],
    llm_result: dict[str, Any],
    *,
    score_value: float,
) -> str:
    """Human-readable reasoning with concrete values per criterion."""
    hits = sum(1 for name in CRITERIA if criteria.get(name))
    lines = [f"Score {score_value}/5 ({hits}/{len(CRITERIA)} criteria matched):"]
    llm_notes = {
        str(key): str(value).strip()
        for key, value in (llm_result.get("criterion_notes") or {}).items()
        if str(value).strip()
    }
    structural_names = {
        "high_turns",
        "high_cost",
        "high_tool_count",
        "large_tool_output",
        "multi_turn",
    }

    for name in CRITERIA:
        matched = bool(criteria.get(name))
        if name in structural_names or name in {"tool_failure", "skill_loading"}:
            line = (
                _structural_criterion_line(name, matched, facts)
                if name in structural_names
                else _deterministic_transcript_line(name, matched, conversation)
            )
        elif name in {"hitl_loop", "write_flow", "read_only"}:
            note = _sanitize_note(llm_notes.get(name, ""))
            if note:
                line = f"{'yes' if matched else 'no'} — {note}"
            else:
                line = "yes" if matched else "no"
        else:
            line = "yes" if matched else "no"
        lines.append(f"- {name}: {line}")

    return "\n".join(lines)


def format_structural_criteria(structural: dict[str, bool]) -> str:
    return json.dumps(structural, indent=2, ensure_ascii=False)


def usefulness_eligible(usefulness: str) -> bool:
    return usefulness.strip().lower() != "useless"


def conversation_from_eval_case(eval_case: EvalCase) -> dict[str, Any]:
    """Build a canonical conversation dict from an offline review EvalCase."""
    metadata = dict(eval_case.metadata or {})
    nested = metadata.get("canonical_conversation")
    if isinstance(nested, dict):
        return nested
    payload = eval_case.to_dict()
    return {
        "input": payload.get("input"),
        "output": payload.get("output"),
        "messages": payload.get("messages") or [],
        "tool_calls": payload.get("tool_calls") or [],
        "metadata": metadata,
        "conversation_id": metadata.get("conversation_id"),
        "module": metadata.get("module"),
    }


@register_metric("harness_conversation_candidate_score")
class HarnessConversationCandidateScoreMetric(LLMConversationMetric):
    """Round 4: eval candidate score (0–5) from equally-weighted criteria."""

    def __init__(
        self,
        llm: BaseLLM,
        threshold: float = 2.5,
        *,
        max_conversation_chars: int = 120_000,
        **kwargs: object,
    ) -> None:
        super().__init__(
            llm=llm,
            threshold=threshold,
            name="harness_conversation_candidate_score",
            dimension=Dimension.TRAJECTORY,
            **kwargs,
        )
        self.max_conversation_chars = max_conversation_chars

    async def a_measure(self, eval_case: EvalCase) -> Score:
        usefulness = str((eval_case.metadata or {}).get("usefulness") or "useful")
        if not usefulness_eligible(usefulness):
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="Conversation marked useless — not scored for eval candidacy",
                metadata={
                    "prompt_version": PROMPT_VERSION,
                    "skipped": True,
                    "skip_reason": "useless",
                    "eval_candidate_score": 0.0,
                    "criteria_hits": 0,
                    "criteria": {name: False for name in CRITERIA},
                },
            )

        conversation = conversation_from_eval_case(eval_case)
        facts = extract_structural_facts(conversation)
        structural = compute_structural_criteria(facts)

        if not eval_case.messages or len(eval_case.messages) < 2:
            criteria = merge_criteria(
                structural,
                {
                    "tool_failure": False,
                    "skill_loading": False,
                    "hitl_loop": False,
                    "write_flow": False,
                    "read_only": True,
                },
                facts,
            )
            score_value = compute_eval_candidate_score(criteria)
            reasoning = build_eval_candidate_reasoning(
                criteria,
                facts,
                conversation,
                {},
                score_value=score_value,
            )
            return Score(
                name=self.name,
                value=score_value / 5.0,
                threshold=self.threshold / 5.0,
                reason=reasoning,
                metadata={
                    "prompt_version": PROMPT_VERSION,
                    "skipped": False,
                    "eval_candidate_score": score_value,
                    "criteria_hits": sum(1 for name in CRITERIA if criteria[name]),
                    "criteria": criteria,
                    "module_tag": f"module:{facts.get('module') or 'none'}",
                },
            )

        conversation_text = format_conversation(eval_case)
        if len(conversation_text) > self.max_conversation_chars:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=(
                    f"Conversation exceeds {self.max_conversation_chars}-character judge limit"
                ),
                metadata={
                    "prompt_version": PROMPT_VERSION,
                    "skipped": True,
                    "skip_reason": "conversation_too_large",
                    "eval_candidate_score": 0.0,
                    "criteria_hits": 0,
                    "criteria": {name: False for name in CRITERIA},
                },
            )

        result = await self.llm.generate_json(
            _PROMPT_TEMPLATE.format(
                structural_facts=format_structural_facts(facts),
                structural_criteria=format_structural_criteria(structural),
                conversation_text=conversation_text,
            ),
            _RESPONSE_SCHEMA,
            system_prompt=_SYSTEM_PROMPT,
        )
        criteria = merge_criteria(structural, result, facts)
        score_value = compute_eval_candidate_score(criteria)
        reasoning = build_eval_candidate_reasoning(
            criteria,
            facts,
            conversation,
            result,
            score_value=score_value,
        )
        evidence = [str(item) for item in (result.get("evidence") or [])]
        module = str(facts.get("module") or "none")

        return Score(
            name=self.name,
            value=score_value / 5.0,
            threshold=self.threshold / 5.0,
            reason=reasoning,
            metadata={
                "prompt_version": PROMPT_VERSION,
                "skipped": False,
                "eval_candidate_score": score_value,
                "criteria_hits": sum(1 for name in CRITERIA if criteria[name]),
                "criteria": criteria,
                "module_tag": f"module:{module}",
                "evidence": evidence,
                "structural_facts": facts,
            },
        )
