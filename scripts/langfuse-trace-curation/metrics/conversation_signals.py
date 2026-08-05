"""LLM judge for weakness / stress signal tags (Round 3).

Review (AIPLAT-952): New metric for golden curation pipeline. Tags stress/coverage
dimensions only — write vs read classification was removed from scoring (still used
when building goldens via build_conversation_goldens.scenario_type_of).

Runs **after** the v3 quality judge (Round 1). Only conversations already labeled
``quality=good`` or ``quality=bad`` are eligible. Round 1 does **not** produce
these tags — it only assigns usefulness / quality / golden_readiness.

This metric labels which stress / coverage buckets a conversation belongs to
for golden curation (high cost, skill loading, HITL loops, etc.).

Structural facts from the canonical conversation (turn count, cost, tool
counts, truncated payloads) are injected into the prompt as evidence so the
judge can apply the documented thresholds consistently.
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

PROMPT_VERSION = "conversation-signals-v3"

QUALITY_FOR_SIGNALS = frozenset({"good", "bad"})

# Review (AIPLAT-952): Structural + transcript stress tags only. write_flow/read_only
# were removed — mutation vs inspection is not a scoring dimension.
SIGNAL_TAG_VALUES = (
    "high_turns",
    "high_cost",
    "high_tool_count",
    "large_tool_output",
    "tool_failure",
    "skill_loading",
    "hitl_loop",
    "multi_turn",
)

_SYSTEM_PROMPT = """You categorize production Harness AI agent conversations into
stress / coverage buckets used for golden selection. Use only the conversation
transcript, tool evidence, and the supplied structural facts. Apply the threshold
rules exactly when facts are given. Cite brief evidence for softer tags
(skill_loading, hitl_loop, tool_failure).

hitl_loop means the same HITL or approval question is asked again after the user
already answered — not merely that an approval gate occurred once."""

_PROMPT_TEMPLATE = """Label which weakness / coverage signal tags apply to this conversation.

This conversation was already judged in Round 1 as quality={quality}.
Do NOT re-judge good vs bad. Only assign signal tags.

Structural facts (from canonical metadata / tool payloads — treat as ground truth):
{structural_facts}

Tag definitions (apply when true):
- high_turns: num_turns >= 10
- high_cost: total_cost_usd >= 0.25
- high_tool_count: num_tool_calls >= 8
- large_tool_output: any truncated tool payload OR max tool output bytes >= 8000
- tool_failure: one or more tool results show errors / failures / HTTP 4xx-5xx /
  status ERROR / exceptions that matter to the trajectory
- skill_loading: the agent launched or relied on a skill (Skill tool, skill path,
  skill launch, or clear skill workflow)
- hitl_loop: the same HITL / AskUserQuestion / approval prompt is asked again after
  the user already responded (repeated question loop — not a single approval gate)
- multi_turn: num_turns >= 2

Also set:
- module_tag: "module:<name>" using the module from structural facts when present,
  else "module:none"
- signal_tags: list of all true tags from the enum above, plus module_tag
- confidence: 0.0-1.0
- reasoning: short justification
- evidence: short bullet strings citing transcript or facts

Complete conversation:
{conversation_text}
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "required": [
        "high_turns",
        "high_cost",
        "high_tool_count",
        "large_tool_output",
        "tool_failure",
        "skill_loading",
        "hitl_loop",
        "multi_turn",
        "module_tag",
        "signal_tags",
        "confidence",
        "reasoning",
        "evidence",
    ],
    "properties": {
        "high_turns": {"type": "boolean"},
        "high_cost": {"type": "boolean"},
        "high_tool_count": {"type": "boolean"},
        "large_tool_output": {"type": "boolean"},
        "tool_failure": {"type": "boolean"},
        "skill_loading": {"type": "boolean"},
        "hitl_loop": {"type": "boolean"},
        "multi_turn": {"type": "boolean"},
        "module_tag": {"type": "string"},
        "signal_tags": {
            "type": "array",
            "items": {"type": "string"},
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
}


def resolve_quality(row: dict[str, str]) -> str:
    """Agent quality label from a Round 1 review/results CSV row."""
    quality = (
        row.get("human_quality")
        or row.get("quality")
        or row.get("human_category")
        or row.get("final_category")
        or ""
    ).strip().lower()
    if quality == "needs_improvement":
        return "good"
    return quality


def quality_eligible_for_signals(quality: str) -> bool:
    return quality in QUALITY_FOR_SIGNALS


def _tool_calls(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    calls = conversation.get("tool_calls") or []
    return [call for call in calls if isinstance(call, dict)]


def _max_tool_output_bytes(conversation: dict[str, Any]) -> int:
    maximum = 0
    for call in _tool_calls(conversation):
        output = call.get("output")
        if output is None:
            continue
        if isinstance(output, str):
            maximum = max(maximum, len(output.encode("utf-8")))
        else:
            try:
                maximum = max(
                    maximum,
                    len(json.dumps(output, ensure_ascii=False).encode("utf-8")),
                )
            except TypeError:
                maximum = max(maximum, len(str(output).encode("utf-8")))
    return maximum


def extract_structural_facts(conversation: dict[str, Any]) -> dict[str, Any]:
    """Pull measurable facts from the canonical conversation for the judge prompt."""
    metadata = conversation.get("metadata") or {}
    truncated_ids = metadata.get("truncated_tool_use_ids") or []
    truncated_count = len(truncated_ids) if isinstance(truncated_ids, list) else 0
    tool_names = [
        str(call.get("name") or "")
        for call in _tool_calls(conversation)
        if call.get("name")
    ]
    return {
        "module": metadata.get("module") or conversation.get("module") or "none",
        "num_turns": int(metadata.get("num_turns") or 0),
        "total_cost_usd": float(metadata.get("total_cost_usd") or 0.0),
        "num_tool_calls": int(
            metadata.get("num_tool_calls") or len(_tool_calls(conversation))
        ),
        "truncated_tool_outputs": truncated_count,
        "max_tool_output_bytes": _max_tool_output_bytes(conversation),
        "tool_names_sample": tool_names[:20],
        "thresholds": {
            "high_turns": "num_turns >= 10",
            "high_cost": "total_cost_usd >= 0.25",
            "high_tool_count": "num_tool_calls >= 8",
            "large_tool_output": "truncated_tool_outputs > 0 OR max_tool_output_bytes >= 8000",
            "multi_turn": "num_turns >= 2",
        },
    }


def format_structural_facts(facts: dict[str, Any]) -> str:
    return json.dumps(facts, indent=2, ensure_ascii=False)


def normalize_signal_result(result: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    """Normalize LLM output into stable tag lists."""
    flags = {
        "high_turns": bool(result.get("high_turns")),
        "high_cost": bool(result.get("high_cost")),
        "high_tool_count": bool(result.get("high_tool_count")),
        "large_tool_output": bool(result.get("large_tool_output")),
        "tool_failure": bool(result.get("tool_failure")),
        "skill_loading": bool(result.get("skill_loading")),
        "hitl_loop": bool(result.get("hitl_loop")),
        "multi_turn": bool(result.get("multi_turn")),
    }

    module = str(facts.get("module") or "none")
    module_tag = str(result.get("module_tag") or f"module:{module}").strip()
    if not module_tag.startswith("module:"):
        module_tag = f"module:{module_tag}"

    tags: list[str] = []
    for name in SIGNAL_TAG_VALUES:
        if flags.get(name):
            tags.append(name)
    if module_tag not in tags:
        tags.append(module_tag)

    # Prefer LLM signal_tags order when valid; otherwise use normalized flags.
    llm_tags = [str(tag) for tag in (result.get("signal_tags") or []) if str(tag).strip()]
    allowed = set(SIGNAL_TAG_VALUES) | {module_tag}
    filtered = [tag for tag in llm_tags if tag in allowed]
    if filtered:
        if module_tag not in filtered:
            filtered.append(module_tag)
        tags = filtered

    confidence = max(0.0, min(1.0, float(result.get("confidence") or 0.0)))
    evidence = [str(item) for item in (result.get("evidence") or [])]
    return {
        **flags,
        "module_tag": module_tag,
        "signal_tags": tags,
        "confidence": confidence,
        "reasoning": str(result.get("reasoning") or ""),
        "evidence": evidence,
    }


@register_metric("harness_conversation_signals")
class HarnessConversationSignalsMetric(LLMConversationMetric):
    """Round 3: weakness / coverage signal tags for good/bad conversations."""

    def __init__(
        self,
        llm: BaseLLM,
        threshold: float = 0.5,
        *,
        max_conversation_chars: int = 120_000,
        **kwargs: object,
    ) -> None:
        super().__init__(
            llm=llm,
            threshold=threshold,
            name="harness_conversation_signals",
            dimension=Dimension.TRAJECTORY,
            **kwargs,
        )
        self.max_conversation_chars = max_conversation_chars

    async def a_measure(self, eval_case: EvalCase) -> Score:
        metadata = dict(eval_case.metadata or {})
        quality = str(metadata.get("round1_quality") or "").strip().lower()
        if quality and not quality_eligible_for_signals(quality):
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"Skipped: Round 1 quality={quality} (only good/bad are tagged)",
                metadata={
                    "prompt_version": PROMPT_VERSION,
                    "skipped": True,
                    "signals_skipped_reason": f"quality={quality}",
                    "signal_tags": [],
                },
            )

        conversation = metadata.get("canonical_conversation")
        if isinstance(conversation, dict):
            facts = extract_structural_facts(conversation)
        else:
            facts = {
                "module": metadata.get("module") or "none",
                "num_turns": metadata.get("num_turns") or 0,
                "total_cost_usd": metadata.get("total_cost_usd") or 0.0,
                "num_tool_calls": metadata.get("num_tool_calls") or 0,
                "truncated_tool_outputs": metadata.get("truncated_tool_outputs") or 0,
                "max_tool_output_bytes": metadata.get("max_tool_output_bytes") or 0,
                "tool_names_sample": metadata.get("tool_names_sample") or [],
                "thresholds": {
                    "high_turns": "num_turns >= 10",
                    "high_cost": "total_cost_usd >= 0.25",
                    "high_tool_count": "num_tool_calls >= 8",
                    "large_tool_output": (
                        "truncated_tool_outputs > 0 OR max_tool_output_bytes >= 8000"
                    ),
                    "multi_turn": "num_turns >= 2",
                },
            }

        conversation_text = format_conversation(eval_case)
        if len(conversation_text) > self.max_conversation_chars:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=(
                    f"Conversation has {len(conversation_text)} characters, exceeding the "
                    f"{self.max_conversation_chars}-character judge limit"
                ),
                metadata={
                    "prompt_version": PROMPT_VERSION,
                    "skipped": True,
                    "signals_skipped_reason": "conversation_too_large",
                    "signal_tags": [],
                    "requires_chunked_evaluation": True,
                },
            )

        result = await self.llm.generate_json(
            _PROMPT_TEMPLATE.format(
                quality=quality or "unknown",
                structural_facts=format_structural_facts(facts),
                conversation_text=conversation_text,
            ),
            _RESPONSE_SCHEMA,
            system_prompt=_SYSTEM_PROMPT,
        )
        normalized = normalize_signal_result(result, facts)
        tag_count = len(normalized["signal_tags"])
        # Score is informational: fraction of possible stress tags (excluding module).
        stress_tags = [tag for tag in normalized["signal_tags"] if tag in SIGNAL_TAG_VALUES]
        value = min(1.0, len(stress_tags) / len(SIGNAL_TAG_VALUES)) if tag_count else 0.0

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            reason=normalized["reasoning"],
            metadata={
                "prompt_version": PROMPT_VERSION,
                "skipped": False,
                "signal_tags": normalized["signal_tags"],
                "module_tag": normalized["module_tag"],
                "confidence": normalized["confidence"],
                "evidence": normalized["evidence"],
                "flags": {
                    key: normalized[key]
                    for key in SIGNAL_TAG_VALUES
                },
                "structural_facts": facts,
            },
        )
