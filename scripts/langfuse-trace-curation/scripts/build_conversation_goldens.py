#!/usr/bin/env python3
"""Convert categorized production conversations into ConversationGolden JSONL.

Reads the offline judge review (``review.csv``) plus the canonical
``*.conversation.json`` files, filters out non-runnable rows, makes each
remaining conversation environment-portable, classifies it, and emits
``ConversationGolden`` rows for the live Harness SSE conversation runner.

Design decisions (see the plan doc):

* Include agent-quality ``good`` / ``bad`` / ``unclear``. Exclude ``useless``.
  Portability is a separate ``golden_readiness`` column (``ready`` /
  ``needs_rewrite``); ``needs_rewrite`` can apply to good or bad.
  Prefer promoting ``ready`` as-is; ``needs_rewrite`` usually needs curated overrides.
* Exclude pipeline error-analysis conversations (they need a specific failed
  execution that does not exist in the eval environment).
* Environment portability:
    - production account/org/project scope is replaced with ``${HARNESS_ORG}`` /
      ``${HARNESS_PROJECT}`` placeholders.
    - read-only or write/mutation rows referencing production-specific named
      resources REQUIRE a curated override; missing overrides fail closed
      (excluded, reason logged).
    - ``elicitation_hints`` never contain hard-coded identifiers.
* Review-gate approval injections ("The user approved the entity ...") are NOT
  real user turns — they are elicitation continuations, so they never become
  scripted user turns and never inflate ``max_turns``.
* A secret/PII scan runs over every emitted field; any residual production URL,
  token, API key, or email fails generation.

Outputs:
* ``.harness/evals/datasets/aiplat-ua/conversation/conversation.jsonl`` — validated ConversationGolden rows
* ``.harness/evals/datasets/aiplat-ua/conversation/conversation.goldens.manifest.jsonl`` — one record per source
  conversation with the decision (emitted / excluded), action, and reason.

Review (AIPLAT-952): Use ``--only-in-review`` with ``goldens-final.csv`` to emit
``.harness/evals/datasets/aiplat-ua/conversation/conversation-readonly.jsonl`` and ``conversation-write.jsonl``
via manual split after build, or point ``--output`` at the combined file.

Usage:
  python scripts/build_conversation_goldens.py \
      --review results/module-coverage-030/review.csv \
      --conversations module-coverage
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATASET_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
CONVERSATION_DATASETS = REPO_ROOT / ".harness" / "evals" / "datasets" / "aiplat-ua" / "conversation"
LOCAL_RUN_TEMPLATE = REPO_ROOT / "local-run" / "aiplat-ua.template"
sys.path.insert(0, str(LOCAL_RUN_TEMPLATE))

from harness_evals.conversation.golden import ConversationGolden  # noqa: E402

DEFAULT_OVERRIDES = DATASET_ROOT / "conversation-golden-overrides.json"
DEFAULT_OUTPUT = CONVERSATION_DATASETS / "conversation.jsonl"

# Agent-quality categories we keep. Portability is tracked separately.
INCLUDED_QUALITIES = {"good", "bad", "unclear"}
# Legacy single-label values from conversation-quality-v2 review CSVs.
_LEGACY_INCLUDED = {"needs_improvement"}
EXCLUDED_LABELS = {"useless", "not_applicable"}
READINESS_VALUES = {"ready", "needs_rewrite"}


def _resolve_review_labels(review_row: dict[str, str]) -> tuple[str, str]:
    """Return (quality, golden_readiness) from a review/results row.

    Supports v3 separate columns and v2 ``final_category=needs_improvement``.
    Human overrides win when present. Readiness is only ``ready`` or ``needs_rewrite``.
    """
    quality = (
        review_row.get("human_quality")
        or review_row.get("quality")
        or review_row.get("human_category")
        or review_row.get("final_category")
        or ""
    ).strip().lower()
    readiness = (
        review_row.get("human_golden_readiness")
        or review_row.get("golden_readiness")
        or ""
    ).strip().lower()

    if quality == "needs_improvement":
        # v2 conflated agent quality + portability into one label.
        quality = "good"
        readiness = readiness or "needs_rewrite"
    if readiness == "unsuitable" or readiness == "not_applicable":
        # Retired readiness labels — treat as rewrite candidates.
        readiness = "needs_rewrite"
    if readiness not in READINESS_VALUES:
        readiness = "needs_rewrite"
    return quality, readiness


# Judge categories we keep. "useless" is dropped; anything else unknown is dropped.
INCLUDED_CATEGORIES = INCLUDED_QUALITIES | _LEGACY_INCLUDED

# Review-gate / elicitation injections: synthetic user messages the platform
# feeds back into the loop (approvals and form-value continuations). These are
# NOT real user follow-ups and must never become scripted turns.
_APPROVAL_INJECTION_RE = re.compile(
    r"user approved the entity"
    r"|call harness_create.*again"
    r"|review gate will inject"
    r"|the user provided the following values"
    r"|continue with the task using these values"
    r"|do not ask the same questions again",
    re.IGNORECASE,
)

# Error / failure analysis signals (drop: needs a specific failed execution).
_ERROR_ANALYSIS_INPUT_RE = re.compile(
    r"analyze (the )?error|diagnose|why did .* fail|why is .* failing|"
    r"pipeline .*(fail|error)|debug (the )?(pipeline|execution)|root cause",
    re.IGNORECASE,
)
_ERROR_ANALYSIS_TOOLS = {
    "mcp__harness__harness_diagnose",
    "harness_diagnose",
    "fetch_and_extract_log_zip",
}
_ERROR_ANALYSIS_REASONING_RE = re.compile(
    r"analyze (the )?(pipeline )?error|diagnos|failed pipeline execution|"
    r"root cause of the (pipeline )?fail",
    re.IGNORECASE,
)

# Write / mutation signals.
_WRITE_TOOL_RE = re.compile(
    r"harness_(create|update|delete)\b", re.IGNORECASE
)
_REVIEW_GATE_OUTPUT_RE = re.compile(
    r"waiting for user to review", re.IGNORECASE
)

# Named-resource read signal (read that targets a specific existing resource id).
_NAMED_READ_HINT_RE = re.compile(
    r"\brefer\b|\bexisting\b|resource_id", re.IGNORECASE
)

# PII / secret scan patterns applied to emitted golden fields.
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("harness_token", re.compile(r"\b(pat|sat|st)\.[A-Za-z0-9._-]{6,}", re.IGNORECASE)),
    ("bearer_token", re.compile(r"bearer\s+[A-Za-z0-9._-]{12,}", re.IGNORECASE)),
    ("prod_url", re.compile(r"https?://[^\s\"']*", re.IGNORECASE)),
    ("api_key_kv", re.compile(r"api[_-]?key\s*[:=]\s*\S+", re.IGNORECASE)),
]
# URLs that are allowed to survive (documentation / schema references only).
_ALLOWED_URL_RE = re.compile(
    r"^https?://(developer\.harness\.io|github\.com|docs\.|backstage\.io)", re.IGNORECASE
)

MAX_USER_TURNS = 6


@dataclass
class ManifestRecord:
    conversation_id: str
    source_file: str
    module: str | None
    environment: str | None
    judge_category: str | None
    decision: str  # "emitted" | "excluded"
    action: str  # unchanged | scope_normalized | override | excluded
    reason: str
    scenario_type: str | None = None
    golden_id: str | None = None
    golden_readiness: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v not in (None, [], "")}


def load_overrides(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    overrides = data.get("overrides", data)
    if not isinstance(overrides, dict):
        raise ValueError(f"Overrides file {path} must map conversation_id -> override object")
    return overrides


def load_review(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            cid = (row.get("conversation_id") or "").strip()
            if cid:
                rows[cid] = row
    return rows


def _user_messages(conversation: dict[str, Any]) -> list[str]:
    """Real user turns, excluding synthetic review-gate approval injections."""
    result: list[str] = []
    for message in conversation.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if _APPROVAL_INJECTION_RE.search(content):
            continue
        result.append(content.strip())
    return result


def _tool_names(conversation: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for call in conversation.get("tool_calls") or []:
        if isinstance(call, dict) and call.get("name"):
            names.append(str(call["name"]))
    return names


def _normalize_tool_name_for_golden(name: str) -> str:
    """Normalize production MCP tool names to short golden form (harness_list, Skill, …)."""
    return name.rsplit("__", 1)[-1] if "__" in name else name


def _trackable_harness_tool_name(name_fragment: str) -> str | None:
    """Return short Harness MCP tool name for trajectory expectations, or None to skip."""
    if not name_fragment or name_fragment in _NON_HARNESS_TOOLS or name_fragment == "Skill":
        return None
    if name_fragment.startswith("harness_") or name_fragment.startswith("validate_"):
        return name_fragment
    if "hql" in name_fragment.lower():
        return name_fragment
    return _normalize_harness_tool_name(name_fragment)


def _expected_tool_calls_from_conversation(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    """Ordered Harness MCP tool calls from the source conversation trace."""
    calls: list[dict[str, Any]] = []
    for call in conversation.get("tool_calls") or []:
        if not isinstance(call, dict) or not call.get("name"):
            continue
        tool_name = _trackable_harness_tool_name(_normalize_tool_name_for_golden(str(call["name"])))
        if not tool_name:
            continue
        args = call.get("input") if isinstance(call.get("input"), dict) else call.get("arguments")
        input_args = _resource_type_input(args)
        payload: dict[str, Any] = {"name": tool_name}
        if input_args:
            payload["input"] = input_args
        calls.append(payload)
    return calls


def _resource_type_input(args: object) -> dict[str, Any] | None:
    if not isinstance(args, dict):
        return None
    resource_type = args.get("resource_type")
    if resource_type is None:
        return None
    return {"resource_type": resource_type}


def _expected_tool_calls_from_sse_checks(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Derive ordered Harness MCP tool calls from assistant_tool_request checks."""
    calls: list[dict[str, Any]] = []
    for check in checks:
        if not isinstance(check, dict) or check.get("event") != "assistant_tool_request":
            continue
        name_fragment: str | None = None
        resource_type: object = None
        for match in check.get("match") or []:
            if not isinstance(match, dict):
                continue
            path = str(match.get("path") or "")
            if "name" in path:
                if "contains" in match:
                    name_fragment = str(match["contains"])
                elif "equals" in match:
                    name_fragment = str(match["equals"])
            if "resource_type" in path and "equals" in match:
                resource_type = match["equals"]
        if not name_fragment:
            continue
        tool_name = _trackable_harness_tool_name(name_fragment)
        if not tool_name:
            continue
        payload: dict[str, Any] = {"name": tool_name}
        input_args = _resource_type_input({"resource_type": resource_type}) if resource_type is not None else None
        if input_args:
            payload["input"] = input_args
        calls.append(payload)
    return calls


def _expected_tool_calls_for(
    conversation: dict[str, Any],
    checks: list[dict[str, Any]],
    override: dict[str, Any] | None = None,
) -> list[dict[str, Any]] | None:
    if override and override.get("expected_tool_calls"):
        return list(override["expected_tool_calls"])
    from_checks = _expected_tool_calls_from_sse_checks(checks)
    if from_checks:
        return from_checks
    from_conversation = _expected_tool_calls_from_conversation(conversation)
    return from_conversation or None


# Non-Harness agent tools we ignore when inferring trajectory expectations.
_NON_HARNESS_TOOLS = frozenset(
    {
        "AskUserQuestion",
        "Read",
        "Write",
        "Grep",
        "Glob",
        "Bash",
        "Task",
        "WebFetch",
        "WebSearch",
    }
)


def _normalize_harness_tool_name(name: str) -> str | None:
    """Return the short Harness MCP tool name (e.g. harness_list) if applicable."""
    if not name or name in _NON_HARNESS_TOOLS:
        return None
    short = name.rsplit("__", 1)[-1] if "__" in name else name
    if short.startswith("harness_") or short.startswith("validate_"):
        return short
    return None


def _primary_harness_tool(conversation: dict[str, Any]) -> str | None:
    """Most frequent Harness MCP tool in the source trace (the obvious expected call)."""
    from collections import Counter

    counts: Counter[str] = Counter()
    for name in _tool_names(conversation):
        short = _normalize_harness_tool_name(name)
        if short:
            counts[short] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _tool_trajectory_checks(tool_short_name: str) -> list[dict[str, Any]]:
    name_match = {"path": "$.name", "contains": tool_short_name}
    return [
        {
            "event": "assistant_tool_request",
            "path": "$.v[*]",
            "match": [name_match],
        },
        {
            "event": "assistant_tool_result",
            "path": "$.v[*]",
            "match": [name_match],
        },
        {"event": "assistant_message", "exists": True},
    ]


def _write_tools_from_conversation(
    conversation: dict[str, Any],
    override: dict[str, Any] | None = None,
) -> list[str]:
    """Infer expected Harness write tools from source trace or curated override text."""
    seen: set[str] = set()
    tools: list[str] = []
    for name in _tool_names(conversation):
        short = _normalize_harness_tool_name(name)
        if short and _WRITE_TOOL_RE.search(short) and short not in seen:
            seen.add(short)
            tools.append(short)
    if tools:
        return tools

    if override:
        text_parts = [override.get("expected_outcome", ""), override.get("scenario", "")]
        text_parts.extend(override.get("turns") or [])
        if override.get("initial_prompt"):
            text_parts.append(override["initial_prompt"])
        blob = " ".join(str(part) for part in text_parts)
        inferred: list[str] = []
        for match in re.finditer(r"\bharness_(create|update|delete)\b", blob, re.IGNORECASE):
            tool = f"harness_{match.group(1).lower()}"
            if tool not in inferred:
                inferred.append(tool)
        if inferred:
            return inferred
        if re.search(r"\b(create|creates|creating)\b", blob, re.IGNORECASE) and re.search(
            r"\b(update|updates|updating)\b", blob, re.IGNORECASE
        ):
            return ["harness_create", "harness_update"]

    return ["harness_create"]


def _write_sse_checks(
    conversation: dict[str, Any],
    override: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """SSE assertions for write flows: require write tool request+result, not entity_mutation."""
    checks: list[dict[str, Any]] = []
    for tool in _write_tools_from_conversation(conversation, override):
        name_match = {"path": "$.name", "contains": tool}
        checks.append(
            {
                "event": "assistant_tool_request",
                "path": "$.v[*]",
                "match": [name_match],
            }
        )
        checks.append(
            {
                "event": "assistant_tool_result",
                "path": "$.v[*]",
                "match": [name_match],
            }
        )
    checks.append({"event": "assistant_message", "exists": True})
    return checks


def is_error_analysis(conversation: dict[str, Any], reasoning: str) -> bool:
    first_user = (_user_messages(conversation) or [""])[0]
    if _ERROR_ANALYSIS_INPUT_RE.search(first_user):
        return True
    if any(name in _ERROR_ANALYSIS_TOOLS for name in _tool_names(conversation)):
        return True
    return bool(reasoning and _ERROR_ANALYSIS_REASONING_RE.search(reasoning))


def is_write_flow(conversation: dict[str, Any]) -> bool:
    for name in _tool_names(conversation):
        if _WRITE_TOOL_RE.search(name):
            return True
    for call in conversation.get("tool_calls") or []:
        output = call.get("output") if isinstance(call, dict) else None
        if isinstance(output, str) and _REVIEW_GATE_OUTPUT_RE.search(output):
            return True
    return False


def references_named_resource(conversation: dict[str, Any]) -> bool:
    """Heuristic: the task depends on a specific pre-existing named resource."""
    first_user = (_user_messages(conversation) or [""])[0]
    if _NAMED_READ_HINT_RE.search(first_user):
        return True
    for call in conversation.get("tool_calls") or []:
        args = call.get("input") if isinstance(call, dict) else None
        if isinstance(args, dict) and args.get("resource_id"):
            return True
    return False


def scenario_type_of(conversation: dict[str, Any]) -> str:
    if is_write_flow(conversation):
        return "write"
    return "read_only"


def _placeholder_scope(text: str, org: str | None, project: str | None) -> str:
    """Replace production org/project identifiers with portable placeholders."""
    out = text
    if project:
        out = re.sub(rf"\b{re.escape(project)}\b", "${HARNESS_PROJECT}", out)
    if org:
        out = re.sub(rf"\b{re.escape(org)}\b", "${HARNESS_ORG}", out)
    return out


def _sse_checks_for(
    scenario_type: str,
    conversation: dict[str, Any],
    override: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if scenario_type == "write":
        return _write_sse_checks(conversation, override)
    primary_tool = _primary_harness_tool(conversation)
    if primary_tool:
        return _tool_trajectory_checks(primary_tool)
    # Fallback when the source trace has no Harness MCP calls to infer from.
    return [
        {"event": "assistant_tool_request", "exists": True},
        {"event": "assistant_tool_result", "exists": True},
        {"event": "assistant_message", "exists": True},
    ]


def _scan_for_secrets(value: Any, path: str, findings: list[str]) -> None:
    if isinstance(value, str):
        for label, pattern in _SECRET_PATTERNS:
            for match in pattern.findall(value):
                text = match if isinstance(match, str) else "".join(match)
                if label == "prod_url" and _ALLOWED_URL_RE.match(text):
                    continue
                findings.append(f"{path}: {label} -> {text[:60]}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _scan_for_secrets(item, f"{path}.{key}", findings)
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            _scan_for_secrets(item, f"{path}[{idx}]", findings)


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:40] or "task"


def build_golden(
    conversation: dict[str, Any],
    review_row: dict[str, str],
    override: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, ManifestRecord]:
    metadata = conversation.get("metadata") or {}
    cid = conversation.get("conversation_id") or ""
    module = metadata.get("module")
    environment = metadata.get("environment")
    org = metadata.get("org_id")
    project = metadata.get("project_id")
    judge_category, golden_readiness = _resolve_review_labels(review_row)
    reasoning = review_row.get("reasoning") or ""
    source_file = metadata.get("transcript_file") or cid

    def manifest(decision: str, action: str, reason: str, **extra: Any) -> ManifestRecord:
        return ManifestRecord(
            conversation_id=cid,
            source_file=source_file,
            module=module,
            environment=environment,
            judge_category=judge_category or None,
            golden_readiness=golden_readiness or None,
            decision=decision,
            action=action,
            reason=reason,
            **extra,
        )

    # 1. Category filter (agent quality). Portability is separate.
    category = judge_category
    if category in EXCLUDED_LABELS or (category and category not in INCLUDED_CATEGORIES):
        return None, manifest("excluded", "excluded", f"judge_category={category or 'missing'}")

    # 2. Explicit curated override (may force-include or force-exclude).
    if override is not None:
        if override.get("exclude"):
            return None, manifest("excluded", "override", override.get("reason", "curated exclusion"))
        return _golden_from_override(conversation, review_row, override, manifest)

    # 3. Oversized conversations require a curated override.
    user_msgs = _user_messages(conversation)
    if len(user_msgs) > MAX_USER_TURNS:
        return None, manifest(
            "excluded", "excluded", f"oversized ({len(user_msgs)} user turns); add a curated override"
        )
    if not user_msgs:
        return None, manifest("excluded", "excluded", "no real user message")

    # 4. Error / failure analysis is out of scope.
    if is_error_analysis(conversation, reasoning):
        return None, manifest("excluded", "excluded", "pipeline error-analysis scenario")

    scenario_type = scenario_type_of(conversation)

    # 5. Rows that reference a production-specific resource need a curated override.
    if references_named_resource(conversation):
        flow = "write flow" if scenario_type == "write" else "read-only flow"
        return None, manifest(
            "excluded",
            "excluded",
            f"{flow} references a production-specific resource; add a curated override",
            scenario_type=scenario_type,
        )

    action = "scope_normalized"
    turns = [_placeholder_scope(msg, org, project) for msg in user_msgs]
    scenario = _placeholder_scope(user_msgs[0], org, project)
    expected_outcome = _derive_expected_outcome(conversation, org, project)

    golden = _assemble_golden(
        cid=cid,
        module=module,
        environment=environment,
        judge_category=judge_category,
        golden_readiness=golden_readiness,
        scenario=scenario,
        expected_outcome=expected_outcome,
        turns=turns,
        scenario_type=scenario_type,
        conversation=conversation,
    )
    return _finalize(golden, manifest, action, scenario_type)


def _derive_expected_outcome(conversation: dict[str, Any], org: str | None, project: str | None) -> str:
    """Environment-neutral outcome: describe the task, not the prod answer."""
    first_user = (_user_messages(conversation) or [""])[0]
    portable = _placeholder_scope(first_user, org, project)
    # Keep it behavioural and short; never copy prod output verbatim. Collapse
    # whitespace so multi-line prompts (e.g. pasted YAML) still yield a sentence.
    condensed = re.sub(r"\s+", " ", portable).strip()[:200]
    return f"The assistant completes the request: {condensed}"


def _assemble_golden(
    *,
    cid: str,
    module: str | None,
    environment: str | None,
    judge_category: str,
    golden_readiness: str,
    scenario: str,
    expected_outcome: str,
    turns: list[str],
    scenario_type: str,
    conversation: dict[str, Any],
) -> dict[str, Any]:
    short = cid.split("-")[0] if cid else "conv"
    golden_id = f"{module or 'harness'}-{short}-{_slug(scenario)}"
    context = ["Org: ${HARNESS_ORG}, Project: ${HARNESS_PROJECT}"]

    golden: dict[str, Any] = {
        "id": golden_id,
        "scenario": scenario,
        "expected_outcome": expected_outcome,
        "mode": "scripted",
        "turns": [{"role": "user", "content": t} for t in turns],
        "max_turns": max(1, len(turns)),
        "max_elicitation_rounds": 8 if scenario_type == "write" else 4,
        "user_persona": "Harness user who provides reasonable answers when asked",
        "context": context,
        "metadata": {
            "sse_checks": _sse_checks_for(scenario_type, conversation),
            "source_conversation_id": cid,
        },
        "tags": {
            "module": str(module or "unknown"),
            "environment": str(environment or "unknown"),
            "judge_category": judge_category or "unknown",
            "golden_readiness": golden_readiness or "unknown",
            "scenario_type": scenario_type,
        },
    }
    golden["elicitation_hints"] = {"llm_on_miss": True}
    if scenario_type == "write":
        golden["elicitation_hints"]["yaml"] = {"default_action": "accept"}
    expected_tool_calls = _expected_tool_calls_for(conversation, golden["metadata"]["sse_checks"])
    if expected_tool_calls:
        golden["expected_tool_calls"] = expected_tool_calls
    return golden


def _golden_from_override(
    conversation: dict[str, Any],
    review_row: dict[str, str],
    override: dict[str, Any],
    manifest,
) -> tuple[dict[str, Any] | None, ManifestRecord]:
    metadata = conversation.get("metadata") or {}
    cid = conversation.get("conversation_id") or ""
    module = metadata.get("module")
    environment = metadata.get("environment")
    judge_category, golden_readiness = _resolve_review_labels(review_row)
    scenario_type = override.get("scenario_type") or scenario_type_of(conversation)
    turns = override.get("turns") or [override.get("initial_prompt", "")]
    golden = _assemble_golden(
        cid=cid,
        module=module,
        environment=environment,
        judge_category=judge_category,
        golden_readiness=golden_readiness,
        scenario=override["scenario"],
        expected_outcome=override["expected_outcome"],
        turns=turns,
        scenario_type=scenario_type,
        conversation=conversation,
    )
    if "elicitation_hints" in override:
        golden["elicitation_hints"] = override["elicitation_hints"]
    golden["elicitation_hints"].setdefault("llm_on_miss", True)
    if "sse_checks" in override:
        golden["metadata"]["sse_checks"] = override["sse_checks"]
    elif scenario_type == "write":
        golden["metadata"]["sse_checks"] = _write_sse_checks(conversation, override)
    if "max_elicitation_rounds" in override:
        golden["max_elicitation_rounds"] = override["max_elicitation_rounds"]
    if "context" in override:
        golden["context"] = override["context"]
    if "user_persona" in override:
        golden["user_persona"] = override["user_persona"]
    if "improve_later" in override:
        # Review (AIPLAT-952): Human notes for goldens that need env seeding (e.g. OPA policy).
        golden["metadata"]["improve_later"] = override["improve_later"]
    expected_tool_calls = _expected_tool_calls_for(
        conversation,
        golden["metadata"]["sse_checks"],
        override,
    )
    if expected_tool_calls:
        golden["expected_tool_calls"] = expected_tool_calls
    return _finalize(golden, manifest, "override", scenario_type)


def _finalize(
    golden: dict[str, Any],
    manifest,
    action: str,
    scenario_type: str,
) -> tuple[dict[str, Any] | None, ManifestRecord]:
    # Secret / PII scan over emitted fields.
    findings: list[str] = []
    for key in (
        "scenario",
        "expected_outcome",
        "initial_prompt",
        "turns",
        "context",
        "elicitation_hints",
        "expected_tool_calls",
    ):
        if key in golden:
            _scan_for_secrets(golden[key], key, findings)
    if findings:
        return None, manifest(
            "excluded",
            "excluded",
            "secret/PII scan failed: " + "; ".join(findings[:5]),
            scenario_type=scenario_type,
        )

    # Validate against the dataclass.
    try:
        ConversationGolden.from_dict(dict(golden))
    except Exception as exc:  # noqa: BLE001 - surface the validation failure
        return None, manifest(
            "excluded", "excluded", f"ConversationGolden validation failed: {exc}", scenario_type=scenario_type
        )

    return golden, manifest(
        "emitted", action, "ok", scenario_type=scenario_type, golden_id=golden["id"]
    )


def find_conversation_files(
    conversations_dirs: list[Path],
    *,
    only_in_review: set[str] | None = None,
    review_order: list[str] | None = None,
) -> list[Path]:
    """Resolve canonical conversation files across one or more dataset directories.

    Review (AIPLAT-952): Supports --conversations module-coverage --conversations random
    and --only-in-review to build exactly the goldens-final.csv subset into JSONL.
    """
    if only_in_review is not None:
        by_id: dict[str, Path] = {}
        for conversations_dir in conversations_dirs:
            if not conversations_dir.is_dir():
                continue
            for canonical_path in conversations_dir.glob("*.conversation.json"):
                conversation = json.loads(canonical_path.read_text())
                cid = conversation.get("conversation_id") or ""
                if cid in only_in_review and cid not in by_id:
                    by_id[cid] = canonical_path
        order = review_order if review_order is not None else sorted(by_id)
        return [by_id[cid] for cid in order if cid in by_id]

    files: list[Path] = []
    seen: set[str] = set()
    for conversations_dir in conversations_dirs:
        if not conversations_dir.is_dir():
            continue
        for canonical_path in sorted(conversations_dir.glob("*.conversation.json")):
            conversation = json.loads(canonical_path.read_text())
            cid = conversation.get("conversation_id") or ""
            if cid and cid in seen:
                continue
            if cid:
                seen.add(cid)
            files.append(canonical_path)
    return files


def convert(
    review_path: Path,
    conversations_dirs: list[Path],
    overrides_path: Path,
    output_path: Path,
    *,
    only_in_review: bool = False,
) -> tuple[int, int, Path]:
    review = load_review(review_path)
    overrides = load_overrides(overrides_path)
    review_ids = set(review) if only_in_review else None
    review_order = list(review) if only_in_review else None
    canonical_files = find_conversation_files(
        conversations_dirs,
        only_in_review=review_ids,
        review_order=review_order,
    )

    if only_in_review:
        missing = set(review) - {
            json.loads(path.read_text()).get("conversation_id") or "" for path in canonical_files
        }
        if missing:
            print(f"Warning: {len(missing)} review row(s) missing *.conversation.json", file=sys.stderr)
            for cid in sorted(missing)[:5]:
                print(f"  missing: {cid}", file=sys.stderr)

    goldens: list[dict[str, Any]] = []
    manifest_records: list[ManifestRecord] = []

    for canonical_path in canonical_files:
        conversation = json.loads(canonical_path.read_text())
        cid = conversation.get("conversation_id") or ""
        review_row = review.get(cid, {})
        if not review_row:
            manifest_records.append(
                ManifestRecord(
                    conversation_id=cid,
                    source_file=canonical_path.name,
                    module=(conversation.get("metadata") or {}).get("module"),
                    environment=(conversation.get("metadata") or {}).get("environment"),
                    judge_category=None,
                    decision="excluded",
                    action="excluded",
                    reason="no review row",
                )
            )
            continue
        golden, record = build_golden(conversation, review_row, overrides.get(cid))
        manifest_records.append(record)
        if golden is not None:
            goldens.append(golden)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for golden in goldens:
            handle.write(json.dumps(golden, ensure_ascii=False) + "\n")

    manifest_path = output_path.with_suffix(".manifest.jsonl")
    with manifest_path.open("w") as handle:
        for record in manifest_records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    return len(goldens), len(manifest_records), manifest_path


def _print_summary(goldens: int, total: int, manifest_path: Path) -> None:
    reasons: dict[str, int] = {}
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("decision") == "excluded":
            reasons[record.get("reason", "unknown")] = reasons.get(record.get("reason", "unknown"), 0) + 1
    print(f"Emitted {goldens}/{total} goldens")
    if reasons:
        print("Excluded by reason:")
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {count:>2}  {reason}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True, help="Path to review.csv")
    parser.add_argument(
        "--conversations",
        type=Path,
        action="append",
        required=True,
        help="Directory of *.conversation.json (repeatable; relative to dataset root or absolute)",
    )
    parser.add_argument(
        "--only-in-review",
        action="store_true",
        help="Process only conversation IDs present in the review CSV (recommended for goldens-final.csv)",
    )
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    review_path = args.review if args.review.is_absolute() else DATASET_ROOT / args.review
    conversations_dirs = [
        d if d.is_absolute() else DATASET_ROOT / d for d in args.conversations
    ]
    overrides_path = args.overrides if args.overrides.is_absolute() else DATASET_ROOT / args.overrides

    if not review_path.is_file():
        parser.error(f"review file not found: {review_path}")
    if not any(d.is_dir() for d in conversations_dirs):
        parser.error(f"no conversations dir found among: {conversations_dirs}")

    if args.output.is_absolute():
        output_path = args.output
    else:
        output_path = (Path.cwd() / args.output).resolve()
    goldens, total, manifest_path = convert(
        review_path,
        conversations_dirs,
        overrides_path,
        output_path,
        only_in_review=args.only_in_review,
    )
    _print_summary(goldens, total, manifest_path)
    print(f"Wrote goldens to {args.output}")
    print(f"Wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
