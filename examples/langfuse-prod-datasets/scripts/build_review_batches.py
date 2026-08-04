#!/usr/bin/env python3
"""Build offline review batches from canonical production conversations.

This is **not** the live conversation eval. It packages fetched transcripts into
JSONL batches for the offline quality judge (categorization only). Under the
hood each eligible row is a harness-evals ``EvalCase`` so the judge runner can
reuse SDK validation — the rows are judge input, not live goldens.

The exporter never invokes the agent and never reads the Markdown review files.
It reads one ``*.conversation.json`` per session, applies structural eligibility
checks, validates eligible rows with ``EvalCase.from_dict()``, and writes:

* ``review-batches/<source>-<count>.jsonl`` — eligible rows for the LLM judge
* ``review-batches/<source>-<count>.ineligible.jsonl`` — structurally unusable cases

Usage:
  python scripts/build_review_batches.py --source module-coverage --limit 5
  python scripts/build_review_batches.py --source random --limit 10 --output /tmp/random-10.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DATASET_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from harness_evals.core.eval_case import EvalCase  # noqa: E402


def eligibility_reasons(conversation: dict[str, Any]) -> list[str]:
    """Return deterministic reasons this conversation cannot be judged."""
    reasons: list[str] = []
    messages = conversation.get("messages")
    if not isinstance(messages, list) or not messages:
        return ["missing_messages"]

    user_messages = [
        message
        for message in messages
        if isinstance(message, dict)
        and message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and message["content"].strip()
    ]
    assistant_messages = [
        message
        for message in messages
        if isinstance(message, dict)
        and message.get("role") == "assistant"
        and (
            (isinstance(message.get("content"), str) and message["content"].strip())
            or message.get("tool_calls")
        )
    ]
    if not user_messages:
        reasons.append("missing_meaningful_user_message")
    if not assistant_messages:
        reasons.append("missing_assistant_response")
    if not conversation.get("conversation_id"):
        reasons.append("missing_conversation_id")
    if not (conversation.get("metadata") or {}).get("trace_ids"):
        reasons.append("missing_trace_ids")
    return reasons


def build_eval_case(conversation: dict[str, Any], canonical_path: Path) -> EvalCase:
    metadata = dict(conversation.get("metadata") or {})
    metadata.update(
        {
            "conversation_id": conversation["conversation_id"],
            "sample_type": conversation.get("sample_type"),
            "canonical_file": canonical_path.name,
            "conversation_schema_version": conversation.get("schema_version"),
            "eligibility": "eligible",
        }
    )
    tags = {
        key: str(value)
        for key, value in {
            "environment": metadata.get("environment"),
            "module": metadata.get("module"),
            "sample_type": conversation.get("sample_type"),
        }.items()
        if value not in (None, "")
    }
    payload = {
        "input": conversation.get("input") or "",
        "output": conversation.get("output") or "",
        "messages": conversation.get("messages") or [],
        "tool_calls": conversation.get("tool_calls") or [],
        "metadata": metadata,
        "tags": tags,
    }
    return EvalCase.from_dict(payload)


def default_output(source: str, limit: int) -> Path:
    return DATASET_ROOT / "review-batches" / f"{source}-{limit:03d}.jsonl"


def export_dataset(source: str, limit: int, output: Path) -> tuple[int, int]:
    source_dir = DATASET_ROOT / source
    canonical_paths = sorted(source_dir.glob("*.conversation.json"))
    if not canonical_paths:
        raise FileNotFoundError(
            f"No canonical conversations found in {source_dir}. "
            "Run build_agent_transcripts.py --backfill-conversations first."
        )
    selected_paths = canonical_paths[: min(limit, len(canonical_paths))]

    eligible_rows: list[dict[str, Any]] = []
    ineligible_rows: list[dict[str, Any]] = []
    for canonical_path in selected_paths:
        conversation = json.loads(canonical_path.read_text())
        reasons = eligibility_reasons(conversation)
        if reasons:
            ineligible_rows.append(
                {
                    "conversation_id": conversation.get("conversation_id"),
                    "original_prompt": conversation.get("input") or "",
                    "canonical_file": canonical_path.name,
                    "usefulness": "useless",
                    "quality": "not_applicable",
                    "golden_readiness": "needs_rewrite",
                    "reasons": reasons,
                }
            )
            continue
        eval_case = build_eval_case(conversation, canonical_path)
        eligible_rows.append(eval_case.to_dict())

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in eligible_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    ineligible_path = output.with_suffix(".ineligible.jsonl")
    with ineligible_path.open("w") as handle:
        for row in ineligible_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(eligible_rows), len(ineligible_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("random", "module-coverage"), required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")
    output = args.output or default_output(args.source, args.limit)
    eligible, ineligible = export_dataset(args.source, args.limit, output)
    print(f"Wrote {eligible} eligible review rows to {output}")
    print(f"Wrote {ineligible} ineligible rows to {output.with_suffix('.ineligible.jsonl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
