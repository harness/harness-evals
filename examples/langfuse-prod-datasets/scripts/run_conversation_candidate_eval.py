#!/usr/bin/env python3
"""Round 4: eval candidate score (0–5) for golden selection.

Runs after Round 1 quality categorization. Scores each non-useless row using
eleven equally-weighted criteria (high_turns, high_cost, tool_failure, etc.).
Higher score = more valuable eval golden candidate.

Usage:
  OPENAI_API_KEY=... python scripts/run_conversation_candidate_eval.py \\
      --review results/module-coverage-200/review.csv \\
      --review results/random-200/review.csv \\
      --conversations . \\
      --output results/review-with-candidate-scores.csv \\
      --provider openai --model gpt-4o
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

DATASET_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
METRICS_DIR = DATASET_ROOT / "metrics"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(METRICS_DIR))

from conversation_candidate_score import (  # noqa: E402
    CRITERIA,
    PROMPT_VERSION,
    HarnessConversationCandidateScoreMetric,
    usefulness_eligible,
)
from conversation_signals import extract_structural_facts, resolve_quality  # noqa: E402
from harness_evals.config.runner import build_llm  # noqa: E402
from harness_evals.config.schema import ModelSpec  # noqa: E402
from harness_evals.core.eval_case import EvalCase  # noqa: E402
from harness_evals.core.runner import a_evaluate  # noqa: E402

SCORE_COLUMNS = [
    "eval_candidate_score",
    "eval_candidate_hits",
    "eval_candidate_reasoning",
    "eval_candidate_skipped_reason",
    "eval_candidate_prompt_version",
    *[f"criterion_{name}" for name in CRITERIA],
    "struct_num_turns",
    "struct_total_cost_usd",
    "struct_num_tool_calls",
    "struct_max_tool_output_bytes",
    "struct_truncated_tool_outputs",
]


def load_review_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                cid = (row.get("conversation_id") or "").strip()
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                rows.append(dict(row))
    if not rows:
        raise ValueError("No review rows loaded")
    return rows


def index_conversations(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(root.rglob("*.conversation.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        cid = str(payload.get("conversation_id") or "").strip()
        if cid and cid not in index:
            index[cid] = path
    return index


def build_eval_case(
    row: dict[str, str],
    conversation: dict[str, Any],
) -> EvalCase:
    facts = extract_structural_facts(conversation)
    metadata = dict(conversation.get("metadata") or {})
    metadata.update(
        {
            "conversation_id": conversation.get("conversation_id") or row.get("conversation_id"),
            "usefulness": row.get("usefulness") or "useful",
            "round1_quality": resolve_quality(row),
            "canonical_conversation": conversation,
            "module": facts["module"],
        }
    )
    tags = {
        key: str(value)
        for key, value in {
            "environment": metadata.get("environment") or row.get("environment"),
            "module": facts["module"],
        }.items()
        if value not in (None, "")
    }
    return EvalCase.from_dict(
        {
            "input": conversation.get("input") or row.get("original_prompt") or "",
            "output": conversation.get("output") or "",
            "messages": conversation.get("messages") or [],
            "tool_calls": conversation.get("tool_calls") or [],
            "metadata": metadata,
            "tags": tags,
        }
    )


def _blank_score_fields(reason: str = "") -> dict[str, str]:
    out = {
        "eval_candidate_score": "",
        "eval_candidate_hits": "",
        "eval_candidate_reasoning": "",
        "eval_candidate_skipped_reason": reason,
        "eval_candidate_prompt_version": PROMPT_VERSION,
    }
    for name in CRITERIA:
        out[f"criterion_{name}"] = ""
    for column in (
        "struct_num_turns",
        "struct_total_cost_usd",
        "struct_num_tool_calls",
        "struct_max_tool_output_bytes",
        "struct_truncated_tool_outputs",
    ):
        out[column] = ""
    return out


def _apply_facts(out: dict[str, str], facts: dict[str, Any]) -> None:
    out["struct_num_turns"] = str(facts.get("num_turns") or "")
    out["struct_total_cost_usd"] = str(facts.get("total_cost_usd") or "")
    out["struct_num_tool_calls"] = str(facts.get("num_tool_calls") or "")
    out["struct_max_tool_output_bytes"] = str(facts.get("max_tool_output_bytes") or "")
    out["struct_truncated_tool_outputs"] = str(facts.get("truncated_tool_outputs") or "")


def _apply_score(out: dict[str, str], payload: dict[str, Any], facts: dict[str, Any]) -> None:
    meta = payload["metadata"]
    if meta.get("skipped"):
        out["eval_candidate_skipped_reason"] = str(meta.get("skip_reason") or "skipped")
        out["eval_candidate_score"] = str(meta.get("eval_candidate_score") or "0")
        out["eval_candidate_hits"] = str(meta.get("criteria_hits") or "0")
        return
    criteria = meta.get("criteria") or {}
    out["eval_candidate_score"] = str(meta.get("eval_candidate_score") or payload["score"].value)
    out["eval_candidate_hits"] = str(meta.get("criteria_hits") or "")
    out["eval_candidate_reasoning"] = str(payload.get("reason") or "")
    out["eval_candidate_skipped_reason"] = ""
    out["eval_candidate_prompt_version"] = str(meta.get("prompt_version") or PROMPT_VERSION)
    for name in CRITERIA:
        out[f"criterion_{name}"] = "yes" if criteria.get(name) else "no"
    _apply_facts(out, facts)


async def evaluate_eligible(
    cases: list[tuple[int, EvalCase]],
    metric: HarnessConversationCandidateScoreMetric,
    *,
    concurrency: int,
) -> dict[int, dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)
    results: dict[int, dict[str, Any]] = {}

    async def evaluate_one(index: int, case: EvalCase) -> None:
        async with semaphore:
            scores = await a_evaluate(case, [metric])
        score = scores[0]
        results[index] = {
            "score": score,
            "metadata": dict(score.metadata or {}),
            "reason": score.reason,
        }

    await asyncio.gather(*(evaluate_one(index, case) for index, case in cases))
    return results


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    for column in SCORE_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    scored = [
        row
        for row in rows
        if row.get("eval_candidate_score") and not row.get("eval_candidate_skipped_reason")
    ]
    scores = [float(row["eval_candidate_score"]) for row in scored]
    skipped = Counter(row.get("eval_candidate_skipped_reason") or "scored" for row in rows)
    return {
        "total_rows": len(rows),
        "scored_rows": len(scored),
        "prompt_version": PROMPT_VERSION,
        "average_score": round(sum(scores) / len(scores), 2) if scores else None,
        "max_score": max(scores) if scores else None,
        "skipped_counts": dict(sorted(skipped.items())),
        "top_15": sorted(
            (
                {
                    "conversation_id": row.get("conversation_id"),
                    "eval_candidate_score": float(row["eval_candidate_score"]),
                    "eval_candidate_hits": row.get("eval_candidate_hits"),
                    "quality": row.get("quality"),
                    "golden_readiness": row.get("golden_readiness"),
                }
                for row in scored
            ),
            key=lambda item: item["eval_candidate_score"],
            reverse=True,
        )[:15],
    }


async def async_main(args: argparse.Namespace) -> int:
    review_paths = [path.resolve() for path in args.review]
    for path in review_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Review CSV not found: {path}")

    output_path = args.output
    if output_path is None:
        if len(review_paths) == 1:
            output_path = review_paths[0].with_name("review-with-candidate-scores.csv")
        else:
            output_path = DATASET_ROOT / "results" / "review-with-candidate-scores.csv"
    output_path = output_path.resolve()

    conversation_index = index_conversations(args.conversations.resolve())
    review_rows = load_review_rows(review_paths)

    eligible_cases: list[tuple[int, EvalCase]] = []
    prepared: list[dict[str, str]] = []
    facts_by_index: dict[int, dict[str, Any]] = {}

    for index, row in enumerate(review_rows):
        out = dict(row)
        out.update(_blank_score_fields())
        usefulness = str(row.get("usefulness") or "useful").strip().lower()
        if not usefulness_eligible(usefulness):
            out["eval_candidate_skipped_reason"] = "useless"
            out["eval_candidate_score"] = "0"
            prepared.append(out)
            continue

        cid = (row.get("conversation_id") or "").strip()
        canonical_path = conversation_index.get(cid)
        if canonical_path is None:
            out["eval_candidate_skipped_reason"] = "missing_canonical_conversation"
            prepared.append(out)
            continue

        try:
            conversation = json.loads(canonical_path.read_text())
        except (OSError, json.JSONDecodeError):
            out["eval_candidate_skipped_reason"] = "invalid_canonical_conversation"
            prepared.append(out)
            continue

        facts = extract_structural_facts(conversation)
        facts_by_index[index] = facts
        _apply_facts(out, facts)
        prepared.append(out)
        eligible_cases.append((index, build_eval_case(row, conversation)))

    print(
        f"Round 4 candidate scoring: {len(eligible_cases)} eligible rows "
        f"of {len(review_rows)} review rows "
        f"({len(review_rows) - len(eligible_cases)} skipped)."
    )
    if args.validate_only:
        return 0

    if eligible_cases:
        llm = build_llm(
            ModelSpec(
                provider=args.provider,
                name=args.model,
            )
        )
        metric = HarnessConversationCandidateScoreMetric(
            llm=llm,
            max_conversation_chars=args.max_conversation_chars,
        )
        judged = await evaluate_eligible(
            eligible_cases,
            metric,
            concurrency=args.concurrency,
        )
        for index, payload in judged.items():
            facts = facts_by_index.get(index) or {}
            _apply_score(prepared[index], payload, facts)

    write_csv(prepared, output_path)
    results_path = output_path.with_suffix(".jsonl")
    with results_path.open("w") as handle:
        for row in prepared:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(prepared)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(
        f"Wrote {output_path} ({summary['scored_rows']}/{summary['total_rows']} rows scored)"
    )
    print(f"Average score: {summary['average_score']}")
    print(f"Results JSONL: {results_path}")
    print(f"Summary: {summary_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Round 4 eval candidate score (0–5) for golden selection."
    )
    parser.add_argument(
        "--review",
        action="append",
        type=Path,
        required=True,
        help="Review CSV from Round 1 (repeatable)",
    )
    parser.add_argument(
        "--conversations",
        type=Path,
        default=DATASET_ROOT,
        help="Root to search for *.conversation.json (default: dataset root)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV (default: sibling review-with-candidate-scores.csv)",
    )
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-conversation-chars", type=int, default=120_000)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Resolve eligible rows without calling the LLM",
    )
    args = parser.parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
