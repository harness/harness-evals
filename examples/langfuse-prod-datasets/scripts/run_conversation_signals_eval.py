#!/usr/bin/env python3
"""Round 3: LLM weakness / coverage signal tags for good/bad conversations.

Runs **after** ``run_conversation_quality_eval.py`` (Round 1). Only rows with
agent ``quality`` (or ``human_quality``) of ``good`` or ``bad`` are judged.
``unclear`` / ``useless`` rows are copied through with ``signals_skipped_reason``.

Round 1 does **not** produce these tags — it only assigns usefulness / quality /
golden_readiness. Round 3 is a separate LLM categorization into stress buckets
(high_turns, skill_loading, hitl_loop, etc.).

Usage:
  OPENAI_API_KEY=... python scripts/run_conversation_signals_eval.py \\
      --review results/module-coverage-200/review.csv \\
      --conversations module-coverage \\
      --provider openai --model gpt-4o

  OPENAI_API_KEY=... python scripts/run_conversation_signals_eval.py \\
      --review results/module-coverage-200/review.csv \\
      --review results/random-200/review.csv \\
      --conversations . \\
      --output results/review-with-signals.csv \\
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

from conversation_signals import (  # noqa: E402
    PROMPT_VERSION,
    HarnessConversationSignalsMetric,
    extract_structural_facts,
    quality_eligible_for_signals,
    resolve_quality,
)
from harness_evals.config.runner import build_llm  # noqa: E402
from harness_evals.config.schema import ModelSpec  # noqa: E402
from harness_evals.core.eval_case import EvalCase  # noqa: E402
from harness_evals.core.runner import a_evaluate  # noqa: E402

SIGNAL_COLUMNS = [
    "signal_tags",
    "scenario_type",
    "module_tag",
    "signals_confidence",
    "signals_reasoning",
    "struct_num_turns",
    "struct_total_cost_usd",
    "struct_num_tool_calls",
    "struct_max_tool_output_bytes",
    "struct_truncated_tool_outputs",
    "signals_skipped_reason",
    "signals_prompt_version",
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
    *,
    quality: str,
) -> EvalCase:
    facts = extract_structural_facts(conversation)
    metadata = dict(conversation.get("metadata") or {})
    metadata.update(
        {
            "conversation_id": conversation.get("conversation_id") or row.get("conversation_id"),
            "round1_quality": quality,
            "canonical_conversation": conversation,
            "module": facts["module"],
            "num_turns": facts["num_turns"],
            "total_cost_usd": facts["total_cost_usd"],
            "num_tool_calls": facts["num_tool_calls"],
            "truncated_tool_outputs": facts["truncated_tool_outputs"],
            "max_tool_output_bytes": facts["max_tool_output_bytes"],
            "tool_names_sample": facts["tool_names_sample"],
        }
    )
    tags = {
        key: str(value)
        for key, value in {
            "environment": metadata.get("environment") or row.get("environment"),
            "module": facts["module"],
            "round1_quality": quality,
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


def _blank_signal_fields(reason: str = "") -> dict[str, str]:
    return {
        "signal_tags": "",
        "scenario_type": "",
        "module_tag": "",
        "signals_confidence": "",
        "signals_reasoning": "",
        "struct_num_turns": "",
        "struct_total_cost_usd": "",
        "struct_num_tool_calls": "",
        "struct_max_tool_output_bytes": "",
        "struct_truncated_tool_outputs": "",
        "signals_skipped_reason": reason,
        "signals_prompt_version": PROMPT_VERSION,
    }


def _apply_facts(out: dict[str, str], facts: dict[str, Any]) -> None:
    out["struct_num_turns"] = str(facts.get("num_turns") or "")
    out["struct_total_cost_usd"] = str(facts.get("total_cost_usd") or "")
    out["struct_num_tool_calls"] = str(facts.get("num_tool_calls") or "")
    out["struct_max_tool_output_bytes"] = str(facts.get("max_tool_output_bytes") or "")
    out["struct_truncated_tool_outputs"] = str(facts.get("truncated_tool_outputs") or "")


async def evaluate_eligible(
    cases: list[tuple[int, EvalCase]],
    metric: HarnessConversationSignalsMetric,
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
    for column in SIGNAL_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    tagged = [row for row in rows if row.get("signal_tags")]
    tag_counts: dict[str, int] = {}
    for row in tagged:
        for tag in (row.get("signal_tags") or "").split(";"):
            if tag:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
    skipped_counts = Counter(
        row.get("signals_skipped_reason") or "tagged"
        for row in rows
        if row.get("signal_tags") or row.get("signals_skipped_reason")
    )
    return {
        "total_rows": len(rows),
        "tagged_rows": len(tagged),
        "prompt_version": PROMPT_VERSION,
        "tag_counts": dict(sorted(tag_counts.items())),
        "skipped_counts": dict(sorted(skipped_counts.items())),
    }


async def async_main(args: argparse.Namespace) -> int:
    review_paths = [path.resolve() for path in args.review]
    for path in review_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Review CSV not found: {path}")

    output_path = args.output
    if output_path is None:
        if len(review_paths) == 1:
            output_path = review_paths[0].with_name("review-with-signals.csv")
        else:
            output_path = DATASET_ROOT / "results" / "review-with-signals.csv"
    output_path = output_path.resolve()

    conversation_index = index_conversations(args.conversations.resolve())
    review_rows = load_review_rows(review_paths)

    eligible_cases: list[tuple[int, EvalCase]] = []
    prepared: list[dict[str, str]] = []
    facts_by_index: dict[int, dict[str, Any]] = {}

    for index, row in enumerate(review_rows):
        out = dict(row)
        out.update(_blank_signal_fields())
        quality = resolve_quality(row)
        if not quality_eligible_for_signals(quality):
            out["signals_skipped_reason"] = f"quality={quality or 'missing'}"
            prepared.append(out)
            continue

        cid = (row.get("conversation_id") or "").strip()
        canonical_path = conversation_index.get(cid)
        if canonical_path is None:
            out["signals_skipped_reason"] = "missing_canonical_conversation"
            prepared.append(out)
            continue

        try:
            conversation = json.loads(canonical_path.read_text())
        except (OSError, json.JSONDecodeError):
            out["signals_skipped_reason"] = "invalid_canonical_conversation"
            prepared.append(out)
            continue

        facts = extract_structural_facts(conversation)
        facts_by_index[index] = facts
        _apply_facts(out, facts)
        prepared.append(out)
        eligible_cases.append((index, build_eval_case(row, conversation, quality=quality)))

    print(
        f"Round 3 signals: {len(eligible_cases)} eligible good/bad rows "
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
        metric = HarnessConversationSignalsMetric(llm=llm)
        judged = await evaluate_eligible(
            eligible_cases,
            metric,
            concurrency=args.concurrency,
        )
        for index, payload in judged.items():
            meta = payload["metadata"]
            out = prepared[index]
            if meta.get("skipped"):
                out["signals_skipped_reason"] = str(
                    meta.get("signals_skipped_reason") or "skipped"
                )
                continue
            tags = meta.get("signal_tags") or []
            out["signal_tags"] = ";".join(str(tag) for tag in tags)
            out["scenario_type"] = str(meta.get("scenario_type") or "")
            out["module_tag"] = str(meta.get("module_tag") or "")
            out["signals_confidence"] = str(meta.get("confidence") or "")
            out["signals_reasoning"] = str(payload.get("reason") or "")
            out["signals_skipped_reason"] = ""
            out["signals_prompt_version"] = str(
                meta.get("prompt_version") or PROMPT_VERSION
            )
            facts = facts_by_index.get(index) or meta.get("structural_facts") or {}
            _apply_facts(out, facts)

    write_csv(prepared, output_path)
    results_path = output_path.with_suffix(".jsonl")
    with results_path.open("w") as handle:
        for row in prepared:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(prepared)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote {output_path} ({summary['tagged_rows']}/{summary['total_rows']} rows tagged)")
    print(f"Results JSONL: {results_path}")
    print(f"Summary: {summary_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Round 3 LLM weakness/coverage signal tags for good/bad conversations."
    )
    parser.add_argument(
        "--review",
        action="append",
        type=Path,
        required=True,
        help="Review CSV from Round 1 run_conversation_quality_eval.py (repeatable)",
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
        help="Output CSV (default: sibling review-with-signals.csv for single --review)",
    )
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Resolve eligible rows without calling the LLM",
    )
    args = parser.parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
