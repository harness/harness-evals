#!/usr/bin/env python3
"""Categorize offline review batches without invoking the Harness agent.

Reads judge-input JSONL produced by ``build_review_batches.py`` (pre-captured
conversations). This step only assigns usefulness / agent quality /
golden_readiness categories for golden curation — it does not run the live
agent eval.

The script uses the Harness Evals SDK's EvalCase, metric, Score, and async
evaluation paths. Human labels/notes are never added to the judge prompt.

Usage:
  python scripts/run_conversation_quality_eval.py \
      --input review-batches/module-coverage-005.jsonl --validate-only

  OPENAI_API_KEY=... python scripts/run_conversation_quality_eval.py \
      --input review-batches/module-coverage-005.jsonl \
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

from conversation_quality import (  # noqa: E402
    PROMPT_VERSION,
    HarnessConversationQualityMetric,
)
from harness_evals.config.runner import build_llm  # noqa: E402
from harness_evals.config.schema import ModelSpec  # noqa: E402
from harness_evals.core.eval_case import EvalCase  # noqa: E402
from harness_evals.core.runner import a_evaluate  # noqa: E402


def load_eval_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                cases.append(EvalCase.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid EvalCase at {path}:{line_number}: {error}") from error
    if not cases:
        raise ValueError(f"No EvalCases found in {path}")
    return cases


def load_ineligible(path: Path) -> list[dict[str, Any]]:
    ineligible_path = path.with_suffix(".ineligible.jsonl")
    if not ineligible_path.is_file():
        return []
    with ineligible_path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


async def evaluate_cases(
    cases: list[EvalCase],
    metric: HarnessConversationQualityMetric,
    *,
    concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate_one(index: int, case: EvalCase) -> tuple[int, dict[str, Any]]:
        async with semaphore:
            scores = await a_evaluate(case, [metric])
        score = scores[0]
        score_metadata = dict(score.metadata or {})
        case_metadata = case.metadata or {}
        row = {
            "conversation_id": case_metadata.get("conversation_id"),
            "original_prompt": (
                case.input
                if isinstance(case.input, str)
                else json.dumps(case.input, ensure_ascii=False)
            ),
            "environment": case_metadata.get("environment"),
            "module": case_metadata.get("module"),
            "canonical_file": case_metadata.get("canonical_file"),
            "usefulness": score_metadata.get("usefulness", "useful"),
            "quality": score_metadata.get("quality", "unclear"),
            "golden_readiness": score_metadata.get("golden_readiness", "needs_rewrite"),
            "final_category": score_metadata.get("final_category", "unclear"),
            "score": score.value,
            "passed": score.passed,
            "confidence": score_metadata.get("confidence", 0.0),
            "goal_achievement": score_metadata.get("goal_achievement"),
            "resolution": score_metadata.get("resolution"),
            "tool_use_quality": score_metadata.get("tool_use_quality"),
            "reasoning": score.reason,
            "evidence": score_metadata.get("evidence", []),
            "requires_chunked_evaluation": score_metadata.get("requires_chunked_evaluation", False),
            "scoring_duration_ms": score.scoring_duration_ms,
            "prompt_version": score_metadata.get("prompt_version", PROMPT_VERSION),
        }
        return index, row

    indexed = await asyncio.gather(*(evaluate_one(index, case) for index, case in enumerate(cases)))
    return [row for _, row in sorted(indexed)]


def deterministic_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "conversation_id": row.get("conversation_id"),
            "original_prompt": row.get("original_prompt", ""),
            "environment": None,
            "module": None,
            "canonical_file": row.get("canonical_file"),
            "usefulness": "useless",
            "quality": "not_applicable",
            "golden_readiness": "needs_rewrite",
            "final_category": "useless",
            "score": 0.0,
            "passed": False,
            "confidence": 1.0,
            "goal_achievement": None,
            "resolution": None,
            "tool_use_quality": None,
            "reasoning": ", ".join(row.get("reasons") or ["structurally ineligible"]),
            "evidence": [],
            "requires_chunked_evaluation": False,
            "scoring_duration_ms": 0.0,
            "prompt_version": "deterministic-eligibility-v1",
        }
        for row in rows
    ]


def write_outputs(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    provider: str,
    model: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    with results_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    review_fields = [
        "conversation_id",
        "original_prompt",
        "environment",
        "module",
        "usefulness",
        "quality",
        "golden_readiness",
        "final_category",
        "confidence",
        "goal_achievement",
        "resolution",
        "tool_use_quality",
        "reasoning",
        "human_quality",
        "human_golden_readiness",
        "human_notes",
        "agreement",
    ]
    with (output_dir / "review.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{field: row.get(field, "") for field in review_fields},
                    "human_quality": "",
                    "human_golden_readiness": "",
                    "human_notes": "",
                    "agreement": "",
                }
            )

    qualities = Counter(str(row.get("quality") or "unknown") for row in rows)
    readiness = Counter(str(row.get("golden_readiness") or "unknown") for row in rows)
    modules = Counter(str(row.get("module") or "unknown") for row in rows)
    confidences = [float(row["confidence"]) for row in rows if row.get("confidence") is not None]
    summary = {
        "total": len(rows),
        "provider": provider,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "quality_counts": dict(sorted(qualities.items())),
        "golden_readiness_counts": dict(sorted(readiness.items())),
        # Backward-compatible alias of quality_counts (agent quality only).
        "category_counts": dict(sorted(qualities.items())),
        "module_counts": dict(sorted(modules.items())),
        "average_confidence": sum(confidences) / len(confidences) if confidences else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


async def async_main(args: argparse.Namespace) -> int:
    cases = load_eval_cases(args.input)
    ineligible = load_ineligible(args.input)
    print(f"Validated {len(cases)} EvalCases; found {len(ineligible)} deterministic ineligible cases.")
    if args.validate_only:
        return 0

    llm = build_llm(
        ModelSpec(
            provider=args.provider,
            name=args.model,
            params={"temperature": args.temperature},
        )
    )
    metric = HarnessConversationQualityMetric(
        llm=llm,
        max_conversation_chars=args.max_conversation_chars,
    )
    judged_rows = await evaluate_cases(cases, metric, concurrency=args.concurrency)
    rows = judged_rows + deterministic_rows(ineligible)
    output_dir = args.output_dir or DATASET_ROOT / "results" / args.input.stem
    write_outputs(rows, output_dir, provider=args.provider, model=args.model)
    print(f"Wrote {len(rows)} categorized results to {output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--provider", choices=("openai", "anthropic", "harness"), default="openai")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-conversation-chars", type=int, default=120_000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.max_conversation_chars < 1:
        parser.error("--max-conversation-chars must be at least 1")
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
