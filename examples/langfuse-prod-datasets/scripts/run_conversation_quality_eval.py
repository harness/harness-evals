#!/usr/bin/env python3
"""Categorize offline review batches without invoking the Harness agent.

Reads judge-input JSONL produced by ``build_review_batches.py`` (pre-captured
conversations). Assigns usefulness / agent quality / golden_readiness, and by
default also ``eval_candidate_score`` (0–5) for golden selection — all in one
pass. Does not run the live agent eval.

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

from conversation_candidate_score import (  # noqa: E402
    CRITERIA,
    PROMPT_VERSION as CANDIDATE_PROMPT_VERSION,
    HarnessConversationCandidateScoreMetric,
    conversation_from_eval_case,
)
from conversation_quality import (  # noqa: E402
    PROMPT_VERSION,
    HarnessConversationQualityMetric,
)
from conversation_signals import extract_structural_facts  # noqa: E402
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


def _blank_candidate_fields() -> dict[str, Any]:
    """Review (AIPLAT-952): Empty Round-4 columns for review.csv when candidate scoring is off."""
    row: dict[str, Any] = {
        "eval_candidate_score": "",
        "eval_candidate_hits": "",
        "eval_candidate_reasoning": "",
        "eval_candidate_prompt_version": "",
    }
    for name in CRITERIA:
        row[f"criterion_{name}"] = ""
    return row


def _candidate_fields_from_score(score_metadata: dict[str, Any], reason: str) -> dict[str, Any]:
    """Review (AIPLAT-952): Flatten HarnessConversationCandidateScoreMetric into CSV columns."""
    out = _blank_candidate_fields()
    if score_metadata.get("skipped"):
        out["eval_candidate_score"] = score_metadata.get("eval_candidate_score", 0.0)
        out["eval_candidate_hits"] = score_metadata.get("criteria_hits", 0)
        out["eval_candidate_reasoning"] = reason or ""
        return out
    criteria = score_metadata.get("criteria") or {}
    out["eval_candidate_score"] = score_metadata.get("eval_candidate_score", 0.0)
    out["eval_candidate_hits"] = score_metadata.get("criteria_hits", 0)
    out["eval_candidate_reasoning"] = reason or ""
    out["eval_candidate_prompt_version"] = score_metadata.get(
        "prompt_version", CANDIDATE_PROMPT_VERSION
    )
    for name in CRITERIA:
        out[f"criterion_{name}"] = "yes" if criteria.get(name) else "no"
    return out


def _session_stats_fields(case: EvalCase) -> dict[str, Any]:
    """Review (AIPLAT-952): session_turns / session_cost_usd for sorting review.csv by stress."""
    facts = extract_structural_facts(conversation_from_eval_case(case))
    num_turns = int(facts.get("num_turns") or 0)
    if num_turns == 0 and case.messages:
        num_turns = len(case.messages)
    total_cost = float(facts.get("total_cost_usd") or 0.0)
    case_meta = case.metadata or {}
    if total_cost == 0.0 and case_meta.get("total_cost_usd") is not None:
        total_cost = float(case_meta["total_cost_usd"])
    return {
        "session_turns": num_turns,
        "session_cost_usd": round(total_cost, 4),
    }


async def evaluate_cases(
    cases: list[EvalCase],
    quality_metric: HarnessConversationQualityMetric,
    candidate_metric: HarnessConversationCandidateScoreMetric | None,
    *,
    concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate_one(index: int, case: EvalCase) -> tuple[int, dict[str, Any]]:
        async with semaphore:
            quality_scores = await a_evaluate(case, [quality_metric])
            quality_score = quality_scores[0]
            quality_metadata = dict(quality_score.metadata or {})

            candidate_fields = _blank_candidate_fields()
            # Review (AIPLAT-952): Round 4 runs in same pass as Round 1; usefulness gates skip.
            if candidate_metric is not None:
                case_for_candidate = EvalCase.from_dict(case.to_dict())
                meta = dict(case_for_candidate.metadata or {})
                meta["usefulness"] = quality_metadata.get("usefulness", "useful")
                meta.setdefault("canonical_conversation", conversation_from_eval_case(case))
                case_for_candidate.metadata = meta
                candidate_scores = await a_evaluate(case_for_candidate, [candidate_metric])
                candidate_score = candidate_scores[0]
                candidate_fields = _candidate_fields_from_score(
                    dict(candidate_score.metadata or {}),
                    candidate_score.reason or "",
                )

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
            "usefulness": quality_metadata.get("usefulness", "useful"),
            "quality": quality_metadata.get("quality", "unclear"),
            "golden_readiness": quality_metadata.get("golden_readiness", "needs_rewrite"),
            "final_category": quality_metadata.get("final_category", "unclear"),
            "score": quality_score.value,
            "passed": quality_score.passed,
            "confidence": quality_metadata.get("confidence", 0.0),
            "goal_achievement": quality_metadata.get("goal_achievement"),
            "resolution": quality_metadata.get("resolution"),
            "tool_use_quality": quality_metadata.get("tool_use_quality"),
            "reasoning": quality_score.reason,
            "evidence": quality_metadata.get("evidence", []),
            "requires_chunked_evaluation": quality_metadata.get(
                "requires_chunked_evaluation", False
            ),
            "scoring_duration_ms": quality_score.scoring_duration_ms,
            "prompt_version": quality_metadata.get("prompt_version", PROMPT_VERSION),
            **_session_stats_fields(case),
            **candidate_fields,
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
            "eval_candidate_score": 0.0,
            "eval_candidate_hits": 0,
            "eval_candidate_reasoning": "",
            "eval_candidate_prompt_version": "",
            **{f"criterion_{name}": "no" for name in CRITERIA},
            "session_turns": "",
            "session_cost_usd": "",
        }
        for row in rows
    ]


def write_outputs(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    provider: str,
    model: str,
    with_candidate_score: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if with_candidate_score:
        rows = sorted(
            rows,
            key=lambda row: float(row.get("eval_candidate_score") or 0.0),
            reverse=True,
        )
    results_path = output_dir / "results.jsonl"
    with results_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    review_fields = [
        "conversation_id",
        "original_prompt",
        "environment",
        "module",
        "session_turns",
        "session_cost_usd",
        "usefulness",
        "quality",
        "golden_readiness",
        "tool_use_quality",
        "reasoning",
    ]
    if with_candidate_score:
        review_fields.extend(
            [
                "eval_candidate_score",
                "eval_candidate_hits",
                "eval_candidate_reasoning",
                *[f"criterion_{name}" for name in CRITERIA],
            ]
        )
    review_fields.extend(
        [
            "human_quality",
            "human_golden_readiness",
            "human_notes",
            "agreement",
        ]
    )
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
    summary: dict[str, Any] = {
        "total": len(rows),
        "provider": provider,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "with_candidate_score": with_candidate_score,
        "quality_counts": dict(sorted(qualities.items())),
        "golden_readiness_counts": dict(sorted(readiness.items())),
        # Backward-compatible alias of quality_counts (agent quality only).
        "category_counts": dict(sorted(qualities.items())),
        "module_counts": dict(sorted(modules.items())),
        "average_confidence": sum(confidences) / len(confidences) if confidences else None,
    }
    if with_candidate_score:
        scored = [
            float(row["eval_candidate_score"])
            for row in rows
            if row.get("eval_candidate_score") not in (None, "")
        ]
        summary["average_eval_candidate_score"] = (
            round(sum(scored) / len(scored), 2) if scored else None
        )
        summary["top_15_by_eval_candidate_score"] = [
            {
                "conversation_id": row.get("conversation_id"),
                "eval_candidate_score": float(row.get("eval_candidate_score") or 0.0),
                "eval_candidate_hits": row.get("eval_candidate_hits"),
                "quality": row.get("quality"),
                "golden_readiness": row.get("golden_readiness"),
            }
            for row in rows[:15]
        ]
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
    candidate_metric = None
    if args.with_candidate_score:
        candidate_metric = HarnessConversationCandidateScoreMetric(
            llm=llm,
            max_conversation_chars=args.max_conversation_chars,
        )
    judged_rows = await evaluate_cases(
        cases,
        metric,
        candidate_metric,
        concurrency=args.concurrency,
    )
    rows = judged_rows + deterministic_rows(ineligible)
    output_dir = args.output_dir or DATASET_ROOT / "results" / args.input.stem
    write_outputs(
        rows,
        output_dir,
        provider=args.provider,
        model=args.model,
        with_candidate_score=args.with_candidate_score,
    )
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
    parser.add_argument(
        "--with-candidate-score",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also score eval candidacy (0–5) in the same pass (default: on)",
    )
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.max_conversation_chars < 1:
        parser.error("--max-conversation-chars must be at least 1")
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
