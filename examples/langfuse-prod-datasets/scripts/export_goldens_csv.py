#!/usr/bin/env python3
"""Build / refresh ``goldens.csv`` inventory from live goldens + ready candidates.

Sources (in priority order for overlapping fields):

1. ``examples/prod-conversation.goldens.jsonl`` — rows already promoted
2. ``examples/prod-conversation.goldens.manifest.jsonl`` — scenario_type / action
3. Any ``results/*/results.jsonl`` — confidence, score, reasoning, canonical_file
4. Optional extra ``--results`` paths — append new ``ready`` candidates,
   regardless of agent quality

Usage:
  python scripts/export_goldens_csv.py

  python scripts/export_goldens_csv.py \\
      --results results/module-coverage-200/results.jsonl \\
      --results results/random-200/results.jsonl \\
      --output goldens.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

DATASET_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = DATASET_ROOT.parent
DEFAULT_GOLDENS = EXAMPLES_ROOT / "prod-conversation.goldens.jsonl"
DEFAULT_MANIFEST = EXAMPLES_ROOT / "prod-conversation.goldens.manifest.jsonl"
DEFAULT_OUTPUT = DATASET_ROOT / "goldens.csv"
RESULTS_ROOT = DATASET_ROOT / "results"

COLUMNS = [
    "golden_id",
    "conversation_id",
    "original_prompt",
    "module",
    "environment",
    "dataset_source",
    "judge_category",
    "golden_readiness",
    "status",
    "scenario_type",
    "portability_action",
    "date_added",
    "date_promoted",
    "confidence",
    "score",
    "goal_achievement",
    "resolution",
    "tool_use_quality",
    "turns",
    "canonical_file",
    "source_file",
    "expected_outcome",
    "reasoning",
    "eval_candidate_score",
    "eval_candidate_hits",
    "notes",
]


def _today() -> str:
    return date.today().isoformat()


def _slug(text: str, max_len: int = 40) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").lower()).strip("-")
    return (cleaned or "prompt")[:max_len].rstrip("-")


def _provisional_golden_id(module: str, conversation_id: str, prompt: str) -> str:
    short = (conversation_id or "unknown").split("-")[0]
    return f"{module or 'none'}-{short}-{_slug(prompt)}"


def _prompt_from_golden(golden: dict[str, Any]) -> str:
    turns = golden.get("turns") or []
    if turns and isinstance(turns[0], dict) and turns[0].get("content"):
        return str(turns[0]["content"])
    return str(golden.get("scenario") or "")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _index_results(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Map conversation_id -> richest judge result found."""
    by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            continue
        dataset = path.parent.name
        for row in _load_jsonl(path):
            conv_id = str(row.get("conversation_id") or "")
            if not conv_id:
                continue
            enriched = dict(row)
            enriched["_dataset"] = dataset
            enriched["_results_path"] = str(path)
            existing = by_id.get(conv_id)
            # Prefer rows that have reasoning / confidence already filled.
            if existing is None or (
                not existing.get("reasoning") and enriched.get("reasoning")
            ):
                by_id[conv_id] = enriched
    return by_id


def _index_candidate_scores(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Map conversation_id -> Round 4 eval candidate score row."""
    by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            continue
        for row in _load_jsonl(path):
            conv_id = str(row.get("conversation_id") or "")
            if not conv_id:
                continue
            by_id[conv_id] = row
    return by_id


def _discover_candidate_score_paths(extra: list[Path]) -> list[Path]:
    candidates: list[Path] = [
        DATASET_ROOT / "results" / "review-with-candidate-scores.jsonl"
    ]
    if RESULTS_ROOT.is_dir():
        candidates.extend(sorted(RESULTS_ROOT.glob("*/review-with-candidate-scores.jsonl")))
    seen: set[Path] = set()
    paths: list[Path] = []
    for path in [*candidates, *extra]:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        paths.append(resolved)
    return paths


def _discover_result_paths(extra: list[Path]) -> list[Path]:
    discovered = sorted(RESULTS_ROOT.glob("*/results.jsonl")) if RESULTS_ROOT.is_dir() else []
    seen: set[Path] = set()
    paths: list[Path] = []
    for path in [*discovered, *extra]:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        paths.append(path)
    return paths


def _blank_row() -> dict[str, str]:
    return {column: "" for column in COLUMNS}


def _conversation_path(canonical_file: str, dataset_source: str) -> Path | None:
    if not canonical_file:
        return None
    # dataset_source may be "module-coverage-200" / "random-200" / goldens filename
    for name in (
        dataset_source,
        dataset_source.rsplit("-", 1)[0] if "-" in dataset_source else "",
        "module-coverage",
        "random",
    ):
        if not name or name.endswith(".jsonl"):
            continue
        path = DATASET_ROOT / name / canonical_file
        if path.is_file():
            return path
    for name in ("module-coverage", "random"):
        path = DATASET_ROOT / name / canonical_file
        if path.is_file():
            return path
    return None


def _enrich_from_conversation(row: dict[str, str]) -> None:
    path = _conversation_path(row.get("canonical_file") or "", row.get("dataset_source") or "")
    if path is None:
        return
    try:
        conversation = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    messages = conversation.get("messages") or []
    if not row.get("turns"):
        row["turns"] = str(len(messages))
    if not row.get("original_prompt") and conversation.get("input"):
        row["original_prompt"] = str(conversation["input"])
    meta = conversation.get("metadata") or {}
    if not row.get("module") and meta.get("module"):
        row["module"] = str(meta["module"])
    if not row.get("environment") and meta.get("environment"):
        row["environment"] = str(meta["environment"])


def _infer_scenario_type(row: dict[str, str]) -> str:
    prompt = (row.get("original_prompt") or "").lower()
    if any(token in prompt for token in ("create", "add", "update", "delete", "toggle", "deploy")):
        return "write"
    if any(token in prompt for token in ("list", "what", "how", "explain", "show", "get", "find")):
        return "read_only"
    return "unknown"


def _apply_judge_fields(row: dict[str, str], judge: dict[str, Any] | None) -> None:
    if not judge:
        return
    for key in (
        "confidence",
        "score",
        "goal_achievement",
        "resolution",
        "tool_use_quality",
        "reasoning",
        "canonical_file",
        "eval_candidate_score",
        "eval_candidate_hits",
    ):
        value = judge.get(key)
        if value is not None and str(value).strip() != "":
            row[key] = value if not isinstance(value, float) else round(value, 4)
    if not row["judge_category"] and judge.get("quality"):
        row["judge_category"] = str(judge["quality"])
    elif not row["judge_category"] and judge.get("final_category"):
        # v2/v3 alias: final_category is agent quality (or useless).
        final = str(judge["final_category"])
        row["judge_category"] = "good" if final == "needs_improvement" else final
    if not row.get("golden_readiness"):
        readiness = str(judge.get("golden_readiness") or "")
        if readiness in {"unsuitable", "not_applicable"}:
            readiness = "needs_rewrite"
        if readiness in {"ready", "needs_rewrite"}:
            row["golden_readiness"] = readiness
        elif str(judge.get("final_category") or "") == "needs_improvement":
            row["golden_readiness"] = "needs_rewrite"
        elif str(judge.get("final_category") or "") == "good":
            row["golden_readiness"] = "ready"
    if not row["module"] and judge.get("module"):
        row["module"] = str(judge["module"])
    if not row["environment"] and judge.get("environment"):
        row["environment"] = str(judge["environment"])
    if not row["original_prompt"] and judge.get("original_prompt"):
        row["original_prompt"] = str(judge["original_prompt"])


def _apply_candidate_score_fields(
    row: dict[str, str], candidate: dict[str, Any] | None
) -> None:
    if not candidate:
        return
    score = candidate.get("eval_candidate_score")
    if score is not None and str(score).strip() != "":
        row["eval_candidate_score"] = score if not isinstance(score, float) else round(score, 2)
    hits = candidate.get("eval_candidate_hits")
    if hits is not None and str(hits).strip() != "":
        row["eval_candidate_hits"] = str(hits)


def build_rows(
    *,
    goldens_path: Path,
    manifest_path: Path,
    candidate_result_paths: list[Path],
    all_result_paths: list[Path],
    candidate_score_paths: list[Path],
    date_added: str,
) -> list[dict[str, str]]:
    goldens = _load_jsonl(goldens_path)
    manifest_by_id = {
        str(row.get("conversation_id") or ""): row for row in _load_jsonl(manifest_path)
    }
    judge_by_id = _index_results(all_result_paths)
    candidate_score_by_id = _index_candidate_scores(candidate_score_paths)

    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    # 1) Already-promoted goldens
    for golden in goldens:
        meta = golden.get("metadata") or {}
        tags = golden.get("tags") or {}
        conv_id = str(meta.get("source_conversation_id") or "")
        if not conv_id or conv_id in seen:
            continue
        seen.add(conv_id)

        manifest = manifest_by_id.get(conv_id) or {}
        judge = judge_by_id.get(conv_id)
        turns = golden.get("turns") or []

        row = _blank_row()
        row.update(
            {
                "golden_id": str(golden.get("id") or manifest.get("golden_id") or ""),
                "conversation_id": conv_id,
                "original_prompt": _prompt_from_golden(golden),
                "module": str(tags.get("module") or manifest.get("module") or ""),
                "environment": str(
                    tags.get("environment") or manifest.get("environment") or ""
                ),
                "dataset_source": goldens_path.name,
                "judge_category": str(
                    tags.get("judge_category") or manifest.get("judge_category") or ""
                ),
                "golden_readiness": str(
                    tags.get("golden_readiness")
                    or manifest.get("golden_readiness")
                    or ""
                ),
                "status": "in_goldens",
                "scenario_type": str(
                    tags.get("scenario_type") or manifest.get("scenario_type") or ""
                ),
                "portability_action": str(manifest.get("action") or ""),
                "date_added": date_added,
                "date_promoted": date_added,
                "turns": str(len(turns)),
                "source_file": str(manifest.get("source_file") or ""),
                "expected_outcome": str(golden.get("expected_outcome") or ""),
                "notes": "",
            }
        )
        _apply_judge_fields(row, judge)
        _apply_candidate_score_fields(row, candidate_score_by_id.get(conv_id))
        if not row["canonical_file"] and row["source_file"]:
            # Manifest stores *.md; prefer *.conversation.json when available.
            stem = Path(row["source_file"]).stem
            row["canonical_file"] = f"{stem}.conversation.json"
        _enrich_from_conversation(row)
        if not row["notes"]:
            row["notes"] = "Already present in prod-conversation.goldens.jsonl"
        rows.append(row)

    # 2) New ready candidates from recent result files. Agent quality is
    # intentionally independent: bad/unclear rows can be useful regression cases.
    for path in candidate_result_paths:
        dataset = path.parent.name
        for judge in _load_jsonl(path):
            quality = str(judge.get("quality") or judge.get("final_category") or "")
            if quality == "needs_improvement":
                quality = "good"
            readiness = str(judge.get("golden_readiness") or "")
            if readiness in {"unsuitable", "not_applicable"}:
                readiness = "needs_rewrite"
            if readiness != "ready":
                continue
            conv_id = str(judge.get("conversation_id") or "")
            if not conv_id or conv_id in seen:
                continue
            seen.add(conv_id)

            module = str(judge.get("module") or "none")
            prompt = str(judge.get("original_prompt") or "")
            row = _blank_row()
            row.update(
                {
                    "golden_id": _provisional_golden_id(module, conv_id, prompt),
                    "conversation_id": conv_id,
                    "original_prompt": prompt,
                    "module": module,
                    "environment": str(judge.get("environment") or ""),
                    "dataset_source": dataset,
                    "judge_category": quality,
                    "golden_readiness": readiness,
                    "status": "candidate",
                    "scenario_type": "",
                    "portability_action": "pending_review",
                    "date_added": date_added,
                    "date_promoted": "",
                    "source_file": "",
                    "expected_outcome": "",
                    "notes": "Pending promotion into prod-conversation.goldens.jsonl",
                }
            )
            _apply_judge_fields(row, judge)
            _apply_candidate_score_fields(row, candidate_score_by_id.get(conv_id))
            _enrich_from_conversation(row)
            if not row["scenario_type"]:
                row["scenario_type"] = _infer_scenario_type(row)
            if not row["source_file"] and row["canonical_file"]:
                row["source_file"] = row["canonical_file"].replace(
                    ".conversation.json", ".md"
                )
            if not row["expected_outcome"]:
                row["expected_outcome"] = (
                    "TBD — write expected_outcome when promoting this candidate"
                )
            rows.append(row)

    return rows


def write_csv(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in COLUMNS})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goldens", type=Path, default=DEFAULT_GOLDENS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--results",
        type=Path,
        action="append",
        default=[],
        help="Result JSONL files whose `good` rows become candidates",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--candidate-scores",
        type=Path,
        action="append",
        default=[],
        help="Round 4 review-with-candidate-scores JSONL (auto-discovered if omitted)",
    )
    parser.add_argument("--date-added", default=_today())
    args = parser.parse_args()

    # Default candidate sources: latest 200-sample runs if present.
    candidate_paths = list(args.results)
    if not candidate_paths:
        for name in ("module-coverage-200", "random-200"):
            path = RESULTS_ROOT / name / "results.jsonl"
            if path.is_file():
                candidate_paths.append(path)

    all_result_paths = _discover_result_paths(candidate_paths)
    candidate_score_paths = _discover_candidate_score_paths(args.candidate_scores)
    rows = build_rows(
        goldens_path=args.goldens,
        manifest_path=args.manifest,
        candidate_result_paths=candidate_paths,
        all_result_paths=all_result_paths,
        candidate_score_paths=candidate_score_paths,
        date_added=args.date_added,
    )
    write_csv(rows, args.output)

    status_counts: dict[str, int] = {}
    empty_counts = {column: 0 for column in COLUMNS}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        for column in COLUMNS:
            if not str(row.get(column) or "").strip():
                empty_counts[column] += 1

    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"status={status_counts}")
    nonempty = {k: len(rows) - v for k, v in empty_counts.items() if v}
    print("still_empty=" + json.dumps({k: empty_counts[k] for k in nonempty}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
