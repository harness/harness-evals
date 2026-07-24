#!/usr/bin/env python3
"""Online evals: fetch Harness traces and score them, emitting CI reports.

Two modes:
  - Explicit trace IDs:  --trace-ids id1,id2,...
  - Lookback window:     --lookback-hours 4   (fetch all root traces in the last N hours)

Usage:
    python scripts/online_eval.py \\
        --trace-ids <id1>,<id2> \\
        --org-id <org> \\
        --project-id <project>

    python scripts/online_eval.py \\
        --lookback-hours 4 \\
        --org-id <org> \\
        --project-id <project>

Required env vars:
    HARNESS_API_KEY, HARNESS_ACCOUNT_ID
    OPENAI_API_KEY  (for LLM-judged metrics such as task_completion)

Optional env vars:
    HARNESS_BASE_URL  (default: https://app.harness.io)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_trace_ids(raw: str) -> list[str]:
    ids = [part.strip() for part in raw.replace("\n", ",").split(",")]
    return [tid for tid in ids if tid]


def _is_usable(case: object) -> bool:
    from harness_evals.core.eval_case import EvalCase

    assert isinstance(case, EvalCase)
    meta = case.metadata or {}
    if meta.get("error"):
        return False
    return bool(str(case.input or "").strip() or str(case.output or "").strip())


_METRIC_NAMES = ("task_completion", "geval", "coherence", "latency", "token_cost")


def _build_metrics(args: argparse.Namespace) -> list[Any]:
    from harness_evals.llm.openai import OpenAILLM

    requested = [m.strip() for m in args.metrics.split(",") if m.strip()]
    metrics = []
    llm = None

    def _llm() -> OpenAILLM:
        nonlocal llm
        if llm is None:
            llm = OpenAILLM(model=args.model)
        return llm

    for name in requested:
        if name == "task_completion":
            from harness_evals.metrics.agent.task_completion import TaskCompletionMetric

            metrics.append(TaskCompletionMetric(llm=_llm(), threshold=args.threshold))
        elif name == "geval":
            from harness_evals.metrics.llm_judge.geval import GEval

            metrics.append(GEval(llm=_llm(), criteria=args.geval_criteria, threshold=args.threshold))
        elif name == "coherence":
            from harness_evals.metrics.conversation.coherence import CoherenceMetric

            metrics.append(CoherenceMetric(llm=_llm(), threshold=args.threshold))
        elif name == "latency":
            from harness_evals.metrics.operational.latency import LatencyMetric

            metrics.append(LatencyMetric(threshold_ms=args.latency_threshold_ms))
        elif name == "token_cost":
            from harness_evals.metrics.operational.token_cost import TokenCostMetric

            metrics.append(TokenCostMetric())
        else:
            print(f"WARNING: unknown metric '{name}', skipping", file=sys.stderr)

    if not metrics:
        print(f"ERROR: no valid metrics in '{args.metrics}'. Choose from: {', '.join(_METRIC_NAMES)}", file=sys.stderr)
        raise SystemExit(2)

    return metrics


def _build_source(args: argparse.Namespace) -> Any:
    from harness_evals.importers.harness_otel import HarnessOTELEvalCaseSource

    return HarnessOTELEvalCaseSource(
        org_id=args.org_id,
        project_id=args.project_id,
    )


def _score_to_dict(score: Any) -> dict[str, Any]:
    return {
        "value": score.value,
        "passed": score.passed,
        "threshold": score.threshold,
        "reason": score.reason,
    }


def _write_fetch_failure_junit(junit_path: Path, skipped: list[dict[str, str]]) -> None:
    from harness_evals.core.eval_case import EvalCase
    from harness_evals.core.score import Score
    from harness_evals.sinks.junit_sink import JUnitSink

    junit = JUnitSink(path=str(junit_path), suite_name="online-evals")
    for item in skipped:
        trace_id = item["trace_id"]
        reason = item["reason"]
        junit.write(
            [
                Score(
                    name="trace_fetch",
                    value=0.0,
                    threshold=1.0,
                    reason=f"Trace {trace_id} could not be converted to a usable EvalCase: {reason}",
                    metadata={"dimension": "correctness"},
                )
            ],
            EvalCase(input=f"trace:{trace_id}", output="", metadata={"trace_id": trace_id}),
        )
    junit.finalize()


async def _resolve_trace_ids(args: argparse.Namespace, source: Any) -> list[str]:
    """Return trace IDs either from --trace-ids or by listing recent traces."""
    if args.trace_ids:
        return _parse_trace_ids(args.trace_ids)

    # Lookback mode: list root traces in the last N hours
    lookback_hours = int(args.lookback_hours)
    print(f"Listing traces from the last {lookback_hours} hour(s)...")
    rows = await source.list_traces(
        lookback_hours=lookback_hours,
        limit=args.lookback_limit,
        extra_filter=args.lookback_filter or None,
    )
    if not rows:
        print("No traces found in the lookback window.", file=sys.stderr)
        return []
    trace_ids = [r["trace_id"] for r in rows if r.get("trace_id")]
    print(f"Found {len(trace_ids)} trace(s) in the last {lookback_hours}h.")
    return trace_ids


async def _run(
    args: argparse.Namespace,
    *,
    source_factory: Callable[[argparse.Namespace], Any] = _build_source,
    metrics_factory: Callable[[argparse.Namespace], list[Any]] = _build_metrics,
) -> int:
    from harness_evals import a_evaluate, summarize
    from harness_evals.sinks.junit_sink import JUnitSink

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    junit_path = out_dir / "junit.xml"
    scores_path = out_dir / "scores.json"

    source = source_factory(args)
    trace_ids = await _resolve_trace_ids(args, source)

    if not trace_ids:
        print("ERROR: no trace IDs provided or discovered", file=sys.stderr)
        return 2

    print(f"Fetching {len(trace_ids)} trace(s)...")
    cases = await source.fetch_traces(trace_ids)

    usable = []
    skipped: list[dict[str, str]] = []
    for tid, case in zip(trace_ids, cases, strict=True):
        if _is_usable(case):
            usable.append(case)
        else:
            reason = (case.metadata or {}).get("error") or "empty input/output"
            skipped.append({"trace_id": tid, "reason": str(reason)})
            print(f"  skip {tid}: {reason}")

    if args.dry_run:
        print(f"Dry run — {len(usable)} usable trace(s), {len(skipped)} skipped:")
        for case in usable:
            tid = (case.metadata or {}).get("trace_id", "")
            print(f"  {tid}")
        for item in skipped:
            print(f"  {item['trace_id']} (skipped: {item['reason']})")
        return 0

    if not usable:
        print("ERROR: no usable eval cases after fetch", file=sys.stderr)
        _write_fetch_failure_junit(junit_path, skipped)
        scores_path.write_text(
            json.dumps(
                {
                    "run_at": datetime.now(timezone.utc).isoformat(),
                    "trace_ids": trace_ids,
                    "org_id": args.org_id,
                    "project_id": args.project_id,
                    "trace_count": 0,
                    "skipped": skipped,
                    "metrics": {},
                    "traces": [],
                },
                indent=2,
            )
            + "\n"
        )
        return 1

    metrics = metrics_factory(args)
    metric_names = args.metrics
    junit = JUnitSink(path=str(junit_path), suite_name="online-evals")

    print(f"Scoring {len(usable)} case(s) with [{metric_names}] (judge: {args.model})...")
    all_scores = []
    for case in usable:
        scores = await a_evaluate(case, metrics=metrics)
        junit.write(scores, case)
        all_scores.append(scores)
    junit.finalize()
    summary = summarize(all_scores)

    traces_out = []
    for case, scores in zip(usable, all_scores, strict=True):
        tid = (case.metadata or {}).get("trace_id", "")
        score_map = {s.name: _score_to_dict(s) for s in scores if s is not None}
        traces_out.append(
            {
                "trace_id": tid,
                "input_preview": str(case.input)[:120],
                "scores": score_map,
            }
        )

    metrics_out = {
        name: {
            "mean": ms.mean,
            "pass_rate": ms.pass_rate,
            "count": ms.count,
            "passed_count": ms.passed_count,
            "failed_count": ms.failed_count,
        }
        for name, ms in summary.by_metric.items()
    }

    payload = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "org_id": args.org_id,
        "project_id": args.project_id,
        "metrics_requested": args.metrics,
        "model": args.model,
        "threshold": args.threshold,
        "trace_ids": trace_ids,
        "trace_count": len(usable),
        "skipped": skipped,
        "metrics": metrics_out,
        "traces": traces_out,
    }
    scores_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Wrote {junit_path}")
    print(f"Wrote {scores_path}")
    for name, ms in summary.by_metric.items():
        print(f"  {name}: mean={ms.mean:.3f} pass_rate={ms.pass_rate:.3f} n={ms.count}")

    if args.fail_below > 0:
        overall_mean = (
            sum(ms.mean for ms in summary.by_metric.values()) / len(summary.by_metric) if summary.by_metric else 0.0
        )
        if overall_mean < args.fail_below:
            print(
                f"FAIL: overall mean {overall_mean:.3f} < --fail-below {args.fail_below}",
                file=sys.stderr,
            )
            return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    # --- Trace source (mutually exclusive modes) ---
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--trace-ids",
        default="",
        help="Comma-separated trace IDs to evaluate",
    )
    source_group.add_argument(
        "--lookback-hours",
        type=int,
        default=None,
        metavar="N",
        help="Fetch all root traces from the last N hours (alternative to --trace-ids)",
    )
    parser.add_argument(
        "--lookback-limit",
        type=int,
        default=100,
        help="Max traces to fetch when using --lookback-hours (default: 100)",
    )
    parser.add_argument(
        "--lookback-filter",
        default="",
        help="Extra HQL filter appended when using --lookback-hours, e.g. \"model = 'gpt-4o'\"",
    )

    # --- Harness org/project ---
    parser.add_argument("--org-id", required=True, help="Harness org identifier")
    parser.add_argument("--project-id", required=True, help="Harness project identifier")

    # --- Scoring ---
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Pass threshold for LLM-judged metrics (default: 0.7)",
    )
    parser.add_argument(
        "--fail-below",
        type=float,
        default=0.0,
        help="Exit 1 if mean score is below this (0 disables)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model for LLM-judged metrics (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--metrics",
        default="task_completion",
        help=f"Comma-separated metrics to run (default: task_completion). Choices: {', '.join(_METRIC_NAMES)}",
    )
    parser.add_argument(
        "--geval-criteria",
        default="Evaluate the answer quality.",
        help="Criteria used when --metrics includes geval",
    )
    parser.add_argument(
        "--latency-threshold-ms",
        type=float,
        default=30_000.0,
        help="Latency threshold in milliseconds when --metrics includes latency",
    )

    # --- Output ---
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for junit.xml and scores.json (default: results)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and print trace IDs that would be evaluated, then exit without scoring",
    )

    args = parser.parse_args()

    if not args.trace_ids and args.lookback_hours is None:
        parser.error("Provide either --trace-ids or --lookback-hours")

    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
