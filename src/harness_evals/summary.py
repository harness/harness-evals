"""Score aggregation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

from harness_evals.core.score import Score

# The dimension whose scores are hard constraints (ADR-003): reported
# separately and never averaged into the quality pass rate.
SAFETY_DIMENSION = "safety"
# Bucket for scores whose metric did not declare a dimension.
UNKNOWN_DIMENSION = "unknown"
# The five canonical evaluation dimensions in ADR-009 display order.
CANONICAL_DIMENSIONS = ("correctness", "groundedness", "safety", "trajectory", "performance")


def order_dimensions(dimensions: list[str]) -> list[str]:
    """Order dimensions for display: canonical five first (ADR-009 order),
    then any extras sorted alphabetically. Shared by every output surface
    (HTML radar, stdout) so a given run renders dimensions in one order.
    """
    present = set(dimensions)
    ordered = [d for d in CANONICAL_DIMENSIONS if d in present]
    extras = sorted(d for d in present if d not in CANONICAL_DIMENSIONS)
    return ordered + extras


@dataclass
class MetricSummary:
    """Aggregate statistics for a single metric across multiple eval cases."""

    name: str
    mean: float
    pass_rate: float
    count: int
    min_value: float
    max_value: float
    passed_count: int
    failed_count: int


@dataclass
class DimensionSummary:
    """Aggregate statistics for a single evaluation dimension (ADR-009).

    A dimension groups every score whose metric declared that dimension
    (via ``score.metadata["dimension"]``). ``is_safety`` marks the Safety
    dimension, whose scores are hard constraints per ADR-003.
    """

    dimension: str
    mean: float
    pass_rate: float
    metric_count: int
    is_safety: bool


@dataclass
class ScoreSummary:
    """Aggregate statistics across all metrics and eval cases.

    ``quality_pass_rate`` covers non-safety scores only; ``safety_pass_rate``
    and ``safety_violations`` keep safety as a separate hard constraint per
    ADR-003. There is deliberately no blended "overall" pass rate — averaging
    safety into quality would dilute a hard-constraint failure.
    """

    by_metric: dict[str, MetricSummary] = field(default_factory=dict)
    by_dimension: dict[str, DimensionSummary] = field(default_factory=dict)
    total_cases: int = 0
    quality_pass_rate: float = 0.0
    safety_pass_rate: float = 0.0
    safety_violations: int = 0


def dimension_of(score: Score) -> str:
    """The dimension a score belongs to, or ``UNKNOWN_DIMENSION`` if undeclared.

    Single source of truth for how a score is mapped to a dimension bucket,
    shared by ``summarize()`` and streaming sinks (e.g. ``OtlpSink``).
    """
    return (score.metadata or {}).get("dimension") or UNKNOWN_DIMENSION


def build_dimension_summary(dimension: str, values: list[float], passed_count: int) -> DimensionSummary:
    """Build a :class:`DimensionSummary` from pre-aggregated values + passed count.

    Keeps the per-dimension aggregation (mean, pass rate, safety flag) in one
    place. ``summarize()`` calls this after bucketing ``Score`` objects;
    streaming sinks that already hold running aggregates call it directly,
    without buffering every score in memory.
    """
    count = len(values)
    return DimensionSummary(
        dimension=dimension,
        mean=sum(values) / count if count else 0.0,
        pass_rate=passed_count / count if count else 0.0,
        metric_count=count,
        is_safety=dimension == SAFETY_DIMENSION,
    )


def summarize(all_scores: list[list[Score]]) -> ScoreSummary:
    """Aggregate a batch of per-case score lists into summary statistics.

    ``all_scores`` is the return value of ``evaluate_cases()`` — a list
    of score lists, one per eval case.  ``None`` entries (from skipped
    metrics) are excluded automatically.
    """
    buckets: dict[str, list[Score]] = {}
    dim_buckets: dict[str, list[Score]] = {}
    for case_scores in all_scores:
        for score in case_scores:
            if score is None:
                continue
            buckets.setdefault(score.name, []).append(score)
            dim_buckets.setdefault(dimension_of(score), []).append(score)

    by_metric: dict[str, MetricSummary] = {}
    for name, scores in buckets.items():
        values = [s.value for s in scores]
        passed = sum(1 for s in scores if s.passed)
        by_metric[name] = MetricSummary(
            name=name,
            mean=sum(values) / len(values),
            pass_rate=passed / len(values),
            count=len(values),
            min_value=min(values),
            max_value=max(values),
            passed_count=passed,
            failed_count=len(values) - passed,
        )

    by_dimension: dict[str, DimensionSummary] = {}
    for dimension, scores in dim_buckets.items():
        values = [s.value for s in scores]
        passed = sum(1 for s in scores if s.passed)
        by_dimension[dimension] = build_dimension_summary(dimension, values, passed)

    total_cases = len(all_scores)
    total_scores = sum(ms.count for ms in by_metric.values())
    total_passed = sum(ms.passed_count for ms in by_metric.values())

    # Safety scores are hard constraints (ADR-003): reported separately and
    # excluded from the quality pass rate so a safety failure is never diluted.
    safety_scores = dim_buckets.get(SAFETY_DIMENSION, [])
    safety_passed = sum(1 for s in safety_scores if s.passed)
    safety_pass_rate = safety_passed / len(safety_scores) if safety_scores else 0.0
    safety_violations = len(safety_scores) - safety_passed

    quality_total = total_scores - len(safety_scores)
    quality_passed = total_passed - safety_passed
    quality_pass_rate = quality_passed / quality_total if quality_total > 0 else 0.0

    return ScoreSummary(
        by_metric=by_metric,
        by_dimension=by_dimension,
        total_cases=total_cases,
        quality_pass_rate=quality_pass_rate,
        safety_pass_rate=safety_pass_rate,
        safety_violations=safety_violations,
    )


@dataclass
class ModelSpendSummary:
    """Aggregate judge LLM spend for one model across an eval run."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    metric_count: int


@dataclass
class JudgeSpendSummary:
    """Aggregate LLM judge spend across all scored metrics in a run."""

    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float | None
    cost_available: bool
    by_model: dict[str, ModelSpendSummary] = field(default_factory=dict)
    by_metric: dict[str, float] = field(default_factory=dict)


def summarize_judge_spend(all_scores: list[list[Score]]) -> JudgeSpendSummary | None:
    """Aggregate judge token/cost metadata attached to metric scores.

    Returns ``None`` when no score reported LLM usage (heuristic-only run).
    """
    total_input = 0
    total_output = 0
    total_cost = 0.0
    saw_cost = False
    by_model: dict[str, ModelSpendSummary] = {}
    by_metric: dict[str, float] = {}
    saw_tokens = False

    for case_scores in all_scores:
        for score in case_scores:
            if score is None or not score.metadata:
                continue
            meta = score.metadata
            input_tokens = meta.get("input_tokens")
            output_tokens = meta.get("output_tokens")
            if input_tokens is not None or output_tokens is not None:
                saw_tokens = True
                total_input += int(input_tokens or 0)
                total_output += int(output_tokens or 0)

            metric_cost = meta.get("cost_usd")
            has_llm_spend = bool(meta.get("llm_spend_by_model")) or bool(meta.get("llm_model"))
            metric_cost = metric_cost if has_llm_spend else None
            if metric_cost is not None:
                saw_cost = True
                total_cost += float(metric_cost)
                by_metric[score.name] = by_metric.get(score.name, 0.0) + float(metric_cost)

            spend_by_model = meta.get("llm_spend_by_model")
            if isinstance(spend_by_model, dict):
                for model, details in spend_by_model.items():
                    if not isinstance(details, dict):
                        continue
                    bucket = by_model.setdefault(
                        str(model),
                        ModelSpendSummary(
                            model=str(model),
                            input_tokens=0,
                            output_tokens=0,
                            cost_usd=0.0,
                            metric_count=0,
                        ),
                    )
                    bucket.input_tokens += int(details.get("input_tokens") or 0)
                    bucket.output_tokens += int(details.get("output_tokens") or 0)
                    bucket.cost_usd += float(details.get("cost_usd") or 0.0)
                    bucket.metric_count += 1
            elif meta.get("llm_model") and metric_cost is not None:
                model = str(meta["llm_model"])
                bucket = by_model.setdefault(
                    model,
                    ModelSpendSummary(
                        model=model,
                        input_tokens=0,
                        output_tokens=0,
                        cost_usd=0.0,
                        metric_count=0,
                    ),
                )
                bucket.input_tokens += int(input_tokens or 0)
                bucket.output_tokens += int(output_tokens or 0)
                bucket.cost_usd += float(metric_cost)
                bucket.metric_count += 1

    if not saw_tokens and not saw_cost:
        return None

    return JudgeSpendSummary(
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_cost_usd=total_cost if saw_cost else None,
        cost_available=saw_cost,
        by_model=by_model,
        by_metric=by_metric,
    )


def judge_spend_to_dict(spend: JudgeSpendSummary) -> dict[str, object]:
    """Serialize :class:`JudgeSpendSummary` for JSON output."""
    payload: dict[str, object] = {
        "total_input_tokens": spend.total_input_tokens,
        "total_output_tokens": spend.total_output_tokens,
        "cost_available": spend.cost_available,
    }
    if spend.total_cost_usd is not None:
        payload["total_cost_usd"] = round(spend.total_cost_usd, 8)
    if spend.by_model:
        payload["by_model"] = {
            model: {
                "input_tokens": ms.input_tokens,
                "output_tokens": ms.output_tokens,
                "cost_usd": round(ms.cost_usd, 8),
                "metric_count": ms.metric_count,
            }
            for model, ms in sorted(spend.by_model.items())
        }
    if spend.by_metric:
        payload["by_metric"] = {name: round(cost, 8) for name, cost in sorted(spend.by_metric.items())}
    return payload


def format_judge_spend(spend: JudgeSpendSummary) -> str:
    """Human-readable judge spend block for stdout summaries."""
    lines = ["  Judge LLM spend:"]
    if spend.total_cost_usd is not None:
        lines.append(f"    total: ${spend.total_cost_usd:.6f}")
    else:
        lines.append("    total cost: unavailable (pip install 'harness-evals[cost]' for dynamic pricing)")
    lines.append(f"    tokens: {spend.total_input_tokens:,} in / {spend.total_output_tokens:,} out")
    if spend.by_model:
        lines.append("    by model:")
        for model, ms in sorted(spend.by_model.items(), key=lambda item: item[1].cost_usd, reverse=True):
            cost_part = f"${ms.cost_usd:.6f}" if spend.cost_available else "cost n/a"
            lines.append(
                f"      {model}: {cost_part} "
                f"({ms.input_tokens:,} in / {ms.output_tokens:,} out, {ms.metric_count} metric(s))"
            )
    return "\n".join(lines)


def summary_to_dict(
    result: ScoreSummary,
    *,
    judge_spend: JudgeSpendSummary | None = None,
) -> dict[str, object]:
    """Serialize a :class:`ScoreSummary` for JSON output (e.g. JSONL footer record)."""
    metrics: dict[str, object] = {}
    for name, ms in result.by_metric.items():
        metrics[name] = {
            "mean": round(ms.mean, 4),
            "pass_rate": round(ms.pass_rate, 4),
            "passed_count": ms.passed_count,
            "count": ms.count,
        }

    dimensions: dict[str, object] = {}
    for dim in order_dimensions(list(result.by_dimension)):
        ds = result.by_dimension[dim]
        entry: dict[str, object] = {
            "mean": round(ds.mean, 2),
            "pass_rate": round(ds.pass_rate, 4),
            "metric_count": ds.metric_count,
            "is_safety": ds.is_safety,
        }
        if ds.is_safety and result.safety_violations:
            entry["violations"] = result.safety_violations
        dimensions[dim] = entry

    payload = {
        "record_type": "summary",
        "total_cases": result.total_cases,
        "quality_pass_rate": round(result.quality_pass_rate, 4),
        "safety_pass_rate": round(result.safety_pass_rate, 4),
        "safety_violations": result.safety_violations,
        "metrics": metrics,
        "dimensions": dimensions,
    }
    if judge_spend is not None:
        payload["judge_spend"] = judge_spend_to_dict(judge_spend)
    return payload
