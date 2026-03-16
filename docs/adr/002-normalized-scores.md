# ADR-002: Normalize all scores to [0.0, 1.0]

## Status

Accepted

## Context

Metrics produce scores. Different metrics have natural ranges: exact match is binary (0 or 1), latency could be milliseconds (0–10000), calibration error is 0–1. We need a consistent contract.

## Decision

All `Score.value` fields are normalized to `[0.0, 1.0]` where 1.0 = best and 0.0 = worst.

## Rationale

1. **Composable thresholds** — `threshold=0.8` means the same thing regardless of metric. Users don't need to know that latency is measured in milliseconds to set a meaningful threshold.

2. **Comparable across metrics** — Scores from different metrics can be placed side-by-side in tables, dashboards, and regression comparisons without unit conversion.

3. **Simple pass/fail** — `success = value >= threshold` is universal. No per-metric comparison logic.

4. **Prior art** — DeepEval, RAGAS, and promptfoo all normalize to 0–1. This is the de facto standard for eval frameworks.

## Trade-offs

- Metrics must implement their own normalization. For latency: `value = max(0, 1 - latency_ms / max_ms)`. This adds a few lines per metric but keeps the consumer interface clean.
- Raw values (actual latency, actual token count) go in `Score.metadata` for debugging.
- Information is lost: a score of 0.8 for latency and 0.8 for exact match mean very different things. The `name` and `metadata` fields restore context.

## Consequences

- All metrics follow the same normalization contract
- `evaluate()` and `assert_test()` don't need metric-specific logic
- Dashboard and comparison tooling can treat all scores uniformly
