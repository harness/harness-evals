# ADR-003: Safety metrics are never averaged into overall scores

## Status

Accepted

## Context

When displaying evaluation results, it's tempting to compute an "overall score" by averaging all metric scores. Safety metrics (PII detection, toxicity, hallucination, prompt injection) must not participate in this.

## Decision

Safety metrics are **reported separately** and **never averaged** into aggregate scores. They are hard constraints, not quality signals.

## Rationale

From Rabanser et al. (2026), *"Towards a Science of AI Agent Reliability"*:

> Safety requires consequence-aware assessment. Drawing from aviation's "one catastrophic error per billion flight hours" standard — not all failures are equal.

A response that scores 0.9 on faithfulness, 0.85 on coherence, and 0.0 on PII detection (leaked a credit card number) should not produce an "overall score" of 0.58. The PII failure is a hard constraint — the response is unacceptable regardless of other scores.

## Implementation

1. Safety metrics live in `metrics/safety/` with a distinct category
2. `evaluate()` returns all scores in a flat list — it doesn't distinguish safety from quality
3. The separation happens at the **consumer** level: dashboards, sinks, and comparison tools should display safety scores separately
4. Phase 3 will add a `category` field to `BaseMetric` (or use the module path) to enable filtering

## Trade-offs

- We don't enforce the separation at the `Score` level. A naive consumer could still average everything together. Documentation and conventions must be clear.
- We don't add a `is_safety` flag to `Score` because that would complicate the core types. The metric's module path (`metrics.safety.*`) provides the categorization.

## Consequences

- Safety failures are always visible, never diluted
- Consumers must handle safety metrics separately in aggregation logic
- This aligns with enterprise compliance requirements (SOC 2, HIPAA)
