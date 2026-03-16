# Architecture

## Overview

harness-evals is a scoring engine. It takes a `TestCase`, runs it through one or more `BaseMetric` instances, produces `Score` objects, and optionally writes them to `BaseSink` destinations.

```
                          ┌────────────────┐
                          │   TestCase     │
                          │  input         │
                          │  actual_output │
                          │  expected_output│
                          │  context       │
                          │  metadata      │
                          │  tags          │
                          │  runs          │
                          └───────┬────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼              ▼
             ┌──────────┐ ┌──────────┐  ┌──────────────┐
             │ Metric 1 │ │ Metric 2 │  │ Metric N     │
             │ measure()│ │ measure()│  │ measure()    │
             └────┬─────┘ └────┬─────┘  └──────┬───────┘
                  │            │               │
                  ▼            ▼               ▼
             ┌─────────┐ ┌─────────┐   ┌─────────┐
             │ Score 1  │ │ Score 2 │   │ Score N │
             └────┬─────┘ └────┬────┘   └────┬────┘
                  │            │              │
                  └────────────┼──────────────┘
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
              ┌──────────┐ ┌────────┐ ┌────────┐
              │StdoutSink│ │JsonSink│ │JUnitSink│
              └──────────┘ └────────┘ └────────┘
```

## Core Data Flow

### Single Evaluation

```python
scores = evaluate(test_case, metrics=[...], sinks=[...])
```

1. `evaluate()` iterates over metrics
2. Each metric calls `measure(test_case)` and returns a `Score`
3. If a metric raises, the exception is caught and a failing `Score` is returned
4. All scores are passed to each sink's `write()` method
5. The list of scores is returned to the caller

### Assertion Mode

```python
scores = assert_test(test_case, metrics=[...], sinks=[...])
```

Same as `evaluate()`, but after collecting scores, raises `AssertionError` if any score has `success=False`. This integrates directly with pytest.

### Dataset Evaluation (Phase 2+)

```python
all_scores = evaluate_dataset(dataset, metrics=[...], sinks=[...])
```

Iterates over every `TestCase` in the dataset, calls `evaluate()` on each, returns `list[list[Score]]`.

## Extension Points

### 1. Metrics (`BaseMetric`)

Every metric is a class with a `measure()` method. Two base classes:

- **`BaseMetric`** — single test case evaluation. Implement `measure(test_case) -> Score`.
- **`ReliabilityMetric`** — multi-run evaluation. Implement `measure_runs(test_case) -> Score`. The base class `measure()` dispatches to `measure_runs()` when `test_case.runs` is populated.

Metrics are stateless. Configuration goes in `__init__()`. No global state, no side effects.

### 2. Sinks (`BaseSink`)

Every sink implements `write(scores, test_case)`. Sinks are called after all metrics finish for a test case. Sinks should not raise exceptions — they should handle errors gracefully.

Current: `StdoutSink`, `JsonSink`. Planned: `JUnitSink`, `CsvSink` (Phase 3).

### 3. LLM Providers (`BaseLLM`, Phase 2+)

LLM-judged metrics accept a `BaseLLM` instance. Providers implement `generate(prompt)` and `generate_json(prompt, schema)`. No global LLM configuration — metrics are explicit about which provider they use.

### 4. Baseline Stores (`BaselineStore`, Phase 3+)

Pluggable storage for score baselines. `JsonBaselineStore` uses local files. Enterprise users can implement remote storage.

### 5. Perturbation Generators (`BasePerturbation`, Phase 5+)

Produce semantically equivalent input variants. Feed into robustness metrics.

## Design Decisions

See [docs/adr/](adr/) for Architecture Decision Records explaining key choices:

- ADR-001: Why `dataclass` over Pydantic for core types
- ADR-002: Why scores are normalized to [0.0, 1.0]
- ADR-003: Why safety metrics are never averaged
- ADR-004: Why `evaluate()` catches exceptions instead of propagating
- ADR-005: Why `list[TestCase]` instead of a Dataset class
- ADR-006: Why async-first for LLM providers

## Module Dependency Graph

```
harness_evals/
├── core/          ← depends on nothing (pure Python + dataclasses)
│   ├── test_case  ← no imports from harness_evals
│   ├── score      ← no imports from harness_evals
│   ├── metric     ← imports score, test_case
│   ├── sink       ← imports score, test_case
│   └── runner     ← imports metric, score, sink, test_case
│
├── metrics/       ← depends on core/
│   ├── deterministic/  ← imports core.metric, core.score, core.test_case
│   ├── structural/     ← imports core.* + deepdiff, jsonschema
│   ├── operational/    ← imports core.*
│   └── reliability/    ← imports core.* (ReliabilityMetric)
│
├── sinks/         ← depends on core/
│   ├── stdout     ← imports core.score, core.sink, core.test_case
│   └── json_sink  ← imports core.score, core.sink, core.test_case
│
├── llm/           ← [Phase 2] depends on nothing (ABC + optional providers)
├── datasets       ← [Phase 2] depends on core.test_case
├── baseline/      ← [Phase 3] depends on core.score
├── synthesizer/   ← [Phase 5] depends on llm/, core.test_case
└── perturbations/ ← [Phase 5] depends on llm/ (for LLM-based perturbation)
```

**Key rule**: Dependencies flow downward. `core/` never imports from `metrics/`, `sinks/`, `llm/`, etc. Metrics never import from other metrics.

## Score Contract

Every `Score` follows these invariants:

1. `value` is in `[0.0, 1.0]` — 0.0 is worst, 1.0 is best
2. `success = value >= threshold` — deterministic pass/fail
3. `name` matches the metric name that produced it
4. `reason` is populated on failure (why it failed)
5. `metadata` carries metric-specific details (latency_ms, token count, etc.)

## TestCase Conventions

| Field | Used By | Convention |
|-------|---------|-----------|
| `input` | All metrics | The prompt / task given to the agent |
| `actual_output` | All metrics | What the agent produced |
| `expected_output` | Deterministic, Structural | Ground truth (may be None for LLM-judged) |
| `context` | RAG metrics | Retrieved context chunks |
| `metadata` | Operational, Reliability | `latency_ms`, `token_usage`, `cost_usd`, `confidence`, `tools_called`, `mcp_trace`, `trajectory` |
| `tags` | Filtering | `env`, `model`, `version` — not used by metrics directly |
| `runs` | Reliability metrics | K repeated runs of the same task |
