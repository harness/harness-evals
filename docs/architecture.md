# Architecture

## Overview

harness-evals is a scoring engine. It takes an `EvalCase`, runs it through one or more `BaseMetric` instances, produces `Score` objects, and optionally writes them to `BaseSink` destinations.

The data model separates authored data (`Golden`) from evaluated data (`EvalCase`):

```
Golden (authored)                    EvalCase (evaluated)
┌─────────────────┐   + agent    ┌───────────────────┐
│ input           │   output     │ input             │
│ expected        │ ──────────►  │ output            │
│ context         │              │ expected          │
│ metadata        │              │ context           │
│ tags            │              │ latency_ms        │
└─────────────────┘              │ token_count       │
                                 │ cost_usd          │
                                 │ retry_count       │
                                 │ confidence        │
                                 │ tags / metadata   │
                                 │ runs              │
                                 └────────┬──────────┘
                                          │
                           ┌──────────────┼──────────────┐
                           ▼              ▼              ▼
                    ┌──────────┐  ┌──────────┐  ┌──────────────┐
                    │ Metric 1 │  │ Metric 2 │  │ Metric N     │
                    │ measure()│  │ measure()│  │ measure()    │
                    └────┬─────┘  └────┬─────┘  └──────┬───────┘
                         │             │               │
                         ▼             ▼               ▼
                    ┌─────────┐  ┌─────────┐  ┌─────────┐
                    │ Score 1  │  │ Score 2 │  │ Score N │
                    └────┬─────┘  └────┬────┘  └────┬────┘
                         │             │             │
                         └─────────────┼─────────────┘
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
scores = evaluate(eval_case, metrics=[...], sinks=[...])
```

1. `evaluate()` iterates over metrics
2. Each metric calls `measure(eval_case)` and returns a `Score`
3. If a metric raises, the exception is caught and a failing `Score` is returned
4. All scores are passed to each sink's `write()` method
5. The list of scores is returned to the caller

### Assertion Mode

```python
scores = assert_test(eval_case, metrics=[...], sinks=[...])
```

Same as `evaluate()`, but after collecting scores, raises `AssertionError` if any score has `passed=False`. This integrates directly with pytest.

### Batch Evaluation

```python
all_scores = evaluate_cases(cases, metrics=[...], sinks=[...])
```

Iterates over every `EvalCase`, calls `evaluate()` on each, returns `list[list[Score]]`.

### Dataset Evaluation (with agent)

```python
all_scores = await evaluate_dataset(goldens, agent_fn, metrics=[...], sinks=[...])
```

Runs `agent_fn` on each `Golden` to produce an `EvalCase`, then calls `evaluate()` on each. Async because `agent_fn` is I/O-bound.

## Extension Points

### 1. Metrics (`BaseMetric`)

Every metric is a class with a `measure()` method. Two base classes:

- **`BaseMetric`** — single eval case evaluation. Implement `measure(eval_case) -> Score`.
- **`ReliabilityMetric`** — multi-run evaluation. Implement `measure_runs(eval_case) -> Score`. The base class `measure()` dispatches to `measure_runs()` when `eval_case.runs` is populated.

Both provide `a_measure()` / `a_measure_runs()` async variants that default to calling the sync version. Phase 2 LLM-judged metrics override the async variants for I/O-bound scoring.

Metrics are stateless. Configuration goes in `__init__()`. No global state, no side effects.

### 2. Sinks (`BaseSink`)

Every sink implements `write(scores, eval_case)`. Sinks are called after all metrics finish for an eval case. Sinks should not raise exceptions — they should handle errors gracefully.

Shipped: `StdoutSink`, `JsonSink`, `CsvSink`, `JUnitSink`, `OtlpSink`, `LangfuseSink`, `HtmlSink`.

### 3. LLM Providers (`BaseLLM`)

LLM-judged metrics accept a `BaseLLM` instance. Providers implement `generate(prompt)` and `generate_json(prompt, schema)`. No global LLM configuration — metrics are explicit about which provider they use.

### 4. Baseline Stores (`BaselineStore`)

Pluggable storage for score baselines. `JsonBaselineStore` uses local files. Enterprise users can implement remote storage.

### 5. Perturbation Generators (`BasePerturbation`)

Produce semantically equivalent input variants. Deterministic generators (JsonFieldReorder, SchemaVariation, TypoInjection) and LLM-based PromptRephrase are shipped.

## Five Dimensions

Every metric belongs to exactly one of five evaluation dimensions. Dimensions answer "where is my agent strong?" and power the radar chart visualization.

```
                    Correctness
                        ╱╲
                       ╱  ╲
            Performance ╱    ╲ Groundedness
                      ╲    ╱
                       ╲  ╱
              Trajectory ╲╱ Safety
```

| Dimension | Question | Remediation |
|-----------|----------|-------------|
| **Correctness** | Is it right? | Better data, prompts, model |
| **Groundedness** | Is it supported by evidence? | Better retrieval, citation enforcement |
| **Safety** | Did it violate policy? | Guardrails, output filtering |
| **Trajectory** | Did it take a good path? | Better planning, tool docs |
| **Performance** | Was it fast and cheap? | Caching, model downgrade |

Dimensions are set by the metric author at definition time — not user-configured. See [ADR-009](adr/009-five-dimensions.md).

## Design Decisions

See [docs/adr/](adr/) for Architecture Decision Records explaining key choices:

- ADR-001: Why `dataclass` over Pydantic for core types
- ADR-002: Why scores are normalized to [0.0, 1.0]
- ADR-003: Why safety metrics are never averaged
- ADR-004: Why `evaluate()` catches exceptions instead of propagating
- ADR-005: Why `list[EvalCase]` instead of a Dataset class
- ADR-006: Why sync `measure()` with async `a_measure()` for LLM providers
- ADR-007: Why Golden and EvalCase are separate types
- ADR-008: Why `measure_dataset` instead of `evaluate_batch`
- ADR-009: Why every metric belongs to exactly one of five dimensions

## Module Dependency Graph

```
harness_evals/
├── core/          ← depends on nothing (pure Python + dataclasses)
│   ├── golden     ← no imports from harness_evals
│   ├── eval_case  ← imports golden
│   ├── score      ← no imports from harness_evals
│   ├── metric     ← imports score, eval_case
│   ├── sink       ← imports score, eval_case
│   └── runner     ← imports metric, score, sink, eval_case, golden
│
├── metrics/       ← depends on core/
│   ├── deterministic/  ← imports core.metric, core.score, core.eval_case
│   ├── structural/     ← imports core.* + deepdiff, jsonschema
│   ├── operational/    ← imports core.*
│   └── reliability/    ← imports core.* (ReliabilityMetric)
│
├── sinks/         ← depends on core/
│   ├── stdout     ← imports core.score, core.sink, core.eval_case
│   └── json_sink  ← imports core.score, core.sink, core.eval_case
│
├── llm/           ← [Phase 2] depends on nothing (ABC + optional providers)
├── baseline/      ← depends on core.score
├── synthesizer/   ← depends on llm/, core.eval_case
└── perturbations/ ← deterministic + LLM-based
```

**Key rule**: Dependencies flow downward. `core/` never imports from `metrics/`, `sinks/`, `llm/`, etc. Metrics never import from other metrics.

## Score Contract

Every `Score` follows these invariants:

1. `value` is in `[0.0, 1.0]` — 0.0 is worst, 1.0 is best
2. `passed = value >= threshold` — auto-computed, deterministic pass/fail
3. `name` matches the metric name that produced it
4. `reason` is populated on failure (why it failed)
5. `metadata` carries metric-specific details (latency_ms, token count, etc.)
6. `created_at` records when the score was produced

## EvalCase Conventions

| Field | Used By | Convention |
|-------|---------|-----------|
| `input` | All metrics | The prompt / task given to the agent |
| `output` | All metrics | What the agent produced |
| `expected` | Deterministic, Structural | Ground truth (may be None for LLM-judged) |
| `context` | RAG metrics | Retrieved context chunks |
| `latency_ms` | LatencyMetric | Response time in milliseconds |
| `token_count` | TokenCostMetric | Total token usage |
| `cost_usd` | CostEfficiencyMetric | Cost per request |
| `retry_count` | RetryCountMetric | Number of retries |
| `confidence` | Future metrics | Model confidence score |
| `metadata` | Custom metrics | Extensible dict for custom keys (e.g., `gpu_memory`) |
| `tags` | Filtering | `env`, `model`, `version` — not used by metrics directly |
| `runs` | Reliability metrics | K repeated runs of the same task |
