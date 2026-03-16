# Review: harness-evals

Final review of the repo after the Phase 1 core redesign. Covers abstractions, code quality, phasing, and strategic positioning.

---

## Current State

- **61 tests, all passing, 0.09s, lint clean, zero warnings**
- **12 metrics** across 4 categories (deterministic, structural, operational, reliability)
- **2 dependencies** (`deepdiff`, `jsonschema`) — no LLM key needed
- **Clean data model**: `Golden` (authored) -> `EvalCase` (evaluated) -> `Score` (result)
- PEP 561 compliant (`py.typed`), CI on Python 3.10/3.11/3.12

---

## Abstractions Assessment

### Golden -> EvalCase -> Score: correct decomposition

Most eval frameworks (DeepEval, RAGAS) use a single type that mixes authored data with runtime data. This creates an awkward gap — you can't represent "I have 100 test cases but haven't run the agent yet." You end up with `actual_output=None` placeholders.

The Golden/EvalCase split maps to how evals actually work:

1. **Author time** — `Golden` lives in a YAML/JSONL file, checked into the repo
2. **Run time** — Agent produces output, `EvalCase.from_golden()` combines them
3. **Score time** — Metrics receive a complete `EvalCase`, return a `Score`

This enables `evaluate_dataset(goldens, agent_fn, metrics)` — the most natural API for batch evaluation. No other framework in the space has this flow this clean.

### BaseMetric with sync measure() + async a_measure(): correct for Phase 1

The dual API avoids forcing async on deterministic metrics while providing the extension point for Phase 2 LLM metrics. The abstraction boundary is at the metric level, not the runner level. `evaluate()` stays sync; individual metrics opt into async when they need it.

### Typed operational fields: correct tradeoff

`eval_case.latency_ms` is better than `(metadata or {}).get("latency_ms")`. Type-safe, IDE-discoverable, no string key typos. The `metadata` dict remains for extensibility. `ResourceConsistencyMetric._get_resource_value()` bridges both — typed field first, metadata fallback for custom keys like `gpu_memory`.

### Score.passed auto-computed: eliminates a bug class

Removing `success` from the constructor and computing `passed = value >= threshold` in `__post_init__` means no metric can accidentally pass `success=True` when `value < threshold`. Every metric's Score construction is now simpler.

### BaseSink: minimal and correct

One method, no lifecycle. Phase 3 adds `open()/close()` when JUnit and OTLP sinks need it. Not before.

---

## Industry Comparison (final state)

| Aspect | DeepEval | RAGAS | promptfoo | harness-evals | Assessment |
|--------|----------|-------|-----------|---------------|------------|
| **Data model** | `LLMTestCase` (single type) | `SingleTurnSample` (single type) | YAML `vars` + `assert` | `Golden` + `EvalCase` (split) | **Better** — clean authored/runtime separation |
| **Metric interface** | `measure()` + `a_measure()` | `_single_turn_ascore()` | Assertion type string | `measure()` + `a_measure()` | Matches DeepEval's proven pattern |
| **Non-assertion eval** | `evaluate()` | `evaluate()` | Default mode | `evaluate()` | Table stakes, covered |
| **Batch eval** | `evaluate(test_cases=[...])` | `evaluate(dataset)` | Matrix (prompts x providers x tests) | `evaluate_cases()` + `evaluate_dataset(goldens, agent_fn)` | **Better** — `evaluate_dataset` with agent_fn is unique |
| **Multi-run** | Not native | Not native | `--repeat` flag | Native `runs` field + `ReliabilityMetric` | **Differentiator** |
| **Output sinks** | Console + hardcoded cloud API | `.to_pandas()` | CLI `-o` (JSON/XML/CSV/HTML) | `BaseSink` ABC, pluggable | Cleaner extension point than all three |
| **Metric organization** | One directory per metric (heavy) | Flat in `metrics/` | Assertion types (not extensible) | One file per metric in category subdirs | Best of both |
| **LLM-free metrics** | ~2 | 0 | ~8 deterministic assertions | 12 in Phase 1 | **Differentiator** |
| **CI/CD native** | Not really | Not really | Yes (`--ci`, JUnit, exit codes) | Yes (pytest + JUnit + baseline) | Matches promptfoo, better than DeepEval/RAGAS |
| **Reliability metrics** | None | None | None | 12 metrics across 4 dimensions | **Unique in the space** |
| **Typed operational fields** | No (metadata dict) | No (metadata dict) | No (config-driven) | Yes (`latency_ms`, `token_count`, etc.) | **Better** DX |
| **Score auto-compute** | Manual `success` | No pass/fail concept | Per-assertion pass/fail | Auto-computed `passed` | Eliminates bugs |

---

## Phasing Critique

### Current phasing (from PLAN.md)

| Phase | Content | Duration |
|-------|---------|----------|
| 1 | Core + deterministic + structural + operational + reliability foundation | 2 weeks |
| 2 | Datasets + LLM abstraction + GEval + RAG + predictability | 2 weeks |
| 3 | Safety + agent + robustness metrics + JUnit + baseline | 2 weeks |
| 4 | Conversation + MCP + trajectory + fault robustness | 2-3 weeks |
| 5 | Synthesizer + perturbation generators | 3-4 weeks |
| 6 | Harness AI Evals integration | 2-3 weeks |

### Problem: perturbation generators are in Phase 5 but robustness metrics are in Phase 3

Phase 3 ships `PromptRobustnessMetric` and `EnvironmentRobustnessMetric`. These metrics need perturbation sets — alternative inputs for the same expected output. But the tools to generate those perturbations (`BasePerturbation`, `JsonFieldReorder`, `PromptRephrase`, `SchemaVariation`) are deferred to Phase 5.

This means Phase 3 ships metrics that require users to hand-craft perturbation sets. That's like shipping `JsonDiffMetric` without `deepdiff` — the metric exists but the input is unreasonably hard to produce.

**Dependency chain:**

```
Phase 2: BaseLLM (OpenAI, Anthropic)
    |
Phase 3: Robustness metrics (need perturbations)
    |         \
    |    Phase 5: Perturbation generators  <-- this is too late
    |
Phase 3 should include: deterministic perturbation generators
```

### Recommended fix: split perturbations from synthesizer

**Move to Phase 3** (alongside robustness metrics they serve):
- `BasePerturbation` ABC
- `JsonFieldReorder` — deterministic, zero deps
- `SchemaVariation` — deterministic, zero deps
- `TypoInjection` — deterministic, zero deps
- `PromptRephrase` — needs `BaseLLM` from Phase 2, available by Phase 3

**Keep in Phase 5** (standalone tool, not a metric prerequisite):
- `Synthesizer` — generates entire datasets from documents, genuinely different scope

This way robustness metrics ship with the tools to actually use them. Phase 5 becomes "Synthesizer" only, which simplifies it.

### Revised phasing

| Phase | Content | Change |
|-------|---------|--------|
| 1 | Core + deterministic + structural + operational + reliability | No change |
| 2 | Datasets + LLM abstraction + GEval + RAG + predictability | No change |
| 3 | Safety + agent + robustness + **perturbation generators** + JUnit + baseline | **Perturbation generators moved here** |
| 4 | Conversation + MCP + trajectory + fault robustness | No change |
| 5 | Synthesizer | **Simplified — perturbations removed** |
| 6 | Harness AI Evals integration | No change |

---

## Code Quality

### What's done right

- **Every metric follows the same pattern** — imports, class, `__init__` with `super().__init__`, `measure()` returns `Score`. A contributor reads any metric and knows exactly what to do.
- **`evaluate()` catches exceptions gracefully** — a broken metric returns a failing Score, doesn't crash the run (ADR-004).
- **`from_dict()` backward compat** — `actual_output`->`output`, `expected_output`->`expected`, `token_usage`->`token_count`. Unknown keys silently ignored.
- **`from_golden()`** — clean factory that copies golden fields and passes `**kwargs` for runtime data.
- **`Score.created_at`** uses `datetime.now(timezone.utc)` not the deprecated `datetime.utcnow()`.
- **`to_dict()` omits None** on both Golden and EvalCase — clean serialization.
- **`ResourceConsistencyMetric._get_resource_value()`** — typed field first, metadata fallback. Preserves the configurable `resource_key` pattern for custom keys.
- **Test coverage** — 61 tests covering happy paths, edge cases, backward compat aliases, typed fields, resource key fallback, batch evaluation, async dataset evaluation.

### What to watch

- **`EvalCase` field growth** — currently 12 fields, clean. Phase 2 adds `tools_called`, `trajectory`; Phase 4 adds `mcp_trace`. Around ~15+ fields, consider sub-dataclasses (e.g., `AgentTrace`). Not a problem today.
- **PLAN.md is stale** — still references `TestCase`, `actual_output`, `expected_output`, `success`, `metadata["latency_ms"]`. The code has moved ahead of the plan. Update PLAN.md to match the implemented types or it will mislead contributors.

---

## What's deferred (correctly)

| Abstraction | Phase | Why not now |
|---|---|---|
| Registry (`register_metric`) | 2 | Dead code without YAML config |
| `BaseLLM` provider abstraction | 2 | No LLM metrics in Phase 1 |
| YAML config (`eval.yaml`, `run_eval()`) | 2 | No one uses config on day one |
| JSON schemas for validation | 2 | Maintenance overhead without payoff yet |
| Dataset loaders (`load_dataset()`) | 2 | Comes with config infrastructure |
| `BaselineStore` | 3 | Needs score persistence first |
| Sink lifecycle (`open()`/`close()`) | 3 | Only JUnit/OTLP sinks need it |
| `BasePerturbation` | 3 (was 5) | **Should ship with robustness metrics** |
| `Synthesizer` | 5 | Standalone tool, not a metric prerequisite |

---

## Verdict

The abstractions are right. They're minimal, they match how developers actually think about evals, and they have clean extension points for Phases 2-6 without carrying dead infrastructure in Phase 1. The Golden/EvalCase split is genuinely better than what DeepEval, RAGAS, or promptfoo have.

One phasing fix needed: move perturbation generators from Phase 5 to Phase 3 so robustness metrics ship with the tools to use them. Everything else is correctly scoped.
