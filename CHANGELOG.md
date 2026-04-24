# Changelog

All notable changes to harness-evals will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`RubricLevel` dataclass** (`harness_evals.metrics.llm_judge.RubricLevel`) — score-band primitive
  (`min_score`, `max_score`, `description`) for structured integer rubrics.
- **`GEvalMetric` enrichments** — optional `evaluation_steps: list[str]` (numbered chain-of-thought),
  optional `rubric: list[RubricLevel]` (integer scoring on the rubric's range, normalized to 0.0-1.0
  with `raw_score` / `min_score` / `max_score` captured in `Score.metadata`), explicit `name` and
  `dimension` overrides, and a universal grounding instruction that tells the judge to score only
  on the provided input/output. Subclasses may declare `criteria`, `evaluation_steps`, `rubric`
  as class attributes; constructor arguments override them.
- **`RubricJudgeMetric` enrichments** — optional `evaluation_steps: list[str]` (numbered
  chain-of-thought), plus the same grounding instruction.
- **Security Remediation metrics** — 7 LLM-as-Judge metrics for evaluating AI-generated
  vulnerability remediations: VulnerabilityCorrectness, SecurityCompleteness, CodeSafety,
  CodeQuality, ExplanationQuality, RootCauseAnalysis, Actionability (all in
  `harness_evals.metrics.security`). Each is a thin `GEvalMetric` subclass declaring its
  criteria, evaluation steps, and 0-10 score-band rubric as class attributes.
- `remediation_quality_index()` — weighted composite scoring (Remediation Quality Index) for
  security metrics.
- `HarnessAILLM` provider — routes LLM calls through the Harness AI Service gateway
  (`harness_evals.llm.harness_ai`).
- `[harness]` optional dependency group (httpx, PyJWT).

### Changed

- `GEvalMetric` / `RubricJudgeMetric` prompts now include an explicit grounding instruction
  ("Evaluate ONLY based on the input and output provided … Do not infer or assume …"). This is a
  prompt-only change; the public API remains backward-compatible.

### Notes

- The enriched `GEvalMetric` is the recommended base for any new structured LLM-as-Judge metric.
  The internal-only `SecurityRemediationMetric` base class that existed during development has
  been removed in favor of direct `GEvalMetric` subclassing.

## [0.5.0] - 2026-04-17

### Added

- `ToolArgumentMatchMetric` — deterministic comparison of tool-call arguments against authored expectations. Companion to `ToolCorrectnessMetric` (names) and the LLM-judged `ArgumentCorrectnessMetric`. Supports `pair=exact|subset`, `arg_match=exact|subset`, `ignore_keys`, and `wildcard_value`. Registered in the catalog as `"tool_argument_match"`.
- `Golden.expected_tool_calls: list[ToolCall] | None` and `EvalCase.expected_tool_calls: list[ToolCall] | None` — optional, defaults to `None`. Lets dataset authors carry expected tool-call arguments alongside `expected_tools`. `Golden.from_dict`, `EvalCase.from_dict`, and `EvalCase.from_golden` handle (de)serialization and propagation.
- ADR-010: Why `ToolArgumentMatchMetric` is a separate metric (not an enrichment of `ToolCorrectness`).

### Changed

- README and PLAN.md updated with the new metric, the canonical `ToolCorrectness` + `ToolArgumentMatch` pairing snippet, and the data-model note.

### Notes

- Fully backward-compatible: no existing field changes its type or default; existing JSONL datasets continue to load unchanged.

## [0.2.0] - 2026-03-16

### Added

- `Golden` dataclass — authored evaluation data (input, expected, context)
- `EvalCase` dataclass — replaces `TestCase`, adds typed operational fields (`latency_ms`, `token_count`, `cost_usd`, `retry_count`, `confidence`)
- `EvalCase.from_golden()` — factory to create EvalCase from Golden + agent output
- `EvalCase.from_dict()` / `Golden.from_dict()` — with backward-compat aliases (`actual_output` -> `output`, `expected_output` -> `expected`, `token_usage` -> `token_count`)
- `Score.passed` — auto-computed from `value >= threshold` in `__post_init__`
- `Score.created_at` — UTC timestamp set at creation
- `Score.to_dict()`, `Golden.to_dict()`, `EvalCase.to_dict()` — serialization methods
- `evaluate_cases()` — sync batch evaluation of pre-captured eval cases
- `evaluate_dataset()` — async evaluation: runs agent on goldens, then scores
- `BaseMetric.a_measure()` — async variant, defaults to calling sync `measure()`
- `ReliabilityMetric.a_measure_runs()` — async variant for multi-run metrics
- ADR-007: Why Golden and EvalCase are separate types

### Changed

- **BREAKING**: `TestCase` removed, replaced by `Golden` + `EvalCase`
- **BREAKING**: `actual_output` renamed to `output`
- **BREAKING**: `expected_output` renamed to `expected`
- **BREAKING**: `Score.success` renamed to `Score.passed` (auto-computed, not in constructor)
- Operational metrics read typed fields (`eval_case.latency_ms`) instead of `metadata` dict
- `ResourceConsistencyMetric` default `resource_key` changed from `"token_usage"` to `"token_count"`
- ADR-006 updated to reflect sync `measure()` + async `a_measure()` pattern

### Removed

- `TestCase` dataclass (use `Golden` + `EvalCase` instead)
- `Score.success` constructor parameter (use auto-computed `Score.passed`)

### Migration

Replace `TestCase` usage:

```python
# Before
tc = TestCase(input="q", actual_output="a", expected_output="e",
              metadata={"latency_ms": 100})
score = metric.measure(tc)
if score.success: ...

# After
ec = EvalCase(input="q", output="a", expected="e", latency_ms=100)
score = metric.measure(ec)
if score.passed: ...
```

Or use `from_dict()` with old field names (backward compatible):

```python
ec = EvalCase.from_dict({"input": "q", "actual_output": "a", "expected_output": "e"})
```

## [0.1.0] - 2026-03-16

### Added

- Core types: `TestCase`, `Score`, `BaseMetric`, `ReliabilityMetric`, `BaseSink`
- Runner functions: `evaluate()` (non-raising) and `assert_test()` (raises on failure)
- Deterministic metrics: `ExactMatchMetric`, `ContainsMetric`, `RegexMetric`, `NumericDiffMetric`
- Structural metrics: `JsonDiffMetric` (DeepDiff-backed), `SchemaValidationMetric` (jsonschema-backed)
- Operational metrics: `LatencyMetric`, `TokenCostMetric`, `CostEfficiencyMetric`, `RetryCountMetric`
- Reliability metrics: `OutcomeConsistencyMetric`, `ResourceConsistencyMetric`
- Output sinks: `StdoutSink`, `JsonSink`
- Project infrastructure: pyproject.toml, .gitignore, .pre-commit-config.yaml
- CI: GitHub Actions workflow for Python 3.10/3.11/3.12
- Documentation: README.md, AGENTS.md, PLAN.md (full 6-phase vision)
- 42 passing tests covering all metrics and core functions
- Example: `examples/basic_eval.py`

### Planned

- Phase 5: Synthesizer, perturbation generators
- Phase 6: Harness AI Evals integration
