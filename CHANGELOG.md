# Changelog

All notable changes to harness-evals will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Security Remediation metrics** — 7 LLM-as-Judge metrics for evaluating AI-generated vulnerability remediations:
  VulnerabilityCorrectness, SecurityCompleteness, CodeSafety, CodeQuality,
  ExplanationQuality, RootCauseAnalysis, Actionability (all in `harness_evals.metrics.security`)
- `remediation_quality_index()` — weighted composite scoring (Remediation Quality Index) for security metrics
- `HarnessAILLM` provider — routes LLM calls through the Harness AI Service gateway (`harness_evals.llm.harness_ai`)
- `[harness]` optional dependency group (PyJWT, requests)
- 78 new tests (62 security metric unit + structural + prompt + pipeline, 16 HarnessAILLM + JSON extraction)

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
