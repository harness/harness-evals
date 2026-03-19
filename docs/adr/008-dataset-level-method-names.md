# ADR-008: Dataset-level metrics use distinct method names, not a shared ABC

## Status

Accepted

## Context

Some metrics are meaningful only when evaluated across a dataset (multiple eval cases), not on a single case. Two families of dataset-level metrics exist with incompatible signatures:

**Predictability metrics** (Calibration, Discrimination, BrierScore):
```python
def measure_dataset(self, cases: list[EvalCase], outcomes: list[bool]) -> Score
```
These need EvalCases with `confidence` values and per-case pass/fail outcomes.

**Robustness metrics** (PromptRobustness, EnvironmentRobustness):
```python
def measure_robustness(self, nominal_passed: list[bool], perturbed_passed: list[list[bool]]) -> Score
```
These need pre-computed pass/fail arrays from nominal and perturbed runs — they don't need EvalCases at all.

The runner's `evaluate_batch_metrics()` dispatches via `hasattr(metric, "measure_dataset")`. If robustness metrics also used `measure_dataset`, the `hasattr` check would pass but the call would fail at runtime due to the signature mismatch.

## Decision

Use **distinct method names** for each dataset-level family:

- `measure_dataset(cases, outcomes)` — for predictability metrics that need EvalCases + outcomes
- `measure_robustness(nominal_passed, perturbed_passed)` — for robustness metrics that need accuracy ratios

Do **not** introduce a shared ABC or Protocol for dataset-level evaluation.

## Rationale

1. **Type safety** — Distinct names prevent the `hasattr` dispatch bug. The runner can target `measure_dataset` without accidentally calling robustness metrics with the wrong arguments.

2. **Genuinely different data requirements** — Predictability metrics need EvalCases (to read `confidence`). Robustness metrics need pre-computed boolean arrays from nominal and perturbed runs. A shared interface would either lose type information (`**kwargs`, `Any`) or force awkward wrappers (stuffing robustness data into EvalCase metadata for the dataset-level call).

3. **No polymorphic dispatch need** — Users don't mix predictability and robustness metrics in the same `evaluate_batch_metrics()` call. They serve different evaluation paradigms: "does the agent know when it will fail?" vs. "does the agent degrade under perturbation?"

4. **Extensible** — Phase 4 adds `FaultRobustnessMetric`, which will use the same `measure_robustness` signature. If a third family emerges, it gets its own method name rather than overloading `measure_dataset`.

## Alternatives Considered

- **Shared Protocol with Union types** — `measure_dataset(data: EvalCases | RobustnessData)` — loses type safety, requires runtime dispatch inside each metric.
- **Single `measure_dataset` name with `*args, **kwargs`** — no static type checking, easy to pass wrong arguments.
- **Force robustness metrics to accept `(cases, outcomes)`** — robustness metrics would need to extract `nominal_passed` and `perturbed_results` from EvalCase metadata, making the dataset-level API less ergonomic than just passing the data directly.

## Consequences

- `evaluate_batch_metrics()` safely dispatches only predictability metrics via `hasattr(metric, "measure_dataset")`
- Robustness dataset-level evaluation is called directly: `metric.measure_robustness(nominal, perturbed)`
- New dataset-level metric families should choose a descriptive method name rather than reusing `measure_dataset`
- Per-case `measure(eval_case)` remains the universal interface defined by `BaseMetric` — all metrics support it
