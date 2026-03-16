# ADR-004: evaluate() catches exceptions, assert_test() raises

## Status

Accepted

## Context

When running multiple metrics on a test case, one metric might raise an exception (invalid input, missing metadata key, LLM timeout). Should this crash the entire evaluation or be captured?

## Decision

- `evaluate()` catches all exceptions from `metric.measure()` and converts them to a failing `Score` with the exception message in `reason`.
- `assert_test()` calls `evaluate()` and then raises `AssertionError` if any score has `success=False`.

## Rationale

1. **Resilience over strictness for exploration** — When evaluating a dataset of 100 test cases with 10 metrics, one malformed test case shouldn't prevent the other 999 evaluations from completing. `evaluate()` is the exploration tool.

2. **Strictness for CI** — `assert_test()` is the CI tool. Any failure (including metric exceptions) surfaces as a test failure. No silent swallowing.

3. **Full picture** — By catching exceptions, `evaluate()` always returns exactly `len(metrics)` scores. The consumer never has to handle partial results or match scores to metrics.

4. **Exception details preserved** — The caught exception message goes into `Score.reason`, so debugging information isn't lost.

## Trade-offs

- A typo in metric code (e.g., `test_case.metdata` instead of `metadata`) produces a failing score instead of an immediate crash. This could delay bug discovery during development. Mitigation: tests catch these early.
- `evaluate()` silences all exceptions. A metric with a genuine bug will produce silent failing scores. Mitigation: `Score.reason` contains the exception message, and `assert_test()` surfaces it.

## Consequences

- `evaluate()` is always safe to call — never raises
- `assert_test()` is the strict variant for testing/CI
- All metrics produce exactly one score per evaluation, guaranteed
