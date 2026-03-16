# ADR-007: Why Golden and EvalCase are separate types

## Status

Accepted

## Context

The original `TestCase` dataclass mixed two concerns:

1. **Authored data** — what you put in your dataset files: input, expected output, context
2. **Runtime data** — what the agent produces and how it performed: actual output, latency, token count, cost

This created several problems:

- `TestCase` required `actual_output` at construction time, but for dataset-driven evaluation the agent hasn't run yet
- Operational metrics read from a generic `metadata` dict with string keys (`metadata["latency_ms"]`), losing type safety and IDE discoverability
- `evaluate_dataset(dataset, agent_fn, metrics)` was impossible to type cleanly — `dataset` needed to be a list of "incomplete" TestCases with no `actual_output`
- The name `TestCase` collided with `unittest.TestCase` in import autocompletion

## Decision

Split `TestCase` into two types:

- **`Golden`** — authored data: `input`, `expected`, `context`, `metadata`, `tags`
- **`EvalCase`** — what metrics receive: all Golden fields + `output` + typed operational fields (`latency_ms`, `token_count`, `cost_usd`, `retry_count`, `confidence`) + `runs`

`EvalCase.from_golden(golden, output=..., latency_ms=..., ...)` bridges the two.

## Rationale

1. **Clean data flow** — `Golden` (authored) + agent output -> `EvalCase` (evaluated) -> `Score` (result). Each type represents one stage.

2. **Type safety for operational metrics** — `eval_case.latency_ms` (typed `float | None`) vs `(metadata or {}).get("latency_ms")` (untyped, string key). IDE autocomplete, no typos, no `float()` coercion needed.

3. **`evaluate_dataset()` types cleanly** — `goldens: list[Golden]` is a proper type. The agent hasn't run, so there's no `output` field to leave empty.

4. **Backward compatibility via `from_dict()`** — `EvalCase.from_dict()` accepts old field names (`actual_output` -> `output`, `expected_output` -> `expected`, `token_usage` -> `token_count`). Existing JSONL datasets work without migration.

5. **Metadata stays as escape hatch** — `EvalCase.metadata` remains for custom keys that aren't promoted to typed fields (e.g., `gpu_memory`, `tools_called`). `ResourceConsistencyMetric` tries typed fields first, falls back to metadata.

## Trade-offs

- Two types instead of one. Users need to learn which to use when. Mitigation: clear naming (Golden = what you write, EvalCase = what you evaluate) and `from_golden()` factory.
- Migration required from `TestCase`. Mitigation: `from_dict()` backward compat + CHANGELOG migration guide.

## Consequences

- `TestCase` is removed
- All metrics accept `EvalCase`
- All sinks accept `EvalCase`
- `evaluate()`, `assert_test()`, `evaluate_cases()` accept `EvalCase`
- `evaluate_dataset()` accepts `list[Golden]` + `agent_fn`
- 5 operational fields are typed on `EvalCase` instead of in `metadata`
