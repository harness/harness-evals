# ADR-005: Use list[TestCase] instead of a Dataset class

## Status

Accepted

## Context

Phase 2 introduces dataset loading. We need a type for "a collection of test cases." Options: a custom `Dataset` class with methods, or a simple type alias `Dataset = list[TestCase]`.

## Decision

`Dataset = list[TestCase]`. The loader function returns a plain list. No custom class.

## Rationale

1. **Composable with Python** — Users can filter, slice, sample, shuffle, and iterate using standard Python:

```python
dataset = load_dataset("tests.jsonl")
subset = [tc for tc in dataset if tc.tags and tc.tags.get("env") == "prod"]
sample = random.sample(dataset, 10)
```

A custom `Dataset` class would need to reimplement or wrap all of these.

2. **No lock-in** — Users can construct datasets from any source: API responses, database queries, CSV files, programmatic generation. A `list[TestCase]` is the simplest possible contract.

3. **Loader is the convenience** — `load_dataset()` handles file parsing (JSONL, JSON array). Everything else is standard Python.

4. **Upgradeable** — If we later need dataset-level metadata (name, version, description), we can add a `DatasetInfo` dataclass alongside the list without breaking the core type.

## Trade-offs

- No dataset-level metadata. A `Dataset` class could carry `name`, `version`, `created_at`. For Phase 2, this isn't needed. File name serves as identity.
- No built-in validation. A `Dataset` class could validate on construction (no empty list, all test cases have expected_output, etc.). Mitigation: `evaluate_dataset()` handles edge cases.
- No lazy loading. Large datasets must fit in memory. For the expected scale (hundreds to low thousands of test cases), this is fine.

## Consequences

- `load_dataset()` returns `list[TestCase]`
- `evaluate_dataset()` accepts `list[TestCase]`
- No import needed beyond `TestCase` — the "Dataset" type is just a list
- Commercial product (`aiEvals`) can wrap with its own `Dataset` class for UI/versioning
