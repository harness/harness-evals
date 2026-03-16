# ADR-001: Use dataclass over Pydantic for core types

## Status

Accepted

## Context

Core types (`TestCase`, `Score`) need a structured data container. Options: stdlib `dataclass`, Pydantic `BaseModel`, or `attrs`.

## Decision

Use stdlib `@dataclass` for `TestCase` and `Score`.

## Rationale

1. **Zero dependencies** — `dataclass` is stdlib. Pydantic adds ~5MB and a compiled dependency (pydantic-core in Rust). For a lightweight scoring library, every dependency matters.

2. **No validation needed at the boundary** — `TestCase` and `Score` are internal types created by the user or by metrics, not deserialized from untrusted input. Pydantic's validation is overkill here.

3. **Simple composition** — `TestCase.runs` is `list[TestCase]`, which works naturally with dataclasses. Pydantic's self-referencing models require `model_rebuild()` and `from __future__ import annotations`.

4. **Familiar** — Every Python developer knows `dataclass`. Pydantic has a learning curve (v1 vs v2, model_config, validators).

## Trade-offs

- We lose automatic JSON serialization (Pydantic's `.model_dump()`). We use `dataclasses.asdict()` instead, which handles nested dataclasses.
- We lose runtime type validation. Users can pass `input=42` and it won't raise until a metric tries to use it. This is acceptable for a library (not a web API).
- If `harness-evals` later needs a REST API (Phase 6 integration), we can add Pydantic models in the API layer without changing core types.

## Consequences

- Core types remain dependency-free
- Users don't need to learn Pydantic to use harness-evals
- API layer (if needed) will have its own Pydantic contracts that convert to/from dataclasses
