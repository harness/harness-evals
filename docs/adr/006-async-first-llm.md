# ADR-006: Sync measure() with async a_measure() for LLM providers

## Status

Accepted (refined from original "async-first" proposal)

## Context

LLM-judged metrics (GEval, Faithfulness, etc.) need to call external APIs. These calls are I/O-bound and high-latency (1–30 seconds). When evaluating a dataset of 100 test cases with 3 LLM-judged metrics, that's 300 API calls.

However, Phase 1 metrics are all deterministic/structural/operational — none do I/O. Making `measure()` async for these metrics would add friction: `asyncio.run()` wrappers, `pytest-asyncio` markers on every test, `async def` with zero `await` statements.

## Decision

`measure()` stays sync. `a_measure()` provides an async variant with a default implementation that calls `measure()`:

```python
class BaseMetric(ABC):
    @abstractmethod
    def measure(self, eval_case: EvalCase) -> Score: ...

    async def a_measure(self, eval_case: EvalCase) -> Score:
        """Async variant. Override for I/O-bound metrics."""
        return self.measure(eval_case)
```

Phase 1 metrics implement only `measure()`. Phase 2+ LLM metrics override `a_measure()`. The sync `measure()` on LLM metrics can call `asyncio.run(self.a_measure(eval_case))` as a convenience.

## Rationale

1. **No friction for Phase 1** — `evaluate()`, `assert_test()`, `evaluate_cases()` are all sync. Tests don't need `@pytest.mark.asyncio`. The basic example is `scores = evaluate(case, metrics=[...])` — no `asyncio.run()`.

2. **Async available when needed** — `evaluate_dataset()` is async (because `agent_fn` is async). Phase 2 LLM metrics use `a_measure()` for concurrent API calls via `asyncio.gather()`.

3. **Precedent** — DeepEval uses this exact pattern: sync `measure()` and async `a_measure()` coexist.

4. **API clients are async** — Both `openai>=1.0` and `anthropic>=0.30` provide native async clients. `a_measure()` gives direct access.

## Trade-offs

- Two methods instead of one. Contributors need to know: override `measure()` for sync metrics, `a_measure()` for async metrics. Mitigation: clear documentation in metrics-guide.md.
- `asyncio.run()` cannot be called from within an existing event loop (e.g., Jupyter). Mitigation: detect and use `nest_asyncio` or document the workaround.

## Consequences

- `BaseMetric.measure()` is sync, `BaseMetric.a_measure()` is async (defaults to calling `measure()`)
- `evaluate()`, `assert_test()`, `evaluate_cases()` remain sync
- `evaluate_dataset()` is async (because `agent_fn` is async)
- Phase 2 adds `a_evaluate()` that uses `asyncio.gather()` on `a_measure()` for concurrent LLM calls
- pytest-asyncio is a dev dependency (for `evaluate_dataset()` tests)
