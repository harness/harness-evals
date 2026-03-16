# ADR-006: Async-first for LLM providers

## Status

Accepted (Phase 2)

## Context

LLM-judged metrics (GEval, Faithfulness, etc.) need to call external APIs. These calls are I/O-bound and high-latency (1–30 seconds). When evaluating a dataset of 100 test cases with 3 LLM-judged metrics, that's 300 API calls.

## Decision

`BaseLLM` methods are `async def`. Sync usage is supported via `asyncio.run()` wrappers.

## Rationale

1. **Concurrency is critical for datasets** — Sequential LLM calls for 300 evaluations would take ~30 minutes. With `asyncio.gather()`, we can run 10–50 concurrent calls and finish in 1–3 minutes.

2. **API clients are async** — Both `openai>=1.0` and `anthropic>=0.30` provide native async clients. Using sync wrappers would waste the capability.

3. **Backward compatible** — Users who don't use LLM metrics never encounter async. Phase 1 metrics are all synchronous. The `evaluate()` function remains synchronous. Only `evaluate_dataset()` with LLM metrics uses async internally.

4. **Simple sync wrapper** — For users who don't want to deal with async:

```python
import asyncio
result = asyncio.run(llm.generate("prompt"))
```

Or we provide a sync convenience:

```python
class OpenAILLM(BaseLLM):
    def generate_sync(self, prompt: str) -> str:
        return asyncio.run(self.generate(prompt))
```

## Trade-offs

- Added complexity for contributors writing LLM-judged metrics. They must use `async def` and `await`. Mitigation: clear examples in metrics-guide.md.
- `asyncio.run()` cannot be called from within an existing event loop (e.g., Jupyter). Mitigation: detect and use `nest_asyncio` or `loop.run_until_complete()`.
- Testing requires `pytest-asyncio`. Already included in dev dependencies.

## Consequences

- `BaseLLM.generate()` and `generate_json()` are async
- `evaluate()` and `assert_test()` remain sync for Phase 1 metrics
- `evaluate_dataset()` with LLM metrics uses async internally for concurrency
- pytest-asyncio is a dev dependency
