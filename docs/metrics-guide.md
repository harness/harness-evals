# Metrics Authoring Guide

## The Pattern

Every metric is a single class in a single file. It extends `BaseMetric`, implements `measure()`, and returns a `Score`. That's it.

```python
from harness_evals.core.metric import BaseMetric
from harness_evals.core.score import Score
from harness_evals.core.eval_case import EvalCase


class MyMetric(BaseMetric):
    def __init__(self, threshold: float = 1.0, **kwargs) -> None:
        super().__init__(name="my_metric", threshold=threshold, **kwargs)

    def measure(self, eval_case: EvalCase) -> Score:
        value = ...  # compute 0.0–1.0
        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
        )
```

`Score.passed` is auto-computed from `value >= threshold` — never set it manually.

## Step by Step

### 1. Choose the Category

| Category | When to Use | Base Class |
|----------|------------|-----------|
| `deterministic/` | Exact comparison, regex, numeric | `BaseMetric` |
| `structural/` | JSON/YAML diff, schema validation | `BaseMetric` |
| `operational/` | Latency, cost, tokens, retries | `BaseMetric` |
| `reliability/` | Multi-run consistency, robustness | `ReliabilityMetric` |
| `llm_judge/` | LLM scores against criteria | `BaseMetric` (takes `llm` param) |
| `rag/` | Faithfulness, relevancy, context | `BaseMetric` (takes `llm` param) |
| `safety/` | PII, toxicity, injection, hallucination | `BaseMetric` |
| `agent/` | Tool correctness, task completion | `BaseMetric` |
| `conversation/` | Multi-turn coherence, resolution | `BaseMetric` |
| `mcp/` | Tool selection, trace completeness | `BaseMetric` |

### 2. Create the File

```
src/harness_evals/metrics/<category>/<metric_name>.py
```

### 3. Implement the Class

#### Deterministic Metric Template

For metrics that compare output vs expected:

```python
class MyDeterministicMetric(BaseMetric):
    def __init__(self, threshold: float = 1.0, **kwargs) -> None:
        super().__init__(name="my_metric", threshold=threshold, **kwargs)

    def measure(self, eval_case: EvalCase) -> Score:
        actual = str(eval_case.output)
        expected = str(eval_case.expected)

        value = 1.0 if actual == expected else 0.0

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
        )
```

#### Operational Metric Template

For metrics that read typed fields from `EvalCase`:

```python
class MyOperationalMetric(BaseMetric):
    def __init__(self, max_value: float = 100, threshold: float = 0.5, **kwargs) -> None:
        super().__init__(name="my_metric", threshold=threshold, **kwargs)
        self.max_value = max_value

    def measure(self, eval_case: EvalCase) -> Score:
        if eval_case.latency_ms is None:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason="latency_ms not provided",
            )

        value = max(0.0, 1.0 - eval_case.latency_ms / self.max_value)

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            metadata={"latency_ms": eval_case.latency_ms, "max_value": self.max_value},
        )
```

#### Reliability Metric Template

For metrics that evaluate across multiple runs:

```python
from harness_evals.core.metric import ReliabilityMetric

class MyReliabilityMetric(ReliabilityMetric):
    def __init__(self, threshold: float = 0.8, k: int = 5, **kwargs) -> None:
        super().__init__(name="my_metric", threshold=threshold, k=k, **kwargs)

    def measure_runs(self, eval_case: EvalCase) -> Score:
        runs = eval_case.runs or []
        if len(runs) < 2:
            return Score(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                reason=f"Need at least 2 runs, got {len(runs)}",
            )

        value = ...  # compute consistency/variance across runs

        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
            metadata={"k": len(runs)},
        )
```

#### LLM-Judged Metric Template (Phase 2+)

For metrics that use an LLM as a judge — override `a_measure()`:

```python
from harness_evals.llm.base import BaseLLM

class MyLLMMetric(BaseMetric):
    def __init__(self, threshold: float = 0.7, llm: BaseLLM | None = None, **kwargs) -> None:
        super().__init__(name="my_metric", threshold=threshold, **kwargs)
        self.llm = llm

    def measure(self, eval_case: EvalCase) -> Score:
        if self.llm is None:
            return Score(name=self.name, value=0.0, threshold=self.threshold,
                         reason="No LLM provider configured")
        import asyncio
        return asyncio.run(self.a_measure(eval_case))

    async def a_measure(self, eval_case: EvalCase) -> Score:
        prompt = f"Rate the following response...\nInput: {eval_case.input}\nOutput: {eval_case.output}"
        result = await self.llm.generate_json(prompt, schema={"score": "number"})
        value = result["score"] / 10.0
        return Score(
            name=self.name,
            value=value,
            threshold=self.threshold,
        )
```

### 4. Export the Metric

Add to `src/harness_evals/metrics/<category>/__init__.py`:

```python
from harness_evals.metrics.<category>.<metric_name> import MyMetric
```

Add to `src/harness_evals/metrics/__init__.py`:

```python
from harness_evals.metrics.<category> import MyMetric
```

### 5. Write Tests

Create `tests/metrics/test_<metric_name>.py`:

```python
import pytest
from harness_evals.core.eval_case import EvalCase
from harness_evals.metrics.<category> import MyMetric


@pytest.mark.unit
class TestMyMetric:
    def test_perfect_score(self):
        ec = EvalCase(input="q", output="expected", expected="expected")
        score = MyMetric(threshold=0.8).measure(ec)
        assert score.passed
        assert score.value == 1.0

    def test_failure(self):
        ec = EvalCase(input="q", output="wrong", expected="expected")
        score = MyMetric(threshold=0.8).measure(ec)
        assert not score.passed
        assert score.value < 0.8

    def test_edge_case(self):
        ec = EvalCase(input="", output="")
        score = MyMetric().measure(ec)
        assert isinstance(score.value, float)
```

### 6. Run Tests

```bash
ruff check src/ tests/          # lint
ruff format --check src/ tests/ # format
pytest tests/ -v                # test
```

## Rules

1. **One metric per file** — keeps PRs small and reviewable.
2. **Score is always [0.0, 1.0]** — normalize whatever you compute. Put raw values in `Score.metadata`.
3. **Never raise from `measure()`** — return a failing Score with a `reason` instead. If you do raise, `evaluate()` catches it, but explicit is better.
4. **Handle missing data gracefully** — operational metrics should check typed fields and return a failing Score with a clear reason if None.
5. **No global state** — all configuration goes in `__init__()`. Metrics are reusable across eval cases.
6. **No cross-metric imports** — metrics should not import from other metrics.
7. **Safety metrics are hard constraints** — see [ADR-003](adr/003-safety-never-averaged.md).
8. **Don't set `passed` manually** — `Score` auto-computes `passed = value >= threshold` in `__post_init__`.
